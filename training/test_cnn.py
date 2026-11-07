"""Fast tests for the cnn-v1 model, its data stream and its metric."""
import json

import numpy as np
import pytest
import torch
from baseline import evaluate
from model import GestureCNN, parameter_count
from train_data import TrainStream, apply_shift, augment_batch

from radarsense_dsp.windows import (
    BalancedUserSampler,
    build_window_index,
    load_complete_members,
    window_class_at,
)

SPLITS = {"train": ["user1", "user2"], "val": ["user3"], "test": ["user10", "user12"]}


def make_cache(tmp_path, spec: dict[str, tuple[int, list[tuple[int, int, int]]]]):
    """Write a tiny cache. The spec maps a stem to (user, [(label, s, L), ...]).

    Channel 0 of the pixel (0, 0) of each frame stores the frame number divided by 100.
    """
    cache = tmp_path / "processed"
    cache.mkdir()
    index = {"version": 1, "files": {}}
    for stem, (user, spans) in spec.items():
        targets = np.zeros((len(spans), 100), dtype=np.int8)
        for r, (label, s, length) in enumerate(spans):
            targets[r, s : s + length] = label
        frames = np.zeros((len(spans), 100, 3, 32, 32), dtype=np.float16)
        frames[:, :, 0, 0, 0] = np.arange(100) / 100
        np.save(cache / f"{stem}.targets.npy", targets)
        np.save(cache / f"{stem}.npy", frames)
        index["files"][stem] = {"user": user, "session": 1, "anomaly": None, "n": len(spans), "status": "complete"}
    (cache / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return cache


def test_model_output_shape_and_parameter_count():
    model = GestureCNN()
    out = model(torch.zeros(2, 32, 3, 32, 32))
    assert out.shape == (2, 6) and out.dtype == torch.float32
    assert 60_000 <= parameter_count(model) <= 70_000


def test_sampler_never_yields_a_test_user(tmp_path):
    spans = [(1, 40, 8), (2, 40, 8), (3, 40, 8), (6, 40, 8), (7, 40, 8)]
    spec = {"user1_e1": (1, spans), "user2_e1": (2, spans), "user3_e1": (3, spans)}
    cache = make_cache(tmp_path, spec)
    members = load_complete_members(cache, SPLITS)
    train = build_window_index(cache, members).split_rows("train")
    sampler = BalancedUserSampler(train["user"], per_user=30, seed=0, classes=train["cls"])
    for epoch in range(5):
        assert set(train["user"][sampler.epoch_indices(epoch)]) <= {1, 2}
    # A cache that contains a test user is refused before any window is indexed.
    (tmp_path / "bad").mkdir()
    bad = make_cache(tmp_path / "bad", spec | {"user10_e1": (10, spans)})
    with pytest.raises(PermissionError):
        load_complete_members(bad, SPLITS)


def test_shifted_window_is_relabelled_by_the_window_rule():
    # The gesture s=40, L=8 has the extended span [30, 47]. A gesture window starts at 16 or later (and at 30 or earlier).
    label, s, length = 3, 40, 8
    start = 16
    cls = window_class_at(label, s, length, start)
    assert cls == 3
    assert apply_shift(label, s, length, start, cls, +3) == (19, window_class_at(label, s, length, 19)) == (19, 3)
    # Shift -3 gives the start 13 and the window [13, 44], which misses the end of the span and is excluded, so the
    # unshifted window is kept.
    assert window_class_at(label, s, length, 13) is None
    assert apply_shift(label, s, length, start, cls, -3) == (start, cls)
    # A no-gesture window stays no-gesture when the shifted window is still far from the gesture ...
    assert apply_shift(label, 70, length, 4, 0, +3) == (7, 0)
    # ... and falls back when the shifted window comes too close (start 12 ends at frame 43, the guard starts at 50).
    assert window_class_at(label, 70, length, 12) == 0 and window_class_at(label, 70, length, 19) is None
    assert apply_shift(label, 70, length, 16, 0, +3) == (16, 0)
    # A shift that leaves the recording is ignored, below frame 0 and above the last start 68.
    assert apply_shift(label, 5, length, 0, 3, -3) == (0, 3)
    assert apply_shift(label, 88, length, 68, 3, +3) == (68, 3)
    assert apply_shift(label, 88, length, 68, 3, -3) == (65, 3)  # The same window shifted inwards is valid.


def test_stream_returns_the_shifted_window_with_the_rule_label(tmp_path):
    cache = make_cache(tmp_path, {"user1_e1": (1, [(2, 40, 8)] * 6), "user2_e1": (2, [(6, 30, 8)] * 6)})
    members = load_complete_members(cache, SPLITS)
    rows = build_window_index(cache, members).split_rows("train")
    sampler = BalancedUserSampler(rows["user"], per_user=16, seed=1, classes=rows["cls"])
    stream = TrainStream(cache, members, rows, sampler, seed=1, epoch_len=32, n_epochs=2)
    idx, shifts = stream.plan(1)
    for pos in range(32):
        window, cls = stream.item(1, pos)
        row = rows[idx[pos]]
        label, s, length = (2, 40, 8) if row["user"] == 1 else (6, 30, 8)
        first = round(float(window[0, 0, 0, 0]) * 100)  # The stored frame number of the first frame.
        assert window.shape == (32, 3, 32, 32)
        assert first - int(row["start"]) in (0, int(shifts[pos]))
        assert cls == window_class_at(label, s, length, first)


def test_augmentation_keeps_zero_cells_and_scales_phase_channels_with_the_magnitude():
    x = torch.zeros(4, 3, 3, 4, 4)
    x[:, :, 0, 1, 1] = 0.5
    x[:, :, 1, 1, 1] = 0.25  # sin(phase difference) times the magnitude, so the ratio to the magnitude is 0.5
    torch.manual_seed(0)
    out = augment_batch(x, db_range=30.0, max_db=3.0, noise_std=0.0)
    assert (out[:, :, :, 0, 0] == 0).all()  # Cells without a reflection stay at 0.
    mag, phase = out[:, :, 0, 1, 1], out[:, :, 1, 1, 1]
    assert torch.allclose(phase, mag * 0.5, atol=1e-6)
    assert ((mag - 0.5).abs() <= 3.0 / 30.0 + 1e-6).all() and (mag != 0.5).any()
    assert torch.equal(augment_batch(x, 30.0, max_db=0.0, noise_std=0.0), x)  # Without gain and noise nothing changes.


def test_macro_f1_on_a_tiny_confusion_matrix():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    res = evaluate(y_true, y_pred)
    assert res["confusion"][:3, :3].tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    # F1 = 2 * tp / (support + predicted): class 0 is 4/6, class 1 is 4/5, class 2 is 0. Classes 3 to 5 are absent.
    assert res["macro_f1"] == pytest.approx((4 / 6 + 4 / 5 + 0) / 3)
    assert res["accuracy"] == pytest.approx(4 / 6)
