"""Tests for the export scripts that do not need LiteRT: BatchNorm folding, window selection, the wire round trip,
the metrics and the model card."""
import json

import export_common as ec
import export_prepare as ep
import numpy as np
import pytest
import torch
from baseline import evaluate
from model import GestureCNN, fold_batchnorm
from test_cnn import SPLITS, make_cache

from radarsense_dsp.windows import (
    CLASS_NAMES,
    CacheReader,
    build_window_index,
    load_complete_members,
)

SPANS = [(label, 40, 8) for label in (1, 2, 3, 6, 7) for _ in range(8)]  # 5 gestures x 8 recordings


def trained_like_model() -> GestureCNN:
    """A model whose BatchNorm layers have non-trivial statistics and affine parameters."""
    torch.manual_seed(0)
    m = GestureCNN()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.normal_(0, 0.3)
    return m.eval()


def test_fold_batchnorm_keeps_the_output_and_removes_the_batchnorm_layers():
    m = trained_like_model()
    x = torch.rand(4, 32, 3, 32, 32) * 2 - 1
    with torch.no_grad():
        before = m(x)
        folded = fold_batchnorm(m)
        after = folded(x)
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in folded.modules())
    assert torch.allclose(before, after, atol=1e-4), (before - after).abs().max()


def test_fold_batchnorm_refuses_a_model_in_train_mode():
    with pytest.raises(ValueError):
        fold_batchnorm(GestureCNN().train())


@pytest.fixture
def cache(tmp_path):
    return make_cache(tmp_path, {"user1_e1": (1, SPANS), "user2_e1": (2, SPANS), "user3_e1": (3, SPANS)})


def train_rows(cache):
    return build_window_index(cache, load_complete_members(cache, SPLITS)).split_rows("train")


def test_select_balanced_has_equal_classes_and_is_deterministic(cache):
    rows = train_rows(cache)
    a, b = ec.select_balanced(rows, 62, seed=0), ec.select_balanced(rows, 62, seed=0)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 62
    counts = np.bincount(rows["cls"][a], minlength=6)
    assert counts.tolist() == [11, 11, 10, 10, 10, 10]  # n % 6 = 2 extra windows go to the first classes
    assert not np.array_equal(a, ec.select_balanced(rows, 62, seed=1))
    for c in range(6):  # both train users take part in every class
        assert set(rows["user"][a][rows["cls"][a] == c]) == {1, 2}


def test_select_balanced_refuses_test_users_and_missing_classes(cache):
    rows = train_rows(cache)
    poisoned = rows.copy()
    poisoned["user"][0] = 10
    with pytest.raises(PermissionError):
        ec.select_balanced(poisoned, 12, seed=0)
    with pytest.raises(ValueError):
        ec.select_balanced(rows, 6 * 10_000, seed=0)


def test_prepare_uses_train_users_for_calibration_and_validation_users_for_evaluation(cache):
    data, digest = ep.prepare(cache, SPLITS, n_calib=30, n_eval=30, seed=0)
    assert data["calib_u8"].shape == (30, 32, 3, 32, 32) and data["calib_u8"].dtype == np.uint8
    assert data["eval_u8"].shape == (30, 32, 3, 32, 32)
    assert set(data["calib_users"].tolist()) == {1, 2}
    assert set(data["eval_users"].tolist()) == {3}
    assert np.bincount(data["eval_classes"], minlength=6).tolist() == [5] * 6
    assert len(digest) == 16
    # Every window equals the wire round trip of the cached window.
    reader = CacheReader(cache, load_complete_members(cache, SPLITS))
    member, recording, start = data["eval_rows"][0]
    window = reader.window({"member": member, "recording": recording, "start": start}).astype(np.float32)
    assert np.array_equal(data["eval_u8"][0], ec.wire_roundtrip(window[None])[0][0])


def test_prepare_refuses_a_cache_with_a_test_user(tmp_path):
    spec = {"user1_e1": (1, SPANS), "user3_e1": (3, SPANS), "user10_e1": (10, SPANS)}
    with pytest.raises(PermissionError):
        ep.prepare(make_cache(tmp_path, spec), SPLITS, n_calib=12, n_eval=12)


def test_main_writes_inputs_checkpoint_copy_and_stamp(cache, tmp_path):
    ckpt = {
        "model_state": GestureCNN().state_dict(), "model": "cnn-v1", "epoch": 7, "val_macro_f1": 0.95,
        "commit": "abc123", "cache_id": "deadbeef", "config": {"run_name": "cnn-v1-test", "max_epochs": 12},
    }
    run = tmp_path / "checkpoints" / "cnn-v1-test"
    run.mkdir(parents=True)
    torch.save(ckpt, run / "best.pt")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(SPLITS), encoding="utf-8")
    out = tmp_path / "models" / "cnn-v1-test"
    args = ["--checkpoint", str(run / "best.pt"), "--cache", str(cache), "--splits", str(splits), "--out-dir", str(out)]
    assert ep.main([*args, "--n-calib", "24", "--n-eval", "24"]) == 0
    stamp = json.loads((out / ec.STAMP_NAME).read_text())
    assert stamp["version"] == "cnn-v1-test" and stamp["checkpoint_commit"] == "abc123" and stamp["epoch"] == 7
    assert stamp["calib_users"] == ["user1", "user2"]
    assert (out / "best.pt").exists() and (out / ec.INPUTS_NAME).exists()
    with np.load(out / ec.INPUTS_NAME) as z:
        assert z["eval_u8"].dtype == np.uint8 and set(z["eval_users"].tolist()) == {3}


def test_wire_roundtrip_error_is_at_most_one_step():
    rng = np.random.default_rng(0)
    w = np.empty((3, 32, 3, 32, 32), dtype=np.float32)
    w[:, :, 0] = rng.random((3, 32, 32, 32))
    w[:, :, 1:] = rng.uniform(-1, 1, (3, 32, 2, 32, 32))
    u8, dec = ec.wire_roundtrip(w)
    assert u8.dtype == np.uint8 and dec.dtype == np.float32
    assert np.abs(dec[:, :, 0] - w[:, :, 0]).max() <= 1 / 255
    assert np.abs(dec[:, :, 1:] - w[:, :, 1:]).max() <= 2 / 255


def test_macro_f1_matches_baseline_evaluate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, 400)
    pred = np.where(rng.random(400) < 0.7, y, rng.integers(0, 6, 400))
    ref = evaluate(y, pred)
    assert ec.macro_f1(y, pred) == pytest.approx(ref["macro_f1"])
    assert ec.accuracy(y, pred) == pytest.approx(ref["accuracy"])
    assert ec.per_class_recall(y, pred) == pytest.approx(ref["recall"])
    only_some = np.array([0, 0, 1, 1])  # classes that do not occur are left out of the average
    assert ec.macro_f1(only_some, only_some) == 1.0


def test_agreement_and_int8_fallback_rule():
    a = np.eye(6)[[0, 1, 2, 3]]
    b = np.eye(6)[[0, 1, 2, 4]]
    assert ec.top1_agreement(a, b) == 0.75
    assert not ec.int8_needs_fallback(0.96, 0.955)
    assert ec.int8_needs_fallback(0.96, 0.9499)  # more than 1 point below


def test_model_card_fields():
    constants = json.loads((ec.ROOT / "dsp" / "radarsense_dsp" / "constants.json").read_text())
    stamp = {
        "model": "cnn-v1", "n_eval": 1000, "seed": 0, "epoch": 6, "val_macro_f1": 0.96,
        "checkpoint_commit": "abc", "cache_id": "cid", "prepared_at_commit": "def",
    }
    m = {"accuracy": 0.9, "macro_f1": 0.9, "recall": [0.9] * 6}
    card = ec.build_model_card(
        "cnn-v1-e12", stamp, constants, {"pytorch": m, "fp32": m, "int8": m}, {"fp32": 1, "int8": 1},
        {"pytorch_vs_fp32": 1.0}, None,
    )
    assert card["version"] == "cnn-v1-e12"
    assert card["input"]["shape"] == [1, 32, 3, 32, 32] and card["input"]["wire_dtype"] == "uint8"
    assert card["classes"] == list(CLASS_NAMES)
    assert card["uint8_decode"]["channel_0"]["formula"] == "x = u / 255"
    assert card["uint8_decode"]["channel_1"]["formula"] == "x = u / 255 * 2 - 1"
    assert card["db_range"]["db_min"] == constants["db_min"] and card["db_range"]["db_max"] == constants["db_max"]
    assert card["db_range"]["source_commit"] == constants["git_commit"]
    assert card["dataset"]["doi"] == "10.5281/zenodo.15178095"
    assert card["test"]["macro_f1"] is None and card["test"]["accuracy"] is None
    assert card["validation"]["users"] == ["user3", "user5"] and card["validation"]["models"]["int8"] == m
    assert card["stamp"] == {"checkpoint_commit": "abc", "cache_id": "cid", "prepared_at_commit": "def"}
    assert card["fallback"] is None
    json.dumps(card)  # serializable


def test_convert_litert_has_no_windows_or_cuda_dependency_at_import_time():
    import ast
    from pathlib import Path

    source = (Path(ec.ROOT) / "training" / "convert_litert.py").read_text(encoding="utf-8")
    top = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.parse(source).body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    # litert_torch, torchao and ai_edge_quantizer are imported inside the functions, so the module imports without them.
    assert top == {"argparse", "sys", "pathlib", "numpy", "torch", "model", "replay"}


def test_convert_litert_loads_the_checkpoint_and_decodes_the_calibration_windows(tmp_path):
    import convert_litert as cl

    torch.manual_seed(0)
    model = trained_like_model()
    torch.save({"model_state": model.state_dict(), "model": "cnn-v1"}, tmp_path / "best.pt")
    loaded = cl.load_model(tmp_path / "best.pt")
    x = torch.rand(2, 32, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
    assert not loaded.training
    with pytest.raises(SystemExit):
        torch.save({"model_state": model.state_dict(), "model": "other"}, tmp_path / "other.pt")
        cl.load_model(tmp_path / "other.pt")


# ---- export_check: the parts that do not need LiteRT ---------------------------------------------------------------

import export_check as ck


def test_quantize_input_and_dequantize_output_round_trip():
    x = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
    q = ck.quantize_input(x, np.dtype(np.int8), 1 / 127, 0)
    assert q.dtype == np.int8 and q.tolist() == [-127, 0, 64, 127]
    assert np.allclose(ck.dequantize_output(q, 1 / 127, 0), x, atol=1 / 127)
    assert np.array_equal(ck.quantize_input(x, np.dtype(np.float32), 0.0, 0), x)  # a float model gets x unchanged
    assert ck.quantize_input(np.array([5.0]), np.dtype(np.int8), 1 / 127, 0).tolist() == [127]  # clipped
    u = ck.quantize_input(np.array([0.0]), np.dtype(np.uint8), 0.5, 128)
    assert u.dtype == np.uint8 and u.tolist() == [128]
    f = np.array([1.5], dtype=np.float32)
    assert ck.dequantize_output(f, 0.0, 0).tolist() == [1.5]


def synthetic_logits(n=120, flip=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 6
    pred = np.where(rng.random(n) < flip, (y + 1) % 6, y)
    return y, np.eye(6)[pred] * 5.0


def test_summarize_accepts_int8_within_one_point_and_flags_float16_beyond_it():
    y, good = synthetic_logits()
    _, noisy = synthetic_logits(flip=0.05, seed=1)
    metrics, agreement, fallback = ck.summarize(y, {"pytorch": good, "fp32": good, "int8": good})
    assert fallback is None and agreement["pytorch_vs_int8"] == 1.0 and metrics["int8"]["macro_f1"] == 1.0
    metrics, agreement, fallback = ck.summarize(y, {"pytorch": good, "fp32": good, "int8": noisy})
    assert fallback == "float16" and agreement["fp32_vs_int8"] < 1.0
    assert metrics["fp32"]["macro_f1"] - metrics["int8"]["macro_f1"] > 0.01
    _, agreement, _ = ck.summarize(y, {"pytorch": good, "fp32": good, "int8": good, "fp16": good})
    assert agreement["fp32_vs_fp16"] == 1.0


def test_export_check_reports_missing_files_instead_of_crashing(tmp_path, capsys):
    assert ck.main(["--dir", str(tmp_path)]) == 2
    assert "model_fp32.tflite" in capsys.readouterr().err
