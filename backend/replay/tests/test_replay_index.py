import json

import numpy as np
import pytest
from conftest import MEMBERS, SPLITS, write_cache

from radarsense_dsp.windows import CLASS_NAMES
from replay.index import ReplayError, ReplayIndex


@pytest.fixture(scope="module")
def index(cache_dir):
    return ReplayIndex(cache_dir, SPLITS)


def test_only_complete_nominal_members_are_indexed(index):
    assert {m.stem for m in index.members} == {"user1_e1", "user2_e1", "user3_e1", "user5_e1"}
    assert index.users == [1, 2, 3, 5]


def test_counts_and_labels_come_from_the_targets(index):
    assert index.recording_counts == {(1, 1): 2, (2, 1): 1, (3, 1): 4, (5, 1): 3}
    assert len(index) == 10
    user3 = index.recs[index.recs["user"] == 3]
    assert user3["label"].tolist() == [2, 3, 6, 1]
    assert user3["cls"].tolist() == [2, 3, 4, 1]
    assert user3["s"].tolist() == [40, 45, 35, 50]
    assert set(user3["length"].tolist()) == {8}


def test_a_violation_file_is_ignored_even_if_its_files_exist(tmp_path):
    cache = write_cache(tmp_path / "c", MEMBERS)
    # Put real data behind the violation entry. The status still decides.
    for suffix in (".npy", ".targets.npy"):
        (cache / f"user4_e1{suffix}").write_bytes((cache / f"user3_e1{suffix}").read_bytes())
    assert 4 not in ReplayIndex(cache, SPLITS).users


def test_a_partial_status_or_missing_files_are_ignored(tmp_path):
    cache = write_cache(tmp_path / "c", MEMBERS)
    idx = json.loads((cache / "index.json").read_text())
    idx["files"]["user2_e1"]["status"] = "partial"
    (cache / "index.json").write_text(json.dumps(idx))
    (cache / "user1_e1.npy").unlink()
    assert ReplayIndex(cache, SPLITS).users == [3, 5]


def test_a_test_user_in_the_index_is_refused(tmp_path):
    members = dict(MEMBERS)
    members["user10_e1"] = (10, 1, None, "complete", [(2, 40, 8)])
    with pytest.raises(PermissionError):
        ReplayIndex(write_cache(tmp_path / "c", members), SPLITS)


def test_default_users_are_the_validation_users(index):
    rng = np.random.default_rng(0)
    users = {int(index.pick(rng, "any", "any")[0]["user"]) for _ in range(200)}
    assert users == {3, 5}


def test_train_users_only_when_named(index):
    rng = np.random.default_rng(0)
    assert {int(index.pick(rng, "user1", "any")[0]["user"]) for _ in range(20)} == {1}
    assert {int(index.pick(rng, "user2", "any")[0]["user"]) for _ in range(20)} == {2}


@pytest.mark.parametrize("user", ["user10", "user12", "user99", "10", "USER3", "user3; x", "", "user"])
def test_unknown_and_test_users_are_errors(index, user):
    with pytest.raises(ReplayError):
        index.pick(np.random.default_rng(0), user, "any")


def test_gesture_filter_and_unknown_gesture(index):
    rng = np.random.default_rng(0)
    for _ in range(30):
        rec, background = index.pick(rng, "any", CLASS_NAMES[2])
        assert rec["cls"] == 2 and not background
    with pytest.raises(ReplayError):
        index.pick(rng, "any", "wave")
    with pytest.raises(ReplayError):
        index.pick(rng, "any", "none")
    with pytest.raises(ReplayError):  # user3 and user5 differ: user2 has no class-2 recording
        index.pick(rng, "user2", CLASS_NAMES[2])


def test_recordings_without_background_frames_are_not_picked_for_background(index):
    rng = np.random.default_rng(0)
    for _ in range(200):
        rec, background = index.pick(rng, "user5", "background")
        # The recording (label 1, s=10, L=80) has no frame outside its zone [-20, 99].
        assert background and rec["s"] != 10
