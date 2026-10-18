import json
from pathlib import Path

S = json.loads((Path(__file__).parent / "splits.json").read_text(encoding="utf-8"))


def test_splits_are_disjoint_and_complete():
    train, val, test = set(S["train"]), set(S["val"]), set(S["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == {f"user{i}" for i in range(1, 13)}
