"""Tests for write_export_report.py with a small fake model card and commit stamp."""
import copy
import json

import pytest
import write_export_report as wr

CARD = {
    "version": "cnn-v9",
    "model": "cnn-v1",
    "input": {"shape": [1, 32, 3, 32, 32], "layout": "(batch, frame, channel, range, Doppler)", "dtype": "float32"},
    "output": {"shape": [1, 6], "kind": "logits"},
    "classes": ["none", "label1", "label2", "label3", "label6", "label7"],
    "uint8_decode": {
        "channel_0": {"formula": "x = u / 255"},
        "channel_1": {"formula": "x = u / 255 * 2 - 1"},
        "channel_2": {"formula": "x = u / 255 * 2 - 1"},
    },
    "db_range": {
        "db_min": 11.11, "db_max": 77.77, "pair_a": [0, 2], "pair_b": [1, 2], "source": "constants.json",
        "source_commit": "c0ffee", "source_users": [1, 2],
    },
    "dataset": {"doi": "10.5281/zenodo.15178095", "license": "CC BY 4.0"},
    "validation": {
        "users": ["user3", "user5"],
        "windows": 1000,
        "selection": "class-balanced, seed 0, after the uint8 wire round trip",
        "checkpoint_epoch": 4,
        "checkpoint_val_macro_f1": 0.9448,
        "models": {
            "pytorch": {"accuracy": 0.9111, "macro_f1": 0.9222},
            "fp32": {"accuracy": 0.9111, "macro_f1": 0.9222},
            "int8": {"accuracy": 0.9333, "macro_f1": 0.9444},
        },
        "top1_agreement": {"pytorch_vs_fp32": 1.0, "pytorch_vs_int8": 0.9876, "fp32_vs_int8": 0.9876},
    },
    "test": {"macro_f1": None, "accuracy": None, "note": "Not evaluated before Week 11."},
    "sizes_bytes": {"pytorch": 2048, "fp32": 4096, "int8": 1536},
    "fallback": None,
}
STAMP = {
    "version": "cnn-v9",
    "checkpoint_commit": "aaaa1111",
    "cache_id": "bbbb2222",
    "prepared_at_commit": "cccc3333+dirty",
    "eval_users": ["user3", "user5"],
    "eval_class_counts": [167, 167, 167, 167, 166, 166],
    "sha256": {"best.pt": "deadbeef", "export_inputs.npz": "feedface"},
    "config": {"run_name": "cnn-v9-run", "seed": 7, "parameters": 12345},
}
HEAD = "0123456789abcdef0123456789abcdef01234567"
SECTIONS = ["## Model", "## Results on the evaluation windows", "## Agreement and the INT8 check", "## Evaluation set",
            "## Normalization and dataset", "## Where the files are", "## Fields not recorded"]


def write_inputs(tmp_path, card=CARD, stamp=STAMP):
    d = tmp_path / "models" / "cnn-v9"
    d.mkdir(parents=True)
    (d / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    (d / "commit_stamp.json").write_text(json.dumps(stamp), encoding="utf-8")
    return d


def test_report_has_every_section_and_the_first_line_names_head():
    report, missing = wr.render(CARD, STAMP, HEAD, "data/models/cnn-v9")
    for s in SECTIONS:
        assert s in report
    assert (f"This record was produced at commit {HEAD} from model_card.json and commit_stamp.json. "
            "Nothing was recomputed.") in report.splitlines()[2]
    assert missing == []
    assert "None. Every field was present" in report


def test_numbers_are_copied_exactly():
    report, _ = wr.render(CARD, STAMP, HEAD, "data/models/cnn-v9")
    assert "| PyTorch | 0.9111 | 0.9222 | 2.0 KB |" in report
    assert "| LiteRT FP32 | 0.9111 | 0.9222 | 4.0 KB |" in report
    assert "| LiteRT INT8 | 0.9333 | 0.9444 | 1.5 KB |" in report
    assert "float16" not in report.split("## Agreement")[0]  # no row for a model that is not in the card
    assert "top-1 agreement of LiteRT INT8 with PyTorch: 0.9876" in report
    assert "top-1 agreement of LiteRT FP32 with PyTorch: 1.0000" in report
    assert "-2.22 points" in report and "INT8 passed the check." in report  # (0.9222 - 0.9444) * 100
    for expected in ("aaaa1111", "bbbb2222", "cccc3333+dirty", "cnn-v9-run", "12345", "11.11", "77.77", "c0ffee",
                     "10.5281/zenodo.15178095", "deadbeef", "feedface", "user3, user5", "[167, 167, 167, 167, 166, 166]",
                     "x = u / 255 * 2 - 1", "none, label1, label2, label3, label6, label7", "training seed: 7"):
        assert expected in report
    assert "user10 and user12 were not used" in report
    assert "not yet verified" in report


def test_a_failed_int8_check_names_float16_as_the_fallback():
    card = copy.deepcopy(CARD)
    card["validation"]["models"]["int8"]["macro_f1"] = 0.90
    card["validation"]["models"]["fp16"] = {"accuracy": 0.9, "macro_f1": 0.9201}
    card["sizes_bytes"]["fp16"] = 1024
    card["validation"]["top1_agreement"]["pytorch_vs_fp16"] = 0.99
    card["fallback"] = "float16"
    report, _ = wr.render(card, STAMP, HEAD, "x")
    assert "+2.22 points" in report and "INT8 failed the check, so float16 is the fallback." in report
    assert "| LiteRT float16 | 0.9000 | 0.9201 | 1.0 KB |" in report
    assert "The card records the fallback as float16." in report
    assert "do not agree" not in report


def test_a_missing_field_is_written_as_not_recorded_and_listed():
    card, stamp = copy.deepcopy(CARD), copy.deepcopy(STAMP)
    del card["validation"]["checkpoint_epoch"]
    del stamp["config"]["parameters"]
    del card["sizes_bytes"]["int8"]
    del card["validation"]["top1_agreement"]["pytorch_vs_int8"]
    report, missing = wr.render(card, stamp, HEAD, "x")
    assert "- best epoch: not recorded" in report and "- parameters: not recorded" in report
    assert "| LiteRT INT8 | 0.9333 | 0.9444 | not recorded |" in report
    assert "top-1 agreement of LiteRT INT8 with PyTorch: not recorded" in report
    for field in ("card.validation.checkpoint_epoch", "stamp.config.parameters", "card.sizes_bytes.int8",
                  "card.validation.top1_agreement.pytorch_vs_int8"):
        assert field in missing and f"`{field}`" in report


def test_a_missing_model_row_is_not_invented():
    card = copy.deepcopy(CARD)
    del card["validation"]["models"]["int8"]
    report, missing = wr.render(card, STAMP, HEAD, "x")
    assert "| LiteRT INT8 | not recorded | not recorded | not recorded |" in report
    assert "INT8 drop against FP32: not recorded" in report and "card.validation.models.int8" in missing


@pytest.mark.parametrize("where", ["card", "stamp"])
@pytest.mark.parametrize("user", ["user10", "user12"])
def test_the_report_refuses_test_users(where, user):
    card, stamp = copy.deepcopy(CARD), copy.deepcopy(STAMP)
    (card["validation"] if where == "card" else stamp)["users" if where == "card" else "eval_users"].append(user)
    with pytest.raises(SystemExit) as e:
        wr.render(card, stamp, HEAD, "x")
    assert user in str(e.value) and "test users" in str(e.value)


def test_main_writes_the_file_and_refuses_test_users_without_writing(tmp_path, capsys):
    d, out = write_inputs(tmp_path), tmp_path / "out" / "report.md"
    assert wr.main(["--dir", str(d), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Export record: cnn-v9")
    assert "Missing fields: none" in capsys.readouterr().out
    bad = copy.deepcopy(CARD)
    bad["validation"]["users"] = ["user3", "user10"]
    d2 = tmp_path / "models" / "bad"
    d2.mkdir()
    (d2 / "model_card.json").write_text(json.dumps(bad), encoding="utf-8")
    (d2 / "commit_stamp.json").write_text(json.dumps(STAMP), encoding="utf-8")
    out2 = tmp_path / "out" / "bad.md"
    with pytest.raises(SystemExit):
        wr.main(["--dir", str(d2), "--out", str(out2)])
    assert not out2.exists()


def test_main_reports_missing_input_files(tmp_path, capsys):
    assert wr.main(["--dir", str(tmp_path), "--out", str(tmp_path / "r.md")]) == 2
    assert "model_card.json" in capsys.readouterr().err
