"""Write the export record training/results/export-<version>.md from the files of the Week 4 export.

Usage: python training/write_export_report.py [--dir data/models/cnn-v1-e12] [--out training/results/export-cnn-v1-e12.md]

It reads model_card.json and commit_stamp.json (written by export_check.py and export_prepare.py) and copies numbers
from them. It never loads the dataset, a checkpoint or a .tflite model, and it never recomputes a metric. The only
derived value is the INT8 drop, which is the FP32 macro-F1 minus the INT8 macro-F1, both copied from the card.
A field that is missing from the files is written as "not recorded", and the script lists the missing fields.
The script stops with an error if the evaluation users include a test user (user10 or user12).
"""
import argparse
import json
import sys
from pathlib import Path

from build_cache import build_commit
from export_common import CARD_NAME, MAX_INT8_DROP, ROOT, STAMP_NAME

NOT_RECORDED = "not recorded"
TEST_USERS = ("user10", "user12")
MODEL_ORDER = (("pytorch", "PyTorch"), ("fp32", "LiteRT FP32"), ("int8", "LiteRT INT8"), ("fp16", "LiteRT float16"))


class Fields:
    """Reads values from the two JSON files and remembers which fields were missing."""

    def __init__(self, card: dict, stamp: dict):
        self.card, self.stamp, self.missing = card, stamp, []

    def get(self, source: str, *path: str):
        node = self.card if source == "card" else self.stamp
        for key in path:
            if not isinstance(node, dict) or key not in node or node[key] is None:
                self.missing.append(f"{source}.{'.'.join(path)}")
                return None
            node = node[key]
        return node


def text(value) -> str:
    return NOT_RECORDED if value is None else str(value)


def number(value, fmt: str) -> str:
    return NOT_RECORDED if value is None else format(value, fmt)


def check_users(users) -> None:
    bad = sorted({str(u) for u in users or []} & set(TEST_USERS))
    if bad:
        raise SystemExit(f"ERROR: The evaluation users include the test users {bad}. They may not be used before Week 11.")


def render(card: dict, stamp: dict, head: str, folder: str) -> tuple[str, list[str]]:
    f = Fields(card, stamp)
    check_users(card.get("validation", {}).get("users"))
    check_users(stamp.get("eval_users"))

    version = f.get("card", "version")
    models = f.get("card", "validation", "models") or {}
    sizes = f.get("card", "sizes_bytes") or {}
    agreement = f.get("card", "validation", "top1_agreement") or {}
    classes = f.get("card", "classes")
    users = f.get("card", "validation", "users")
    shape = f.get("card", "input", "shape")

    out = [f"# Export record: {text(version)}", ""]
    out += [f"This record was produced at commit {head} from {CARD_NAME} and {STAMP_NAME}. Nothing was recomputed.", ""]
    out += [
        (f"The checkpoint was trained at commit {text(f.get('stamp', 'checkpoint_commit'))} on cache "
        f"`{text(f.get('stamp', 'cache_id'))}`. The export inputs were prepared at commit "
        f"{text(f.get('stamp', 'prepared_at_commit'))}. A `+dirty` ending means that tracked files had uncommitted "
        "changes at that moment."),
        "",
    ]

    out += ["## Model", ""]
    out += [
        f"- version: {text(version)}",
        f"- run name: {text(f.get('stamp', 'config', 'run_name'))}",
        f"- architecture: {text(f.get('card', 'model'))}",
        f"- training seed: {text(f.get('stamp', 'config', 'seed'))}",
        (f"- best epoch: {text(f.get('card', 'validation', 'checkpoint_epoch'))} "
        f"(validation macro-F1 at that epoch during training: "
        f"{number(f.get('card', 'validation', 'checkpoint_val_macro_f1'), '.4f')})"),
        f"- parameters: {text(f.get('stamp', 'config', 'parameters'))}",
        f"- input shape: {text(shape)} as {text(f.get('card', 'input', 'layout'))}, {text(f.get('card', 'input', 'dtype'))}",
        f"- output: {text(f.get('card', 'output', 'shape'))} {text(f.get('card', 'output', 'kind'))}",
        f"- class order: {', '.join(classes) if classes else NOT_RECORDED}",
        "",
        "The gesture names are not yet verified. The class names are placeholders that follow the dataset label values.",
        "",
    ]

    out += ["## Results on the evaluation windows", "", "| model | accuracy | macro-F1 | file size |", "|---|---|---|---|"]
    for key, name in MODEL_ORDER:
        if key not in models:
            if key != "fp16":  # The float16 model is optional.
                f.missing.append(f"card.validation.models.{key}")
                out.append(f"| {name} | {NOT_RECORDED} | {NOT_RECORDED} | {NOT_RECORDED} |")
            continue
        m = models[key]
        size = sizes.get(key)
        if size is None:
            f.missing.append(f"card.sizes_bytes.{key}")
        size_cell = NOT_RECORDED if size is None else f"{size / 1024:.1f} KB"
        out.append(f"| {name} | {number(m.get('accuracy'), '.4f')} | {number(m.get('macro_f1'), '.4f')} | {size_cell} |")
    out += ["", "File sizes use 1 KB = 1,024 bytes. The PyTorch size is the checkpoint file `best.pt`.", ""]

    out += ["## Agreement and the INT8 check", ""]
    for key, name in MODEL_ORDER[1:]:
        if key in models:
            v = agreement.get(f"pytorch_vs_{key}")
            if v is None:
                f.missing.append(f"card.validation.top1_agreement.pytorch_vs_{key}")
            out.append(f"- top-1 agreement of {name} with PyTorch: {number(v, '.4f')}")
    f1_fp32 = (models.get("fp32") or {}).get("macro_f1")
    f1_int8 = (models.get("int8") or {}).get("macro_f1")
    limit = MAX_INT8_DROP * 100
    fallback = card.get("fallback", "missing")
    if fallback == "missing":
        f.missing.append("card.fallback")
    if f1_fp32 is None or f1_int8 is None:
        out += ["", f"INT8 drop against FP32: {NOT_RECORDED} (limit {limit:.0f} point)."]
    else:
        drop = (f1_fp32 - f1_int8) * 100
        passed = drop <= limit
        verdict = (
            "INT8 passed the check."
            if passed
            else "INT8 failed the check, so float16 is the fallback."
        )
        out += [
            "",
            (f"INT8 drop against FP32: {drop:+.2f} points of macro-F1 (FP32 minus INT8, so a negative number means INT8 "
            f"scored higher). The limit is {limit:.0f} point. {verdict}"),
            f"The card records the fallback as {NOT_RECORDED if fallback == 'missing' else (fallback or 'none')}.",
        ]
        if fallback != "missing" and passed != (not fallback):
            out += ["", "Warning: the drop and the fallback field in the card do not agree. Check export_check.py."]
    out += [""]

    n_windows = f.get("card", "validation", "windows")
    out += ["## Evaluation set", ""]
    out += [
        f"- windows: {text(n_windows)}, {text(f.get('card', 'validation', 'selection'))}",
        f"- users: {', '.join(users) if users else NOT_RECORDED}",
        f"- class counts (order as above): {text(f.get('stamp', 'eval_class_counts'))}",
        "",
        ("These are validation windows. Each window was encoded to uint8 and decoded back with the replay codec, so all "
        "models saw the frames as the app sees them. The test users user10 and user12 were not used. Test metrics are "
        f"{'empty' if card.get('test', {}).get('macro_f1') is None else 'present'}. "
        f"The card says: {text(f.get('card', 'test', 'note'))}"),
        "",
    ]

    out += ["## Normalization and dataset", ""]
    out += [
        (f"- dB range: {text(f.get('card', 'db_range', 'db_min'))} to {text(f.get('card', 'db_range', 'db_max'))} dB "
        f"(from {text(f.get('card', 'db_range', 'source'))}, computed at commit "
        f"{text(f.get('card', 'db_range', 'source_commit'))} from the train users "
        f"{text(f.get('card', 'db_range', 'source_users'))})"),
        (f"- antenna pairs: channel 1 uses {text(f.get('card', 'db_range', 'pair_a'))}, "
        f"channel 2 uses {text(f.get('card', 'db_range', 'pair_b'))}"),
        f"- uint8 decode, channel 0: {text(f.get('card', 'uint8_decode', 'channel_0', 'formula'))}",
        f"- uint8 decode, channels 1 and 2: {text(f.get('card', 'uint8_decode', 'channel_1', 'formula'))}",
        f"- dataset DOI: {text(f.get('card', 'dataset', 'doi'))} ({text(f.get('card', 'dataset', 'license'))})",
        "",
    ]

    out += ["## Where the files are", ""]
    out += [
        (f"The files are in `{folder}` on the development PC. The folder is git-ignored and is not in the repository: "
        "`best.pt`, `export_inputs.npz`, `golden.npz`, the `.tflite` files, `model_card.json` and `commit_stamp.json`."),
        (f"The SHA-256 of `best.pt` is {text(f.get('stamp', 'sha256', 'best.pt'))} and of `export_inputs.npz` is "
        f"{text(f.get('stamp', 'sha256', 'export_inputs.npz'))}."),
        ("The LiteRT conversion ran on Google Colab with `training/convert_litert.py`, because litert-torch has no Windows "
        "wheel. The comparison ran on Windows with `training/export_check.py`."),
        "",
    ]

    missing = sorted(set(f.missing))
    out += ["## Fields not recorded", ""]
    out += [f"- `{m}`" for m in missing] if missing else ["None. Every field was present in the two files."]
    return "\n".join(out) + "\n", missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "models" / "cnn-v1-e12")
    ap.add_argument("--out", type=Path, default=ROOT / "training" / "results" / "export-cnn-v1-e12.md")
    args = ap.parse_args(argv)
    for name in (CARD_NAME, STAMP_NAME):
        if not (args.dir / name).exists():
            print(f"ERROR: {args.dir / name} does not exist. Run export_prepare.py and export_check.py first.", file=sys.stderr)
            return 2
    card = json.loads((args.dir / CARD_NAME).read_text(encoding="utf-8"))
    stamp = json.loads((args.dir / STAMP_NAME).read_text(encoding="utf-8"))
    try:
        folder = args.dir.absolute().relative_to(ROOT).as_posix()  # not resolve(): data/ is a junction
    except ValueError:
        folder = args.dir.as_posix()
    report, missing = render(card, stamp, build_commit(), folder)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}.")
    print("Missing fields: " + (", ".join(missing) if missing else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
