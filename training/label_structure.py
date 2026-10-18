"""Label structure check, TRAIN USERS ONLY: per-recording labelled frames, start frame, run length, contiguity.

Usage: python training/label_structure.py
Reads targets via io.read_meta (headers + targets only, never inputs data) for the members below and writes the
`## Label structure (train users)` section of training/dataset_inventory.md, replacing any previous version.
Output contains only counts.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))

from radarsense_dsp.io import read_meta
from radarsense_dsp.labels import analyze_targets

ZIP = ROOT / "data" / "radar_dataset.zip"
MEMBERS = ["fulldataset/user4_e1.npz", "fulldataset/user9_e1.npz"]  # both train users
OUT = ROOT / "training" / "dataset_inventory.md"
HEADING = "## Label structure (train users)"


def fmt(hist: dict[int, int]) -> str:
    return " ".join(f"{k}:{v}" for k, v in hist.items()) or "-"


def render(stem: str, a: dict) -> list[str]:
    n_gesture = a["n"] - a["zero_gesture"]
    lines = [
        f"### {stem} (N={a['n']})",
        f"- labelled frames per recording (`frames:recordings`): {fmt(a['labelled_hist'])}",
        f"- recordings with zero gesture frames: {a['zero_gesture']}",
        f"- recordings with more than one distinct gesture value: {a['multi_value']['count']}"
        + (f" (first indices {a['multi_value']['first_indices']})" if a["multi_value"]["count"] else ""),
    ]
    s = a["start"]
    if s:
        lines.append(
            f"- gesture start frame: min {s['min']}, median {s['median']:g}, max {s['max']}; "
            f"10-frame bins (`bin_start:recordings`): {fmt(s['hist10'])}"
        )
    lines += [
        f"- gesture span, last - first + 1 (`frames:recordings`): {fmt(a['span_hist'])}",
        f"- contiguous runs of labelled frames per recording (`runs:recordings`): {fmt(a['runs_hist'])}",
        f"- contiguous (single run): {a['contiguous']} of {n_gesture} recordings with a gesture",
        "",
    ]
    return lines


def write_section(lines: list[str]) -> None:
    text = OUT.read_text(encoding="utf-8")
    start = text.find(HEADING)
    if start != -1:
        end = text.find("\n## ", start + 1)
        text = text[:start].rstrip("\n") + "\n" + (text[end:] if end != -1 else "")
    text = text.rstrip("\n") + "\n\n" + "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")


def main() -> None:
    lines = [
        HEADING,
        "",
        "Counts only. Gesture = any non-zero label. Start frame, span and runs cover recordings that contain a gesture.",
        "",
    ]
    for member in MEMBERS:
        n, targets = read_meta(ZIP, member)
        assert n == len(targets)
        lines += render(Path(member).stem, analyze_targets(targets))
    print("\n".join(lines))
    write_section(lines)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
