"""Check how the labelled frames are arranged inside the recordings of two train members.

For each recording the script counts the labelled frames, finds the start frame and the length of the gesture, and
checks whether the gesture is one contiguous run.

Usage: python training/label_structure.py
The script reads the targets with `io.read_meta`, which reads only headers and targets and never the inputs data. It
writes the section "Label structure (train users)" of training/dataset_inventory.md and replaces any earlier version
of that section. The output contains only counts.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))

from radarsense_dsp.io import read_meta
from radarsense_dsp.labels import analyze_targets

ZIP = ROOT / "data" / "radar_dataset.zip"
MEMBERS = ["fulldataset/user4_e1.npz", "fulldataset/user9_e1.npz"]  # Both members belong to train users.
OUT = ROOT / "training" / "dataset_inventory.md"
HEADING = "## Label structure (train users)"


def fmt(hist: dict[int, int]) -> str:
    return " ".join(f"{k}:{v}" for k, v in hist.items()) or "-"


def render(stem: str, a: dict) -> list[str]:
    n_gesture = a["n"] - a["zero_gesture"]
    lines = [
        f"### {stem} (N={a['n']})",
        f"- Labelled frames per recording (`frames:recordings`): {fmt(a['labelled_hist'])}",
        f"- Recordings with no gesture frames: {a['zero_gesture']}",
        f"- Recordings with more than one distinct gesture value: {a['multi_value']['count']}"
        + (f" (first indices {a['multi_value']['first_indices']})" if a["multi_value"]["count"] else ""),
    ]
    s = a["start"]
    if s:
        lines.append(
            f"- Gesture start frame: minimum {s['min']}, median {s['median']:g}, maximum {s['max']}. "
            f"Counts in 10-frame bins (`bin_start:recordings`): {fmt(s['hist10'])}"
        )
    lines += [
        f"- Gesture span, which is the last labelled frame minus the first plus 1 (`frames:recordings`): {fmt(a['span_hist'])}",
        f"- Contiguous runs of labelled frames per recording (`runs:recordings`): {fmt(a['runs_hist'])}",
        f"- Recordings whose gesture is a single contiguous run: {a['contiguous']} of {n_gesture}.",
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
        ("This section checks how the labelled frames are arranged inside the recordings of two train members, "
        "user4_e1 and user9_e1. It reports counts only. A frame is a gesture frame if its label is not 0. The start "
        "frame, span and run statistics cover only the recordings that contain a gesture."),
        "",
    ]
    for member in MEMBERS:
        n, targets = read_meta(ZIP, member)
        assert n == len(targets)
        lines += render(Path(member).stem, analyze_targets(targets))
    print("\n".join(lines))
    write_section(lines)
    print(f"Wrote {OUT}.")


if __name__ == "__main__":
    main()
