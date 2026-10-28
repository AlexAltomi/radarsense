"""Explore which antenna pair each gesture label moves along. This is an exploratory analysis.

It reads only the nominal e1 files of the train users user2, user4, user9 and user11.

Usage: python training/infer_labels_antennas.py
The script streams each file one recording at a time and keeps up to KEEP recordings per label value. It stops
reading a file as soon as every label has KEEP recordings. In each kept recording the frames are aligned to the
start of the labelled span, with offsets from -PRE to +POST. For every frame the script looks at the strongest cell of
the unclipped dB map, which is averaged over the antennas, and records three things:

  The range bin of that cell.
  The Doppler bin, which is the signed bin (bin - 16) averaged over the cells within REL_DB of the peak and weighted
  by the power of the cells.
  The unit phasor of the phase difference delta_phi = angle(X_b) - angle(X_a) for the antenna pairs (0,1), (0,2) and
  (1,2).

The script writes the figure training/figures/label_antenna_inference.png and the section "Label and antenna
inference (exploratory)" of training/dataset_inventory.md. That section contains counts and averages only. The script
does not write constants.json, and the dB percentiles it reports are for orientation only.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))

from radarsense_dsp.frames import magnitude_db, range_doppler_spectra
from radarsense_dsp.io import iter_recordings

ZIP = ROOT / "data" / "radar_dataset.zip"
USERS = (2, 4, 9, 11)  # Only train users are analyzed.
LABELS = (1, 2, 3, 6, 7)
PAIRS = ((0, 1), (0, 2), (1, 2))
KEEP = 40
PRE, POST = 10, 20
REL_DB = 20.0
OFFSETS = np.arange(-PRE, POST + 1)
FIG = ROOT / "training" / "figures" / "label_antenna_inference.png"
OUT = ROOT / "training" / "dataset_inventory.md"
HEADING = "## Label and antenna inference (exploratory)"


def window_features(spectra: np.ndarray, start: int, length: int) -> dict:
    """Return the features of frames start-PRE to start+POST of one recording. The spectra have shape (frames, 3, range, Doppler)."""
    w = spectra[start - PRE : start + POST + 1]
    n = len(w)
    db = magnitude_db(w)
    flat = db.reshape(n, -1)
    peak = flat.argmax(axis=1)
    r, d = peak // db.shape[2], peak % db.shape[2]
    power = (np.abs(w) ** 2).mean(axis=1)
    weights = power * (db >= flat.max(axis=1)[:, None, None] - REL_DB)
    doppler_axis = np.arange(db.shape[2]) - db.shape[2] // 2
    doppler = (weights * doppler_axis).sum(axis=(1, 2)) / weights.sum(axis=(1, 2))
    frames = np.arange(n)
    phasors = []
    for a, b in PAIRS:
        z = w[frames, b, r, d] * np.conj(w[frames, a, r, d])
        phasors.append(z / (np.abs(z) + 1e-30))
    phasors = np.stack(phasors)  # Shape: (3, n).
    gesture = np.unwrap(np.angle(phasors[:, PRE : PRE + length]), axis=1)
    slopes = np.array([np.polyfit(np.arange(length), g, 1)[0] for g in gesture])
    return {"range": r.astype(float), "doppler": doppler, "phasors": phasors, "slopes": slopes}


def collect(member: str) -> tuple[dict[int, list[dict]], np.ndarray, int, int]:
    kept: dict[int, list[dict]] = {label: [] for label in LABELS}
    db_samples: list[np.ndarray] = []
    n_read = skipped = 0
    gen = iter_recordings(ZIP, member)
    for raw, targets in gen:
        n_read += 1
        nz = np.flatnonzero(targets)
        label = int(targets[nz[0]]) if len(nz) else 0
        if (
            label not in kept
            or len(np.unique(targets[nz])) != 1
            or nz[-1] - nz[0] + 1 != len(nz)
            or nz[0] - PRE < 0
            or nz[0] + POST + 1 > len(raw)
        ):
            skipped += 1  # Skip recordings without a gesture, with an unexpected label, with several values, with a split span, or whose window leaves the recording.
            continue
        if len(kept[label]) >= KEEP:
            continue
        spectra = range_doppler_spectra(raw)
        db_samples.append(magnitude_db(spectra).astype(np.float32).ravel())
        kept[label].append(window_features(spectra, int(nz[0]), len(nz)))
        if all(len(v) >= KEEP for v in kept.values()):
            break
    gen.close()
    return kept, np.percentile(np.concatenate(db_samples), [1, 99.5]), n_read, skipped


def mean_phase_curve(phasors: list[np.ndarray]) -> np.ndarray:
    """Return the circular mean of the phasors over the recordings as delta_phi(t) - delta_phi(0), unwrapped outwards from offset 0. The shape is (3, n)."""
    angle = np.angle(np.mean(phasors, axis=0))
    curve = np.zeros_like(angle)
    fwd = np.unwrap(angle[:, PRE:], axis=1)
    bwd = np.unwrap(angle[:, : PRE + 1][:, ::-1], axis=1)[:, ::-1]
    curve[:, PRE:] = fwd - fwd[:, :1]
    curve[:, :PRE] = (bwd - bwd[:, -1:])[:, :-1]
    return curve


def slope_tables(data: dict[int, dict[int, list[dict]]]) -> list[str]:
    lines = []
    for p, (a, b) in enumerate(PAIRS):
        title = f"Mean slope of the phase difference delta_phi = angle(ant{b}) - angle(ant{a}) over the labelled span, in rad/frame"
        lines += [
            f"{title}. Brackets show the share of recordings with a positive slope.",
            "",
            "| label | " + " | ".join(f"user{u}" for u in USERS) + " | all |",
            "|---|" + "---|" * (len(USERS) + 1),
        ]
        for label in LABELS:
            cells, pooled = [], []
            for u in USERS:
                s = np.array([f["slopes"][p] for f in data[u][label]])
                pooled.append(s)
                cells.append(f"{s.mean():+.3f} ({(s > 0).mean():.0%})" if len(s) else "-")
            s = np.concatenate(pooled)
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {s.mean():+.3f} ({(s > 0).mean():.0%}) |")
        lines.append("")
    return lines


def plot(data: dict[int, dict[int, list[dict]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ["range bin", "Doppler bin (signed)"] + [f"pair ({a},{b}): dphi(t) - dphi(0), rad" for a, b in PAIRS]
    fig, axes = plt.subplots(len(LABELS), 5, figsize=(20, 13), sharex=True)
    colors = dict(zip(USERS, ["tab:blue", "tab:orange", "tab:green", "tab:red"], strict=True))
    for i, label in enumerate(LABELS):
        for u in USERS:
            recs = data[u][label]
            if not recs:
                continue
            curves = [
                np.mean([f["range"] for f in recs], axis=0),
                np.mean([f["doppler"] for f in recs], axis=0),
                *mean_phase_curve([f["phasors"] for f in recs]),
            ]
            for j, y in enumerate(curves):
                axes[i, j].plot(OFFSETS, y, color=colors[u], label=f"user{u} (n={len(recs)})")
        axes[i, 0].set_ylabel(f"label {label}")
        for j in range(5):
            axes[i, j].axvline(0, color="gray", lw=0.8, ls="--")
            axes[i, j].grid(alpha=0.3)
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)
            if i == len(LABELS) - 1:
                axes[i, j].set_xlabel("frame offset from labelled span start")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Label/antenna inference (exploratory, train users, mean over up to 40 recordings per label)")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=100)
    plt.close(fig)


def write_section(lines: list[str]) -> None:
    text = OUT.read_text(encoding="utf-8")
    start = text.find(HEADING)
    if start != -1:
        end = text.find("\n## ", start + 1)
        text = text[:start].rstrip("\n") + "\n" + (text[end:] if end != -1 else "")
    OUT.write_text(text.rstrip("\n") + "\n\n" + "\n".join(lines), encoding="utf-8")


def main() -> None:
    data: dict[int, dict[int, list[dict]]] = {}
    notes = []
    for u in USERS:
        print(f"Reading user{u}_e1 one recording at a time.", flush=True)
        kept, pct, n_read, skipped = collect(f"fulldataset/user{u}_e1.npz")
        data[u] = kept
        counts = " ".join(f"{label}:{len(kept[label])}" for label in LABELS)
        note = (
            f"user{u}_e1: read {n_read} recordings, kept per label `{counts}`, skipped {skipped}. "
            f"The 1st and 99.5th percentiles of the raw dB values are {pct[0]:.1f} and {pct[1]:.1f}."
        )
        print(note, flush=True)
        notes.append(f"- {note}")

    tables = slope_tables(data)
    print("\n".join(tables))
    plot(data)
    print(f"Wrote the figure {FIG}.")
    users = ", ".join(f"user{u}" for u in USERS)
    intro = (
        f"This section looks for the antenna pair and the sweep direction that belong to each label. It uses only "
        f"the nominal e1 files of the train users {users}, and up to {KEEP} recordings per label, with all frames "
        f"aligned to the start of the labelled span. In every frame it takes the strongest cell of the unclipped dB "
        f"map. The Doppler value is the signed bin (bin - 16), averaged over the cells within {REL_DB:g} dB of the "
        f"peak and weighted by their energy. The figure is `training/figures/label_antenna_inference.png`. The dB "
        f"percentiles are for orientation only and are not used as constants. In the figure the phase change falls "
        f"inside the labelled span for user2 and user4, but about 8 frames before the span start for user9 and "
        f"user11. Users 9 and 11 therefore label the gesture about 8 frames after the motion. The slope tables "
        f"below are taken over the labelled span, so they cannot be compared between these two groups. Compare the "
        f"figure instead."
    )
    write_section([HEADING, "", intro, "", *notes, "", *tables])
    print(f"Wrote {OUT}.")


if __name__ == "__main__":
    main()
