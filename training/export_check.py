"""Check the LiteRT models against PyTorch on the evaluation windows (Windows, runs in .venv-export).

Usage: .venv-export\\Scripts\\python.exe training/export_check.py --dir data/models/cnn-v1-e12

Needs these files in --dir: best.pt, export_inputs.npz and commit_stamp.json (from export_prepare.py), and model_fp32.tflite
and model_int8.tflite (from convert_litert.py in Colab). model_fp16.tflite is optional.

It runs PyTorch (the checkpoint, CPU), the FP32 tflite and the INT8 tflite on the 1,000 VALIDATION windows of
export_inputs.npz. Their input is the float32 decode of the uint8 wire frames, so the numbers describe what the app
sees. It prints the top-1 agreement, the macro-F1 and the sizes, and writes golden.npz (the uint8 windows, their classes
and the logits) and model_card.json. If the INT8 macro-F1 is more than 1 point below FP32, float16 is marked as the
fallback in the card. Test metrics stay empty until Week 11.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from export_common import (
    CARD_NAME,
    FP16_NAME,
    FP32_NAME,
    GOLDEN_NAME,
    INPUTS_NAME,
    INT8_NAME,
    MAX_INT8_DROP,
    ROOT,
    STAMP_NAME,
    build_model_card,
    int8_needs_fallback,
    model_metrics,
    top1_agreement,
    write_json,
)
from model import GestureCNN

from replay import codec

CONSTANTS = ROOT / "dsp" / "radarsense_dsp" / "constants.json"


def quantize_input(x: np.ndarray, dtype: np.dtype, scale: float, zero_point: int) -> np.ndarray:
    """Convert float input to the integer input type of a quantized model. A float model gets x unchanged."""
    if np.issubdtype(dtype, np.floating):
        return x.astype(dtype)
    info = np.iinfo(dtype)
    return np.clip(np.rint(x / scale) + zero_point, info.min, info.max).astype(dtype)


def dequantize_output(y: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if np.issubdtype(y.dtype, np.floating):
        return y.astype(np.float32)
    return ((y.astype(np.float32) - zero_point) * scale).astype(np.float32)


def run_tflite(path: Path, x: np.ndarray) -> np.ndarray:
    """Run a .tflite model on windows of shape (N, 32, 3, 32, 32) and return float32 logits (N, 6)."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    batch = int(inp["shape"][0])
    if x.shape[1:] != tuple(inp["shape"][1:]):
        raise ValueError(f"{path.name} expects the input shape {tuple(inp['shape'])}, got {x.shape}.")
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    logits = []
    for i in range(0, len(x) - batch + 1, batch):
        interp.set_tensor(inp["index"], quantize_input(x[i : i + batch], inp["dtype"], in_scale, in_zp))
        interp.invoke()
        logits.append(dequantize_output(interp.get_tensor(out["index"]).copy(), out_scale, out_zp))
    return np.concatenate(logits)


def run_pytorch(checkpoint: Path, x: np.ndarray) -> np.ndarray:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = GestureCNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i : i + 100])).numpy() for i in range(0, len(x), 100)])


def summarize(y: np.ndarray, logits: dict[str, np.ndarray]) -> tuple[dict, dict, str | None]:
    """Return the metrics of every model, the pairwise top-1 agreement and the fallback ("float16" or None).

    `logits` has the keys "pytorch", "fp32" and "int8", and optionally "fp16".
    """
    metrics = {name: model_metrics(y, lg) for name, lg in logits.items()}
    pairs = [("pytorch", "fp32"), ("pytorch", "int8"), ("fp32", "int8")]
    agreement = {f"{a}_vs_{b}": top1_agreement(logits[a], logits[b]) for a, b in pairs}
    for a, b in [("pytorch", "fp16"), ("fp32", "fp16")]:
        if b in logits:
            agreement[f"{a}_vs_{b}"] = top1_agreement(logits[a], logits[b])
    fallback = "float16" if int8_needs_fallback(metrics["fp32"]["macro_f1"], metrics["int8"]["macro_f1"]) else None
    return metrics, agreement, fallback


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="data/models/<version>")
    args = ap.parse_args(argv)
    d = args.dir

    missing = [n for n in ("best.pt", INPUTS_NAME, STAMP_NAME, FP32_NAME, INT8_NAME) if not (d / n).exists()]
    if missing:
        print(f"ERROR: {d} lacks {missing}. Run export_prepare.py and convert_litert.py (Colab) first.", file=sys.stderr)
        return 2
    stamp = json.loads((d / STAMP_NAME).read_text(encoding="utf-8"))
    with np.load(d / INPUTS_NAME) as z:
        eval_u8, y = z["eval_u8"], z["eval_classes"].astype(int)
    x = codec.decode_frames(eval_u8)
    print(f"{len(x)} validation windows of {stamp['eval_users']}, decoded from the uint8 wire frames.")

    logits = {"pytorch": run_pytorch(d / "best.pt", x), "fp32": run_tflite(d / FP32_NAME, x), "int8": run_tflite(d / INT8_NAME, x)}
    if (d / FP16_NAME).exists():
        logits["fp16"] = run_tflite(d / FP16_NAME, x)
    metrics, agreement, fallback = summarize(y, logits)
    files = {"pytorch": "best.pt", "fp32": FP32_NAME, "int8": INT8_NAME, "fp16": FP16_NAME}
    sizes = {k: (d / files[k]).stat().st_size for k in logits}

    print(f"\n{'model':8} {'accuracy':>9} {'macro-F1':>9} {'size':>10}")
    for k, m in metrics.items():
        print(f"{k:8} {m['accuracy']:9.4f} {m['macro_f1']:9.4f} {sizes[k] / 1024:8.1f} KB")
    print("\nTop-1 agreement:", ", ".join(f"{k} {v:.4f}" for k, v in agreement.items()))
    drop = metrics["fp32"]["macro_f1"] - metrics["int8"]["macro_f1"]
    if fallback:
        have = "available" if "fp16" in logits else f"missing: convert it ({FP16_NAME} was not produced)"
        print(f"\nINT8 macro-F1 is {drop * 100:.2f} points below FP32 (limit {MAX_INT8_DROP * 100:.0f}). "
              f"Use float16 as the fallback ({have}).")
    else:
        print(f"\nINT8 macro-F1 is {drop * 100:.2f} points below FP32 (limit {MAX_INT8_DROP * 100:.0f}): INT8 is acceptable.")

    golden = {"eval_u8": eval_u8, "classes": y.astype(np.int8), "logits_pytorch": logits["pytorch"],
              "logits_fp32": logits["fp32"], "logits_int8": logits["int8"]}
    if "fp16" in logits:
        golden["logits_fp16"] = logits["fp16"]
    np.savez_compressed(d / GOLDEN_NAME, **golden)

    constants = json.loads(CONSTANTS.read_text(encoding="utf-8"))
    card = build_model_card(stamp["version"], stamp, constants, metrics, sizes, agreement, fallback)
    write_json(d / CARD_NAME, card)
    print(f"\nWrote {d / GOLDEN_NAME} ({(d / GOLDEN_NAME).stat().st_size / 1024 / 1024:.1f} MB) and {d / CARD_NAME}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
