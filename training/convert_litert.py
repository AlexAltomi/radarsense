"""Convert the cnn-v1 checkpoint to LiteRT (.tflite) files. LINUX ONLY: run it in Google Colab, not on Windows.

litert-torch needs the litert-converter package, which has no Windows wheel (checked 2026-11-15: wheels exist for
Linux x86_64 and macOS arm64 only). This script needs only numpy, torch, torchao and litert-torch. It imports nothing
from the repository that needs Windows or CUDA: training/model.py (torch only) and backend/replay/replay/codec.py
(numpy only, for the uint8 decode formulas).

Steps in Colab (a CPU runtime is enough):

  1. Open a new notebook and run:   !git clone https://github.com/AlexAltomi/radarsense.git
                                    %cd radarsense
     Then check out the commit that you are exporting:   !git checkout <commit>
  2. Install the converter:         !pip install litert-torch torchao
  3. Upload the two files from data/models/<version>/ on your PC (the Files panel on the left, or
     `from google.colab import files; files.upload()`), into the folder export_in/ of the clone:
         best.pt   and   export_inputs.npz
  4. Run:                           !python training/convert_litert.py --checkpoint export_in/best.pt \\
                                        --inputs export_in/export_inputs.npz --out-dir export_out
  5. Download export_out/model_fp32.tflite, export_out/model_int8.tflite and export_out/model_fp16.tflite (the last one
     only exists if the float16 step worked) with `files.download(...)`, and put them into data/models/<version>/
     next to best.pt on your PC.
  6. On Windows, run training/export_check.py in .venv-export.

What it writes to --out-dir:
  model_fp32.tflite   BatchNorm folded into the convolutions, float32, NCHW, input (1, 32, 3, 32, 32).
  model_int8.tflite   PT2E static INT8 (per-channel weights), calibrated on the 500 TRAIN-user windows of
                      export_inputs.npz. The input and output stay float32. The calibration windows are the wire
                      round trip of the cache, decoded with replay.codec.
  model_fp16.tflite   Weights stored as float16 (ai-edge-quantizer FLOAT_CASTING), if the API supports it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "backend" / "replay"))

from model import MODEL_NAME, GestureCNN, fold_batchnorm

from replay import codec

SAMPLE_SHAPE = (1, 32, 3, 32, 32)


def load_model(checkpoint: Path) -> GestureCNN:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt["model"] != MODEL_NAME:
        raise SystemExit(f"The checkpoint holds the model {ckpt['model']}, not {MODEL_NAME}.")
    model = GestureCNN()
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def size_kb(path: Path) -> str:
    return f"{path.stat().st_size / 1024:.1f} KB"


def convert_fp32(folded: GestureCNN, sample: torch.Tensor, path: Path):
    import litert_torch

    edge = litert_torch.convert(folded.eval(), (sample,))
    edge.export(str(path))
    return edge


def convert_int8(folded: GestureCNN, sample: torch.Tensor, calib: np.ndarray, path: Path) -> None:
    import litert_torch
    from litert_torch.quantize import pt2e_quantizer
    from torchao.quantization.pt2e import quantize_pt2e

    exported = torch.export.export(folded.eval(), (sample,)).module()
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)
    with torch.no_grad():
        for window in calib:  # the observers record the range of every activation
            prepared(torch.from_numpy(window[None]))
    quantized = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    litert_torch.convert(quantized, (sample,)).export(str(path))


def convert_fp16(fp32_path: Path, path: Path) -> None:
    from ai_edge_quantizer import qtyping, quantizer, recipe_manager
    from ai_edge_quantizer.algorithm_manager import AlgorithmName

    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=16,
        algorithm_key=AlgorithmName.FLOAT_CASTING,
    )
    qt = quantizer.Quantizer(str(fp32_path))
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(path, overwrite=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True, help="export_inputs.npz from export_prepare.py")
    ap.add_argument("--out-dir", type=Path, default=Path("export_out"))
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.inputs) as z:
        calib = codec.decode_frames(z["calib_u8"])  # the same decode as the app
        probe = codec.decode_frames(z["eval_u8"][:8])
    model = load_model(args.checkpoint)
    folded = fold_batchnorm(load_model(args.checkpoint))
    sample = torch.zeros(SAMPLE_SHAPE)
    with torch.no_grad():
        diff = (model(torch.from_numpy(probe)) - folded(torch.from_numpy(probe))).abs().max().item()
    print(f"BatchNorm folded: max logit difference to the original model on 8 windows: {diff:.2e}")

    fp32 = args.out_dir / "model_fp32.tflite"
    edge = convert_fp32(folded, sample, fp32)
    print(f"Wrote {fp32} ({size_kb(fp32)}).")
    try:  # a quick look at the converted model against PyTorch
        out = np.asarray(edge(torch.from_numpy(probe[:1])))
        with torch.no_grad():
            ref = model(torch.from_numpy(probe[:1])).numpy()
        print(f"FP32 tflite vs PyTorch, max logit difference on 1 window: {np.abs(out - ref).max():.2e}")
    except Exception as e:  # noqa: BLE001
        print(f"(skipped the quick FP32 check: {e})")

    int8 = args.out_dir / "model_int8.tflite"
    convert_int8(folded, sample, calib, int8)
    print(f"Wrote {int8} ({size_kb(int8)}), calibrated on {len(calib)} windows.")

    fp16 = args.out_dir / "model_fp16.tflite"
    try:
        convert_fp16(fp32, fp16)
        print(f"Wrote {fp16} ({size_kb(fp16)}).")
    except Exception as e:  # noqa: BLE001
        print(f"Float16 variant skipped: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
