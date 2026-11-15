# Export record: cnn-v1-e12

This record was produced at commit c4e4e8aa0657d56b3c03f288b27c349deae77c53 from model_card.json and commit_stamp.json. Nothing was recomputed.

The checkpoint was trained at commit 3b0b57ef9a22b5b00c00777b7256c17ba4dcb595 on cache `4ecddb8cb8697e2d`. The export inputs were prepared at commit f161f9584584234ecdad1ea41d713cb302feb487+dirty. A `+dirty` ending means that tracked files had uncommitted changes at that moment.

## Model

- version: cnn-v1-e12
- run name: cnn-v1-e12
- architecture: cnn-v1
- training seed: 0
- best epoch: 4 (validation macro-F1 at that epoch during training: 0.9448)
- parameters: 65286
- input shape: [1, 32, 3, 32, 32] as (batch, frame, channel, range, Doppler); NCHW inside each frame, float32
- output: [1, 6] logits
- class order: none, label1, label2, label3, label6, label7

The gesture names are not yet verified. The class names are placeholders that follow the dataset label values.

## Results on the evaluation windows

| model | accuracy | macro-F1 | file size |
|---|---|---|---|
| PyTorch | 0.9310 | 0.9308 | 264.5 KB |
| LiteRT FP32 | 0.9310 | 0.9308 | 263.2 KB |
| LiteRT INT8 | 0.9380 | 0.9373 | 276.8 KB |
| LiteRT float16 | 0.9290 | 0.9289 | 137.7 KB |

File sizes use 1 KB = 1,024 bytes. The PyTorch size is the checkpoint file `best.pt`.

## Agreement and the INT8 check

- top-1 agreement of LiteRT FP32 with PyTorch: 1.0000
- top-1 agreement of LiteRT INT8 with PyTorch: 0.9790
- top-1 agreement of LiteRT float16 with PyTorch: 0.9980

INT8 drop against FP32: -0.64 points of macro-F1 (FP32 minus INT8, so a negative number means INT8 scored higher). The limit is 1 point. INT8 passed the check.
The card records the fallback as none.

## Evaluation set

- windows: 1000, class-balanced, seed 0, after the uint8 wire round trip
- users: user3, user5
- class counts (order as above): [167, 167, 167, 167, 166, 166]

These are validation windows. Each window was encoded to uint8 and decoded back with the replay codec, so all models saw the frames as the app sees them. The test users user10 and user12 were not used. Test metrics are empty. The card says: Not evaluated before Week 11.

## Normalization and dataset

- dB range: 24.38 to 62.35 dB (from dsp/radarsense_dsp/constants.json, computed at commit c6381f842ddecdbe1b6b428bd5c05ae3e63d9be7 from the train users [1, 2, 4, 6, 7, 8, 9, 11])
- antenna pairs: channel 1 uses [0, 2], channel 2 uses [1, 2]
- uint8 decode, channel 0: x = u / 255
- uint8 decode, channels 1 and 2: x = u / 255 * 2 - 1
- dataset DOI: 10.5281/zenodo.15178095 (CC BY 4.0)

## Where the files are

The files are in `data/models/cnn-v1-e12` on the development PC. The folder is git-ignored and is not in the repository: `best.pt`, `export_inputs.npz`, `golden.npz`, the `.tflite` files, `model_card.json` and `commit_stamp.json`.
The SHA-256 of `best.pt` is f454098121ae29d3c404d14ce2b3c59cd335d424614a5a2ac9f5bbfbe06c298d and of `export_inputs.npz` is 9232ecd9ac7d356b9c3514f77daec34d42b513e873fa4495e71b6233d40d91aa.
The LiteRT conversion ran on Google Colab with `training/convert_litert.py`, because litert-torch has no Windows wheel. The comparison ran on Windows with `training/export_check.py`.

## Fields not recorded

None. Every field was present in the two files.
