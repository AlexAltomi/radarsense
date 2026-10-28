# RadarSense v1

RadarSense is an Android app that recognizes five hand gestures, and the absence of a gesture, from recordings of a
60 GHz FMCW (frequency-modulated continuous-wave) radar. The app does not use radar hardware. A replay server streams
recorded radar frames to the phone, and the app classifies them in real time. The project is a solo lab project for
Software Development at UPT, 2026.

The rules, dataset facts and specifications that the code follows are collected in
[docs/project-rules.md](docs/project-rules.md). Please read that document first.

## Dataset credit (CC BY 4.0)
Frame-Labeled 60 GHz FMCW Radar Gesture Dataset (Infineon), Zenodo,
DOI: 10.5281/zenodo.15178095. The dataset is not included in this repo.

## Repository layout

| Folder | Content |
|---|---|
| `dsp/` | The `radarsense_dsp` package: dataset access, signal processing, cache builder and training windows. Its tests are in `dsp/tests`. |
| `training/` | Scripts that inspect the dataset, compute constants, build the cache and train the baseline model. The frozen user split is `training/splits.json`, and `training/dataset_inventory.md` describes the dataset. |
| `backend/` | The replay server (`replay`), the API (`api`) and the reverse proxy (`caddy`). |
| `android/` | The Android app (package `com.radarsense.app`). |
| `docs/` | The project rules. |
| `data/` | A junction to `D:\radarsense-data`, which holds the dataset and the processed cache. It is ignored by git. |

## Data pipeline

The dataset is a 36.5 GB zip file with one `.npz` member per user, session and recording type. The code streams it one
recording at a time and never extracts it. Each raw radar frame has 3 antennas, 32 chirps and 64 samples. The script
`training/build_cache.py` turns every frame into an image of 3 channels of 32 by 32 values. It applies a Hann-windowed
FFT along the samples to find the range, and a second one along the chirps to find the Doppler speed, which gives a
range-Doppler map. The first channel holds the magnitude of that map in dB, clipped to a fixed range and scaled to 0
to 1. The other two channels hold the phase differences between two antenna pairs, which depend on the direction of
the movement. The images are stored as float16 in `data/processed`, together with an `index.json` that records the
status of every member and checks that each recording contains exactly one contiguous gesture. The module
`radarsense_dsp.windows` then cuts every recording into windows of 32 frames, one every 4 frames, and labels each
window as a gesture, as no gesture, or excludes it when it overlaps a gesture only partly. The script
`training/baseline.py` computes hand-crafted features for these windows and trains a logistic regression and a random
forest on the train users. It evaluates them on the validation users and writes the results to
`training/results/baseline.md`. The test users stay locked until Week 11.

## Setup

The project uses Python 3.12 on Windows. All commands run from the repository root.

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
cp scripts/pre-commit .git/hooks/pre-commit
```

The last line installs a git hook that blocks a commit when ruff or the tests fail. The dataset has to be placed in
`D:\radarsense-data`, and the junction `data` points to that folder (`mklink /J data D:\radarsense-data`).

## Running the checks

```
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
pyright
```

The tests use small synthetic data and do not need the dataset. Pyright is installed globally with
`npm install -g pyright`, and its settings are in `pyrightconfig.json`.

## Android notes
- targetSdk = 34 on purpose: apps targeting API 37+ are blocked from local-network
  addresses (ACCESS_LOCAL_NETWORK, Android 17), and the replay server is reached
  over the local network. Revisit only if the app ever needs to target 37+.
