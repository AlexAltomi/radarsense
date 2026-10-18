# CLAUDE.md — RadarSense v1

Android app that recognizes 5 hand gestures + "no gesture" from **replayed** Infineon 60 GHz FMCW radar recordings
(Zenodo, CC BY 4.0, DOI 10.5281/zenodo.15178095). UPT Software Development lab project, solo, €0, 19 Oct 2026 → 10 Jan 2027.
Master plan: "RadarSense v1 — Master Plan" Claude Doc. If this file and the Doc disagree, the Doc wins — ask before acting.

## Hard rules (never break, never "temporarily" relax)
1. **Test users user10 and user12 are never loaded before Week 11.** The loader refuses them unless `RADARSENSE_ALLOW_TEST=1`.
   Do not set that variable, do not compute statistics, normalization constants or plots that include them.
2. **Frozen split** (`training/splits.json`, commit c546264):
   train user1,2,4,6,7,8,9,11 · val user3,5 · test user10,12 · robustness = anomalous (fast/slow/wrist) files; the
   headline robustness number comes from user10/user12 anomalous files only, train-user anomaly accuracy is reported
   separately as secondary (those users were seen in training).
   Never edit splits.json. Normalization constants and label/antenna inference use **train users only**.
3. **Train on nominal files only.** Anomalous files (fast/slow/wrist) are for the robustness evaluation only.
4. **No mirroring augmentation** of range-Doppler or angle maps (it swaps left/right / up/down labels).
   Allowed: ±3 frame shift, ±3 dB amplitude, small noise.
5. **Data never enters git.** `data/` is a junction to `D:\radarsense-data` and is git-ignored.
6. **Never extract the whole 36.5 GB zip and never load a whole `inputs` array.**
   Each `.npz` holds `inputs` float32 (N,100,3,32,64), ≈ 2.46 GB per 1000 recordings; N varies per file
   (about 1000–3000), read from the .npy header. Stream one recording at a time.
   Temporary extraction of a single `.npz` to `D:\radarsense-data\tmp` is allowed; delete it after processing.
7. **Scope contract.** Out of v1: live radar hardware, custom gestures, Play Store, cloud production hosting, thesis-level DSP.
   Do not add features not in the plan. Nothing new before Week 9.
8. Every reported metric records git commit + model version next to it.

## Dataset facts (verified locally, Week 1)
- 41 `.npz` files in `radar_dataset.zip` (md5 d63a9645f27ebded613807b674bba63a).
- `inputs`: float32, (recordings, 100 frames, 3 antennas, 32 chirps, 64 samples). `targets`: int32, (recordings, 100).
  N (recordings) varies per file (about 1000–3000), read from the .npy header.
- Members are `fulldataset/user{N}_e{S}[_{fast|slow|wrist}].npz`; user1 has sessions e1–e6, others e1 only;
  outer members are ZIP_DEFLATED.
- Train composition: user1 = 57% of nominal train recordings; balance per user at training time; the cache keeps everything.
- Label values {0,1,2,3,6,7}; 0 = no gesture.
- Gesture structure: verified per recording for user4_e1 (8-frame span) and user9_e1 (10-frame span): exactly one
  contiguous gesture per recording, no gesture-free recordings. For other nominal files, spans of 8 (users 1-8) or 10
  (users 9, 11) are inferred from inventory label counts assuming 200 recordings per class; anomalous files: span varies per file (8, 10 or mixed); never assume
  it, always read it from targets. The cache builder verifies this per recording.
- Label → gesture name mapping: **UNVERIFIED** (to be inferred in Week 2 from train users; update this line).
- Antenna geometry (train users 2/4/9/11): antenna index 2 is the L-corner; orthogonal pairs are (0,2) and (1,2);
  (0,1) is the diagonal and responds to all swipes, never use it. Labels 2/3 = opposite sweeps on pair (0,2);
  labels 6/7 = opposite sweeps on pair (1,2); label 1 shows no consistent angular sweep (likely push, UNVERIFIED).
  Which axis is horizontal and the direction within each axis: UNVERIFIED.

## Specs to keep consistent
- DSP per frame → 3×32×32: subtract chirp mean → Hann + range FFT, keep 32 positive bins → subtract mean over chirps
  per range bin → Hann + Doppler FFT, fftshift → ch0 = magnitude dB averaged over 3 antennas; ch1 = sin(Δφ) of pair (0,2)
  ('axis A', labels 2/3), ch2 = sin(Δφ) of pair (1,2) ('axis B', labels 6/7), weighted by normalized magnitude → clip ch0 to fixed dB range (constants from train users,
  stored in `dsp/radarsense_dsp/constants.json`), scale 0–1, store float16.
  Processed frame layout: `[channel, range, Doppler]` = (3, 32, 32); the ch1/ch2 weight is the scaled ch0.
  Doppler sign: approaching hand = signed Doppler < 0 (bins below 16), likely, from the figure (2bca1f5).
  No left/right/up/down names in code until verified.
- Processed cache: `data/processed/<stem>.npy` float16 (N,100,3,32,32), targets int8, `index.json`; resumable
  (write `.partial`, rename when complete, skip finished files). `build_cache.py` (not built yet) must, per
  recording, assert exactly one contiguous gesture span; record per-file span-length histogram and violation count in
  `index.json`; and fail with a clear message on any violation.
- Wire format to the app: 3×32×32 uint8 frames; binary frame = 16-byte header (uint32 seq, uint64 server ms,
  uint8 true label or 255, 3 reserved) + 3,072 bytes = 3,088 bytes.
- Detector: infer every 4 frames; accept when last 2 predictions agree with conf ≥ 0.8; 600 ms cooldown;
  re-arm after one "no gesture". Inference never on the Android main thread.
- Window labelling (Week 3, [tune]): gesture if the window contains the extended span [s-10, s+L-1] of that
  recording's own labelled span; windows with partial overlap are excluded; never shift dataset labels. Reason: users
  9/11 label ~8 frames after the motion, users 2/4 label over the motion (figure 2bca1f5).
  Background/false-trigger frames must lie outside [s-20, s+L+9] (>=10 frames from the extended span).

## Repo layout
dsp/ (radarsense_dsp + tests, fixtures in dsp/tests/fixtures) · training/ · backend/replay · backend/api ·
backend/caddy · android/ (com.radarsense.app, minSdk 26, targetSdk 34) · data/ (junction, ignored) ·
docker-compose.yml · .github/workflows/ci.yml

## How to work
- Python: ALWAYS use the repo venv — `.venv\Scripts\python.exe` (3.12.10). Never bare `python`/`pip`/`py`,
  never install packages outside the venv. Lint: `.venv\Scripts\python.exe -m ruff check .`
  Test: `.venv\Scripts\python.exe -m pytest`
- Run all tests from the repo root. Each backend service gets a uniquely named package (api: app; replay: replay, never app).
- Small, testable units. Write the test first for DSP (synthetic targets at known range/Doppler bins).
- Branches: `feat/...`, `exp/...`, merged to main by PR with a **merge commit** (not squash).
- If a task seems to require breaking a hard rule, stop and ask.
