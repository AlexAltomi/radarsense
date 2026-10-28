# RadarSense v1: project rules

This document collects the rules, dataset facts and specifications that the code in this repository follows.
Where a fact has not been confirmed yet, it is marked "Not yet verified:". If a task would break one of these
rules, stop and ask before going on.

## Project

RadarSense v1 is an Android app that recognizes five hand gestures and a "no gesture" state from replayed
recordings of an Infineon 60 GHz FMCW (frequency-modulated continuous-wave) radar. The recordings come from a public
dataset on Zenodo, licensed under CC BY 4.0 (DOI 10.5281/zenodo.15178095). The app does not talk to radar hardware.
A replay server streams recorded frames to the app, which classifies them.

The work is a solo project for the Software Development lab at UPT. It has no budget (€0) and runs from 19 October
2026 to 10 January 2027. If these rules and the project plan document disagree, the project plan wins, and the
disagreement should be raised before any work is done.

**Scope.** The following are out of scope for version 1: live radar hardware, custom gestures, publication on the
Play Store, cloud production hosting, and thesis-level signal processing. No feature that is not in the plan is
added, and nothing new is started before Week 9.

**Repository layout.**

- `dsp/`: the `radarsense_dsp` package and its tests. Test fixtures are in `dsp/tests/fixtures`.
- `training/`: dataset scripts, the frozen split and the baseline model.
- `backend/replay`, `backend/api` and `backend/caddy`: the replay server, the API and the reverse proxy.
- `android/`: the app (package `com.radarsense.app`, minSdk 26, targetSdk 34).
- `data/`: a junction to `D:\radarsense-data`, ignored by git.
- `docker-compose.yml` and `.github/workflows/ci.yml`: deployment and continuous integration.

## Data rules

These rules are never broken, not even temporarily.

1. **Test users stay locked until Week 11.** The data of user10 and user12 is not loaded before Week 11. The loader
   refuses these users unless the environment variable `RADARSENSE_ALLOW_TEST=1` is set. That variable is never set,
   and no statistic, normalization constant or plot may include these users.
2. **The split is frozen.** `training/splits.json` (frozen at commit c546264) assigns users as follows.
   - Train: user1, user2, user4, user6, user7, user8, user9, user11.
   - Validation: user3, user5.
   - Test: user10, user12.
   - Robustness evaluation uses the anomalous recordings (fast, slow and wrist). The headline robustness number
     comes from the anomalous files of user10 and user12 only. The accuracy on anomalous files of train users is
     reported separately as a secondary number, because those users were seen in training.

   `splits.json` is never edited. Normalization constants and the label and antenna inference use train users only.
3. **Training uses nominal files only.** Anomalous files (fast, slow, wrist) are used only for the robustness
   evaluation.
4. **No mirroring augmentation.** Range-Doppler maps and angle maps are never mirrored, because mirroring swaps the
   left/right and up/down labels. The allowed augmentations are a frame shift of up to 3 frames, an amplitude change
   of up to 3 dB, and small noise.
5. **Data never enters git.** `data/` is a junction to `D:\radarsense-data` and is listed in `.gitignore`.
6. **The zip file is streamed, never extracted or loaded whole.** The archive is 36.5 GB. Each `.npz` member holds an
   `inputs` array of float32 values with shape (N, 100, 3, 32, 64), which takes about 2.46 GB per 1000 recordings.
   N varies per file (about 1000 to 3000) and is read from the `.npy` header. Code streams one recording at a time
   and never loads a whole `inputs` array. A single `.npz` may be extracted temporarily to `D:\radarsense-data\tmp`
   and must be deleted after processing.
7. **Scope contract.** The scope listed under "Project" is binding. Features that are not in the plan are not added.
8. **Every reported metric is traceable.** Each metric is recorded together with the git commit and the model
   version that produced it.

## Dataset facts

All facts in this section were checked locally in Week 1, except those marked "Not yet verified:".

- The file `radar_dataset.zip` contains 41 `.npz` members (md5 d63a9645f27ebded613807b674bba63a). Their names have
  the form `fulldataset/user{N}_e{S}[_{fast|slow|wrist}].npz`. User1 has sessions e1 to e6, and every other user has
  e1 only. The outer members are compressed with ZIP_DEFLATED.
- `inputs` is a float32 array with shape (recordings, 100 frames, 3 antennas, 32 chirps, 64 samples). `targets` is an
  int32 array with shape (recordings, 100). The number of recordings N varies per file (about 1000 to 3000) and is
  read from the `.npy` header.
- The label values are 0, 1, 2, 3, 6 and 7. Label 0 means "no gesture".
- User1 supplies 57% of the nominal training recordings. Training balances the users, and the cache keeps all
  recordings.
- Every recording contains exactly one contiguous gesture. This was verified recording by recording for user4_e1
  (span of 8 frames) and user9_e1 (span of 10 frames), and no recording is free of gestures. For the other nominal
  files the span length is inferred from the label counts in the inventory, assuming 200 recordings per class: 8
  frames for users 1 to 8, and 10 frames for users 9 and 11. For anomalous files the span length differs from file to
  file (8, 10 or a mixture), so it is never assumed and always read from `targets`. The cache builder checks this for
  every recording.
- Not yet verified: the mapping from label values to gesture names. It will be inferred from the train users in
  Week 2.
- Antenna geometry, observed on the train users 2, 4, 9 and 11: antenna 2 sits in the corner of the L-shaped
  arrangement. The orthogonal antenna pairs are (0,2) and (1,2). The pair (0,1) is the diagonal; it responds to all
  swipes and is never used. Labels 2 and 3 are opposite sweeps on the pair (0,2). Labels 6 and 7 are opposite sweeps
  on the pair (1,2). Label 1 shows no consistent angular sweep.
- Not yet verified: that label 1 is a push, which axis is horizontal, and the direction within each axis.

## Signal processing

Each radar frame is turned into an image of 3 channels of 32 by 32 values. The steps are, in this order:

1. Subtract the mean over the chirps.
2. Apply a Hann window (a smooth taper that reduces spectral leakage) and a range FFT (fast Fourier transform), and
   keep the 32 positive range bins.
3. Subtract the mean over the chirps for each range bin.
4. Apply a Hann window and a Doppler FFT, then `fftshift` so that zero Doppler is at bin 16.
5. Build the three channels.
   - Channel 0 is the magnitude in dB, averaged over the 3 antennas.
   - Channel 1 is sin(Δφ) of the antenna pair (0,2), called axis A (labels 2 and 3).
   - Channel 2 is sin(Δφ) of the antenna pair (1,2), called axis B (labels 6 and 7).
   - Channels 1 and 2 are weighted by the normalized magnitude, which is the scaled channel 0.
6. Clip channel 0 to a fixed dB range and scale it to 0 to 1. The range constants are computed from the train users
   and stored in `dsp/radarsense_dsp/constants.json`.
7. Store the result as float16.

A processed frame has the layout `[channel, range, Doppler]`, which is (3, 32, 32).

Not yet verified: that an approaching hand has a negative signed Doppler (bins below 16). The figure that suggests
this was produced at commit 2bca1f5. No left, right, up or down names appear in the code until the axes are
verified.

**Output to the app.**

- Wire format: frames of 3×32×32 uint8 values. A binary frame has a 16-byte header followed by 3,072 bytes of data,
  3,088 bytes in total. The header holds a uint32 sequence number, a uint64 server timestamp in milliseconds, a uint8
  true label (255 if unknown) and 3 reserved bytes.
- Detector: inference runs every 4 frames. A gesture is accepted when the last 2 predictions agree and the
  confidence is at least 0.8. After an accepted gesture there is a cooldown of 600 ms, and the detector re-arms
  after one "no gesture" prediction. Inference never runs on the Android main thread.

## Cache format

The processed cache lives in `data/processed`.

- `<stem>.npy` holds float16 frames with shape (N, 100, 3, 32, 32).
- `<stem>.targets.npy` holds int8 targets with shape (N, 100).
- `index.json` describes every member.

The build is resumable. Frames are written to a `.partial` file that is renamed when it is complete, and finished
members are skipped on the next run. For every recording, `build_cache.py` asserts that there is exactly one
contiguous gesture span. It records the span-length histogram and the violation count for each file in `index.json`.
When a file has a violation, the builder skips that file, marks it with the status "violation" together with the
first offending indices, continues with the other files, and exits with code 1 and a summary. Code that reads the
cache must never read a file whose index status is not "complete". The cache never contains user10 or user12.

## Training windows

The classifier sees windows of 32 frames, taken every 4 frames, which matches the detector's inference cadence. For a
recording with labelled span [s, s+L-1]:

- A window is a gesture window if it contains the extended span [s-10, s+L-1] completely. The extension exists
  because users 9 and 11 label the gesture about 8 frames after the motion, while users 2 and 4 label over the motion
  (figure produced at commit 2bca1f5).
- A window is a no-gesture window if it does not overlap [s-20, s+L+9]. In other words, background and false-trigger
  frames stay at least 10 frames away from the extended span.
- Every other window is excluded and is never labelled as no-gesture.
- Dataset labels are never shifted.

Labels 0, 1, 2, 3, 6 and 7 map to classes 0 to 5. The mapping, the window rules and the per-user balancing are
defined in one place, `dsp/radarsense_dsp/windows.py`.

## Development workflow

- Python code always runs with the repository's virtual environment: `.venv\Scripts\python.exe` (Python 3.12.10).
  Bare `python`, `pip` and `py` are not used, and no package is installed outside the venv.
  - Lint: `.venv\Scripts\python.exe -m ruff check .`
  - Test: `.venv\Scripts\python.exe -m pytest`
- All tests are run from the repository root. Each backend service has a uniquely named package: `app` for the API
  and `replay` for the replay server (never `app`).
- The pre-commit hook is installed once per clone with `cp scripts/pre-commit .git/hooks/pre-commit`. It blocks a
  commit when ruff or pytest fails.
- Work is done in small, testable units. For signal processing the test is written first, using synthetic targets at
  known range and Doppler bins.
- Branches are named `feat/...` or `exp/...` and are merged into main by pull request with a merge commit, not a
  squash.
