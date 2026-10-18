# RadarSense v1

Android app that recognizes 5 hand gestures from replayed 60 GHz FMCW radar recordings.
Software Development lab project, UPT, 2026.

## Dataset credit (CC BY 4.0)
Frame-Labeled 60 GHz FMCW Radar Gesture Dataset (Infineon), Zenodo,
DOI: 10.5281/zenodo.15178095. The dataset is not included in this repo.

## Android notes
- targetSdk = 34 on purpose: apps targeting API 37+ are blocked from local-network
  addresses (ACCESS_LOCAL_NETWORK, Android 17), and the replay server is reached
  over the local network. Revisit only if the app ever needs to target 37+.
