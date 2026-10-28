# Dataset inventory

This file describes the Infineon 60 GHz radar gesture dataset that RadarSense uses. It contains counts only and no radar data. The scripts in `training/` generate it. `inventory.py` writes the glossary and the member list, `label_structure.py` and `infer_labels_antennas.py` add their own sections, and the section on constants is kept by hand from the output of `compute_constants.py`.

## Glossary

- **Member**: one `.npz` file of the dataset, for example `user4_e1` or `user9_e1_fast`. The name gives the user, the session and, for anomalous members, the type.
- **Recording**: one gesture example inside a member. A member holds N recordings.
- **Frame**: one radar snapshot. Every recording consists of 100 frames, and every frame has a label.
- **Nominal**: a member whose gestures were performed in the normal way.
- **Anomalous**: a member that deviates from the normal way, namely the fast, slow and wrist variants. These members are used only to test robustness.
- **Labelled span**: the run of consecutive frames in a recording that carry a non-zero label. Label 0 means that no gesture is happening.
- **Train, validation and test users**: the split of the people in the dataset, frozen in `training/splits.json`. The train users are user1, user2, user4, user6, user7, user8, user9 and user11. The validation users are user3 and user5. The test users are user10 and user12, who stay locked until Week 11.

## Members

This table lists every member of the dataset with its user, session, type, number of recordings N and label counts. The label counts cover all frames of all recordings in the member and are written as `value:frames`. The members of the test users are listed as locked, because their data is not opened.

| member | user | session | type | N | label counts |
|---|---|---|---|---|---|
| user10_e1 | 10 | e1 | nominal | locked until Week 11 |  |
| user10_e1_fast | 10 | e1 | fast | locked until Week 11 |  |
| user10_e1_slow | 10 | e1 | slow | locked until Week 11 |  |
| user10_e1_wrist | 10 | e1 | wrist | locked until Week 11 |  |
| user11_e1 | 11 | e1 | nominal | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user11_e1_fast | 11 | e1 | fast | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user11_e1_slow | 11 | e1 | slow | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user11_e1_wrist | 11 | e1 | wrist | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user12_e1 | 12 | e1 | nominal | locked until Week 11 |  |
| user12_e1_fast | 12 | e1 | fast | locked until Week 11 |  |
| user12_e1_slow | 12 | e1 | slow | locked until Week 11 |  |
| user12_e1_wrist | 12 | e1 | wrist | locked until Week 11 |  |
| user1_e1 | 1 | e1 | nominal | 2000 | 0:184000 1:3200 2:3200 3:3200 6:3200 7:3200 |
| user1_e1_fast | 1 | e1 | fast | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user1_e1_slow | 1 | e1 | slow | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user1_e1_wrist | 1 | e1 | wrist | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user1_e2 | 1 | e2 | nominal | 2000 | 0:184000 1:3200 2:3200 3:3200 6:3200 7:3200 |
| user1_e3 | 1 | e3 | nominal | 2000 | 0:184000 1:3200 2:3200 3:3200 6:3200 7:3200 |
| user1_e4 | 1 | e4 | nominal | 2000 | 0:184000 1:3200 2:3200 3:3200 6:3200 7:3200 |
| user1_e5 | 1 | e5 | nominal | 2000 | 0:184000 1:3200 2:3200 3:3200 6:3200 7:3200 |
| user1_e6 | 1 | e6 | nominal | 2000 | 0:184000 1:3200 2:3200 3:3200 6:3200 7:3200 |
| user2_e1 | 2 | e1 | nominal | 3000 | 0:276000 1:4800 2:4800 3:4800 6:4800 7:4800 |
| user2_e1_fast | 2 | e1 | fast | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user2_e1_slow | 2 | e1 | slow | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user2_e1_wrist | 2 | e1 | wrist | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user3_e1 | 3 | e1 | nominal | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user4_e1 | 4 | e1 | nominal | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user5_e1 | 5 | e1 | nominal | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user6_e1 | 6 | e1 | nominal | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user6_e1_fast | 6 | e1 | fast | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user6_e1_slow | 6 | e1 | slow | 1000 | 0:91500 1:2000 2:1700 3:1600 6:1600 7:1600 |
| user6_e1_wrist | 6 | e1 | wrist | 1000 | 0:90700 1:2000 2:1700 3:1600 6:2000 7:2000 |
| user7_e1 | 7 | e1 | nominal | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user7_e1_fast | 7 | e1 | fast | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user7_e1_slow | 7 | e1 | slow | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user7_e1_wrist | 7 | e1 | wrist | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user8_e1 | 8 | e1 | nominal | 1000 | 0:92000 1:1600 2:1600 3:1600 6:1600 7:1600 |
| user9_e1 | 9 | e1 | nominal | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user9_e1_fast | 9 | e1 | fast | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user9_e1_slow | 9 | e1 | slow | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |
| user9_e1_wrist | 9 | e1 | wrist | 1000 | 0:90000 1:2000 2:2000 3:2000 6:2000 7:2000 |

## Label structure (train users)

This section checks how the labelled frames are arranged inside the recordings of two train members, user4_e1 and user9_e1. It reports counts only. A frame is a gesture frame if its label is not 0. The start frame, span and run statistics cover only the recordings that contain a gesture.

### user4_e1 (N=1000)
- Labelled frames per recording (`frames:recordings`): 8:1000
- Recordings with no gesture frames: 0
- Recordings with more than one distinct gesture value: 0
- Gesture start frame: minimum 21, median 40, maximum 70. Counts in 10-frame bins (`bin_start:recordings`): 20:106 30:360 40:372 50:128 60:28 70:6
- Gesture span, which is the last labelled frame minus the first plus 1 (`frames:recordings`): 8:1000
- Contiguous runs of labelled frames per recording (`runs:recordings`): 1:1000
- Recordings whose gesture is a single contiguous run: 1000 of 1000.

### user9_e1 (N=1000)
- Labelled frames per recording (`frames:recordings`): 10:1000
- Recordings with no gesture frames: 0
- Recordings with more than one distinct gesture value: 0
- Gesture start frame: minimum 30, median 44, maximum 78. Counts in 10-frame bins (`bin_start:recordings`): 30:312 40:414 50:216 60:48 70:10
- Gesture span, which is the last labelled frame minus the first plus 1 (`frames:recordings`): 10:1000
- Contiguous runs of labelled frames per recording (`runs:recordings`): 1:1000
- Recordings whose gesture is a single contiguous run: 1000 of 1000.

## Label and antenna inference (exploratory)

This section looks for the antenna pair and the sweep direction that belong to each label. It uses only the nominal e1 files of the train users user2, user4, user9, user11, and up to 40 recordings per label, with all frames aligned to the start of the labelled span. In every frame it takes the strongest cell of the unclipped dB map. The Doppler value is the signed bin (bin - 16), averaged over the cells within 20 dB of the peak and weighted by their energy. The figure is `training/figures/label_antenna_inference.png`. The dB percentiles are for orientation only and are not used as constants. In the figure the phase change falls inside the labelled span for user2 and user4, but about 8 frames before the span start for user9 and user11. Users 9 and 11 therefore label the gesture about 8 frames after the motion. The slope tables below are taken over the labelled span, so they cannot be compared between these two groups. Compare the figure instead.

- user2_e1: read 219 recordings, kept per label `1:40 2:40 3:40 6:40 7:40`, skipped 0. The 1st and 99.5th percentiles of the raw dB values are 13.9 and 52.3.
- user4_e1: read 230 recordings, kept per label `1:40 2:40 3:40 6:40 7:40`, skipped 0. The 1st and 99.5th percentiles of the raw dB values are 14.0 and 53.7.
- user9_e1: read 226 recordings, kept per label `1:40 2:40 3:40 6:40 7:40`, skipped 0. The 1st and 99.5th percentiles of the raw dB values are 13.9 and 51.5.
- user11_e1: read 246 recordings, kept per label `1:40 2:40 3:40 6:40 7:40`, skipped 0. The 1st and 99.5th percentiles of the raw dB values are 13.9 and 48.9.

Mean slope of the phase difference delta_phi = angle(ant1) - angle(ant0) over the labelled span, in rad/frame. Brackets show the share of recordings with a positive slope.

| label | user2 | user4 | user9 | user11 | all |
|---|---|---|---|---|---|
| 1 | +0.201 (95%) | +0.039 (78%) | -0.200 (0%) | -0.108 (0%) | -0.017 (43%) |
| 2 | +0.417 (100%) | +0.617 (100%) | -0.120 (5%) | -0.111 (18%) | +0.201 (56%) |
| 3 | -0.308 (0%) | -0.156 (20%) | -0.083 (8%) | -0.111 (8%) | -0.165 (9%) |
| 6 | +0.444 (100%) | +0.576 (95%) | -0.061 (15%) | -0.120 (10%) | +0.210 (55%) |
| 7 | -0.459 (0%) | -0.494 (8%) | -0.010 (40%) | -0.030 (25%) | -0.248 (18%) |

Mean slope of the phase difference delta_phi = angle(ant2) - angle(ant0) over the labelled span, in rad/frame. Brackets show the share of recordings with a positive slope.

| label | user2 | user4 | user9 | user11 | all |
|---|---|---|---|---|---|
| 1 | -0.001 (68%) | +0.027 (68%) | -0.093 (2%) | -0.059 (10%) | -0.031 (37%) |
| 2 | +0.376 (100%) | +0.559 (100%) | -0.028 (35%) | -0.014 (35%) | +0.223 (68%) |
| 3 | -0.406 (0%) | -0.338 (0%) | -0.066 (10%) | -0.092 (5%) | -0.226 (4%) |
| 6 | +0.122 (98%) | +0.142 (100%) | -0.027 (8%) | -0.048 (12%) | +0.047 (54%) |
| 7 | -0.130 (2%) | -0.098 (5%) | -0.034 (10%) | -0.008 (40%) | -0.067 (14%) |

Mean slope of the phase difference delta_phi = angle(ant2) - angle(ant1) over the labelled span, in rad/frame. Brackets show the share of recordings with a positive slope.

| label | user2 | user4 | user9 | user11 | all |
|---|---|---|---|---|---|
| 1 | -0.102 (10%) | -0.107 (20%) | +0.121 (95%) | +0.049 (92%) | -0.010 (54%) |
| 2 | -0.041 (18%) | -0.058 (18%) | +0.092 (98%) | +0.082 (90%) | +0.019 (56%) |
| 3 | -0.098 (0%) | -0.183 (2%) | +0.018 (75%) | +0.039 (82%) | -0.056 (40%) |
| 6 | -0.322 (0%) | -0.458 (0%) | +0.034 (75%) | +0.073 (82%) | -0.168 (39%) |
| 7 | +0.329 (100%) | +0.409 (95%) | -0.024 (20%) | +0.022 (72%) | +0.184 (72%) |

## Constants

This section lists the constants that the signal processing uses to scale channel 0 and to choose the antenna pairs. They are stored in `dsp/radarsense_dsp/constants.json`. The value `db_min` is the 50th percentile and `db_max` is the 99.9th percentile of the raw dB values, which are averaged over the antennas, with every train user weighted equally. Only nominal files were used, with up to 40 recordings per label per file and all 100 frames. The script is `training/compute_constants.py`, and the numbers were produced at commit c6381f8. The antenna pairs are `pair_a` = (0,2) and `pair_b` = (1,2).

| user | nominal files | db_min (p50) | db_max (p99.9) |
|---|---|---|---|
| 1 | 6 (e1-e6) | 24.53 | 63.47 |
| 2 | 1 | 24.41 | 64.07 |
| 4 | 1 | 24.49 | 67.23 |
| 6 | 1 | 24.33 | 59.56 |
| 7 | 1 | 24.32 | 58.87 |
| 8 | 1 | 24.25 | 54.03 |
| 9 | 1 | 24.33 | 64.31 |
| 11 | 1 | 24.31 | 61.58 |
| **used (equal per user)** | 13 | **24.38** | **62.35** |
| pooled (user1-heavy, for comparison) | 13 | 24.43 | 62.83 |

The value `db_max` varies between 54 and 67 dB across users, because users differ in gesture strength. The noise floor is identical for all users. The constants are fixed on purpose, and amplitude augmentation is applied in Week 3 instead.
