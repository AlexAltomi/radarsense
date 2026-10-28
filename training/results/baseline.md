# Baseline: hand-crafted window features

This report was produced at commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d`.

The model is trained on the nominal files of the train users and evaluated on the nominal files of the validation users only. Test users are never loaded.

## Class map

`{'label_to_class': {'0': 0, '1': 1, '2': 2, '3': 3, '6': 4, '7': 5}, 'class_names': ['none', 'label1', 'label2', 'label3', 'label6', 'label7']}`. The mapping is defined only in `dsp/radarsense_dsp/windows.py`. Not yet verified: the real gesture names.

## Window counts (commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d`)

| split | 0 none | 1 label1 | 2 label2 | 3 label3 | 4 label6 | 5 label7 | total | recordings with a no-gesture window |
|---|---|---|---|---|---|---|---|---|
| train | 100546 | 14845 | 15499 | 15455 | 14523 | 15513 | 176381 | 98.5% |
| val | 7523 | 1504 | 1495 | 1493 | 1438 | 1509 | 14962 | 98.0% |

## Cache members used (commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d`)

- user11_e1 (train, n=1000)
- user1_e1 (train, n=2000)
- user1_e2 (train, n=2000)
- user1_e3 (train, n=2000)
- user1_e4 (train, n=2000)
- user1_e5 (train, n=2000)
- user1_e6 (train, n=2000)
- user2_e1 (train, n=3000)
- user3_e1 (val, n=1000)
- user4_e1 (train, n=1000)
- user5_e1 (val, n=1000)
- user6_e1 (train, n=1000)
- user7_e1 (train, n=1000)
- user8_e1 (train, n=1000)
- user9_e1 (train, n=1000)

Features: 30 per window (range_centroid, doppler_centroid, energy, axis_a, axis_b x mean, min, max, slope, argmax, argmin). The report uses 176381 train windows and 14962 validation windows.

## baseline-lr-v1

These results were produced at commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-lr-v1`.

- overall accuracy: 0.9101
- macro-F1 (classes present in val): 0.8937

Per-class recall (commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-lr-v1`):

| class | recall | val windows |
|---|---|---|
| 0 none | 0.9295 | 7523 |
| 1 label1 | 0.8697 | 1504 |
| 2 label2 | 0.9405 | 1495 |
| 3 label3 | 0.8419 | 1493 |
| 4 label6 | 0.8811 | 1438 |
| 5 label7 | 0.9185 | 1509 |

Confusion matrix with the true class in the rows and the predicted class in the columns (commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-lr-v1`):

| true \ pred | 0 none | 1 label1 | 2 label2 | 3 label3 | 4 label6 | 5 label7 |
|---|---|---|---|---|---|---|
| 0 none | 6993 | 23 | 200 | 59 | 84 | 164 |
| 1 label1 | 14 | 1308 | 27 | 18 | 18 | 119 |
| 2 label2 | 75 | 1 | 1406 | 11 | 2 | 0 |
| 3 label3 | 131 | 87 | 7 | 1257 | 4 | 7 |
| 4 label6 | 124 | 29 | 11 | 1 | 1267 | 6 |
| 5 label7 | 99 | 1 | 0 | 11 | 12 | 1386 |

Accuracy for each validation user (commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-lr-v1`):

| user | accuracy | windows |
|---|---|---|
| user3 | 0.8903 | 7164 |
| user5 | 0.9283 | 7798 |

## baseline-rf-v1

These results were produced at commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-rf-v1`.

- overall accuracy: 0.8765
- macro-F1 (classes present in val): 0.8439

Per-class recall (commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-rf-v1`):

| class | recall | val windows |
|---|---|---|
| 0 none | 0.9947 | 7523 |
| 1 label1 | 0.7806 | 1504 |
| 2 label2 | 0.7331 | 1495 |
| 3 label3 | 0.6209 | 1493 |
| 4 label6 | 0.8046 | 1438 |
| 5 label7 | 0.8463 | 1509 |

Confusion matrix with the true class in the rows and the predicted class in the columns (commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-rf-v1`):

| true \ pred | 0 none | 1 label1 | 2 label2 | 3 label3 | 4 label6 | 5 label7 |
|---|---|---|---|---|---|---|
| 0 none | 7483 | 0 | 7 | 25 | 4 | 4 |
| 1 label1 | 26 | 1174 | 20 | 9 | 0 | 275 |
| 2 label2 | 348 | 1 | 1096 | 3 | 47 | 0 |
| 3 label3 | 499 | 54 | 1 | 927 | 0 | 12 |
| 4 label6 | 271 | 10 | 0 | 0 | 1157 | 0 |
| 5 label7 | 224 | 1 | 0 | 5 | 2 | 1277 |

Accuracy for each validation user (commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-rf-v1`):

| user | accuracy | windows |
|---|---|---|
| user3 | 0.7786 | 7164 |
| user5 | 0.9664 | 7798 |
