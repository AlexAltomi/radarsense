# cnn-v1: per-frame CNN with a temporal head

This report was produced at commit `be00a630abf133a8d21f5079a30b3bd91628113d` · cache `4ecddb8cb8697e2d` · model `cnn-v1`.

The checkpoint is the best epoch (6) of a run trained at commit `11eb67351d5801936e4e9932fd4422ae7c80722c+dirty` on cache `4ecddb8cb8697e2d`, chosen by the macro-F1 on the validation users. It is evaluated on the nominal files of the validation users user3 and user5 only. Test users are never loaded.

## Class map

`{'label_to_class': {'0': 0, '1': 1, '2': 2, '3': 3, '6': 4, '7': 5}, 'class_names': ['none', 'label1', 'label2', 'label3', 'label6', 'label7']}`. The mapping is defined only in `dsp/radarsense_dsp/windows.py`. Not yet verified: the real gesture names.

## cnn-v1

These results were produced at commit `be00a630abf133a8d21f5079a30b3bd91628113d` · cache `4ecddb8cb8697e2d` · model `cnn-v1`.

- overall accuracy: 0.9668
- macro-F1 (classes present in val): 0.9610 (epoch 6, chosen on the validation users)

Per-class recall (commit `be00a630abf133a8d21f5079a30b3bd91628113d` · cache `4ecddb8cb8697e2d` · model `cnn-v1`):

| class | recall | val windows |
|---|---|---|
| 0 none | 0.9745 | 7523 |
| 1 label1 | 0.9947 | 1504 |
| 2 label2 | 0.9773 | 1495 |
| 3 label3 | 0.9799 | 1493 |
| 4 label6 | 0.8449 | 1438 |
| 5 label7 | 0.9934 | 1509 |

Confusion matrix with the true class in the rows and the predicted class in the columns (commit `be00a630abf133a8d21f5079a30b3bd91628113d` · cache `4ecddb8cb8697e2d` · model `cnn-v1`):

| true \ pred | 0 none | 1 label1 | 2 label2 | 3 label3 | 4 label6 | 5 label7 |
|---|---|---|---|---|---|---|
| 0 none | 7331 | 56 | 25 | 65 | 45 | 1 |
| 1 label1 | 0 | 1496 | 0 | 0 | 0 | 8 |
| 2 label2 | 30 | 1 | 1461 | 3 | 0 | 0 |
| 3 label3 | 22 | 8 | 0 | 1463 | 0 | 0 |
| 4 label6 | 143 | 75 | 0 | 0 | 1215 | 5 |
| 5 label7 | 6 | 4 | 0 | 0 | 0 | 1499 |

The matrix is also saved as `training/figures/cnn_confusion.png`.

Accuracy for each validation user (commit `be00a630abf133a8d21f5079a30b3bd91628113d` · cache `4ecddb8cb8697e2d` · model `cnn-v1`):

| user | accuracy | windows |
|---|---|---|
| user3 | 0.9408 | 7164 |
| user5 | 0.9906 | 7798 |

## Training history

The validation macro-F1 of every epoch of the training run (commit `11eb67351d5801936e4e9932fd4422ae7c80722c+dirty` · cache `4ecddb8cb8697e2d` · model `cnn-v1`). Epoch 6 was chosen because it has the highest macro-F1 on the validation users, which are also the users this report is computed on. The number at the chosen epoch is therefore the best of a noisy sequence and is optimistic.

| epoch | train loss | val loss | val macro-F1 | val accuracy |
|---|---|---|---|---|
| 1 | 0.3791 | 0.5352 | 0.7173 | 0.8272 |
| 2 | 0.0507 | 0.2781 | 0.8897 | 0.9138 |
| 3 | 0.0336 | 0.2304 | 0.9035 | 0.9252 |
| 4 | 0.0309 | 0.1819 | 0.9408 | 0.9450 |
| 5 | 0.0230 | 0.1579 | 0.9382 | 0.9462 |
| 6 (chosen) | 0.0245 | 0.1243 | 0.9610 | 0.9668 |
| 7 | 0.0185 | 0.2567 | 0.8962 | 0.9143 |
| 8 | 0.0171 | 0.1995 | 0.9223 | 0.9332 |
| 9 | 0.0154 | 0.2486 | 0.9065 | 0.9214 |
| 10 | 0.0147 | 0.3601 | 0.8630 | 0.8900 |
| 11 | 0.0133 | 0.2406 | 0.9044 | 0.9246 |

Last 5 epochs (epochs 7 to 11): mean macro-F1 0.8985, standard deviation 0.0219, minimum 0.8630, maximum 0.9223. Over all 11 epochs the macro-F1 ranges from 0.7173 to 0.9610.

## Comparison with the baselines

All three models are evaluated on the same validation windows. Each row carries the stamp of the run that produced it.

| model | accuracy | macro-F1 | user3 | user5 | stamp |
|---|---|---|---|---|---|
| cnn-v1 | 0.9668 | 0.9610 | 0.9408 | 0.9906 | commit `be00a630abf133a8d21f5079a30b3bd91628113d` · cache `4ecddb8cb8697e2d` · model `cnn-v1` |
| baseline-lr-v1 | 0.9101 | 0.8937 | 0.8903 | 0.9283 | commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-lr-v1` |
| baseline-rf-v1 | 0.8765 | 0.8439 | 0.7786 | 0.9664 | commit `3bec92ffb66af1d554748b3ffc0018242685e56e` · cache `4ecddb8cb8697e2d` · model `baseline-rf-v1` |

Per-class recall (commit `be00a630abf133a8d21f5079a30b3bd91628113d` · cache `4ecddb8cb8697e2d` · model `cnn-v1`):

| class | cnn-v1 | baseline-lr-v1 | baseline-rf-v1 |
|---|---|---|---|
| 0 none | 0.9745 | 0.9295 | 0.9947 |
| 1 label1 | 0.9947 | 0.8697 | 0.7806 |
| 2 label2 | 0.9773 | 0.9405 | 0.7331 |
| 3 label3 | 0.9799 | 0.8419 | 0.6209 |
| 4 label6 | 0.8449 | 0.8811 | 0.8046 |
| 5 label7 | 0.9934 | 0.9185 | 0.8463 |

cnn-v1 minus baseline-lr-v1 in macro-F1: +0.0673.

cnn-v1 minus baseline-rf-v1 in macro-F1: +0.1171.

## Windows inside the rule and near its edge

The margin of a window is the number of frames it could slide towards the edge of its rule before the label rule of `window_class_at` changes its class. For a gesture window it is the smaller free space around the extended span. For a no-gesture window it is the number of frames it could move towards the gesture and stay 10 frames away. A window is near the edge when its margin is below 3 frames, which is the largest shift used in training. The macro-F1 of a group averages over the classes that occur in that group, with the predictions of the whole model.

Windows by group and class (commit `be00a630abf133a8d21f5079a30b3bd91628113d` · cache `4ecddb8cb8697e2d` · model `cnn-v1`):

| group | windows | accuracy | macro-F1 | 0 none | 1 label1 | 2 label2 | 3 label3 | 4 label6 | 5 label7 |
|---|---|---|---|---|---|---|---|---|---|
| inside (margin of 3 frames or more) | 10531 | 0.9738 | 0.9659 | 5977 | 889 | 905 | 907 | 962 | 891 |
| near the edge (margin below 3 frames) | 4431 | 0.9501 | 0.9491 | 1546 | 615 | 590 | 586 | 476 | 618 |
