# cnn-v1 training runs

This table was produced at commit `f4c5e4def1b0f07c6f7b3d9abeda9b4a9e0f5e46` from the logs of the runs (`training/runs/<run>/log.csv` and `config.json`). Nothing was recomputed.

| run | note | seed | differs from the defaults | commit stamp | epochs run | best epoch | best val macro-F1 | last epoch | last val macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| `cnn-v1-e12` |  | 0 | max-epochs 12 | `3b0b57e` · cache `4ecddb8cb8697e2d` | 9 | 4 | 0.9448 | 9 | 0.9136 |
| `cnn-v1-e12-s1` |  | 1 | max-epochs 12 | `3b0b57e` · cache `4ecddb8cb8697e2d` | 8 | 3 | 0.9494 | 8 | 0.9103 |
| `cnn-v1-e12-s2` |  | 2 | max-epochs 12 | `3b0b57e` · cache `4ecddb8cb8697e2d` | 12 | 8 | 0.9355 | 12 | 0.9050 |
| `cnn-v1-wd5e2` | weight decay 5e-2, not adopted | 0 | max-epochs 12, weight-decay 0.05 | `3b0b57e` · cache `4ecddb8cb8697e2d` | 7 | 2 | 0.9445 | 7 | 0.9269 |
| `cnn-v1-clean` | default 30-epoch schedule | 0 | none | `1035dc8` · cache `4ecddb8cb8697e2d` | 11 | 6 | 0.9610 | 11 | 0.9044 |
| **mean** | seeds 0, 1, 2 |  |  |  |  |  | 0.9432 |  | 0.9096 |
| **range** | seeds 0, 1, 2 |  |  |  |  |  | 0.9355 to 0.9494 |  | 0.9050 to 0.9136 |

The best epoch is the first epoch with the highest validation macro-F1 (the checkpointed epoch). The validation users user3 and user5 also choose the epoch, so the best value is optimistic. The last epoch is where training stopped, by early stopping or at max-epochs. The mean and range rows cover the seed group only.
