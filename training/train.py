"""Train the cnn-v1 model on the nominal windows of the train users.

Usage: python training/train.py [--seed 0] [--epoch-windows 40000] [--num-workers 2] [--bench-workers]

The model sees windows of the train users only. The validation users user3 and user5 are used for early stopping and
for reporting, and nothing else. The loader refuses the test users, and this script never refers to them. Windows are
read one at a time from the memory-mapped cache, so the cache is never loaded into memory.

An epoch is a fixed number of sampled windows, not a pass over the data. Each epoch draws the same number of windows
from every user, and inside each user the gesture classes get equal shares and the no-gesture class gets twice a
gesture share (`BalancedUserSampler`). The loss is not weighted.

Each run writes to training/runs/<run>/ (TensorBoard events, log.csv and config.json) and the best checkpoint to
training/checkpoints/<run>/best.pt. Both folders are ignored by git.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dsp"))
sys.path.insert(0, str(ROOT / "training"))

from baseline import evaluate, per_user_accuracy
from build_cache import build_commit
from model import MODEL_NAME, GestureCNN, parameter_count
from train_data import MAX_DB, MAX_SHIFT, NOISE_STD, TrainStream, augment_batch, predict

from radarsense_dsp.cache import load_constants
from radarsense_dsp.windows import (
    N_CLASSES,
    BalancedUserSampler,
    CacheReader,
    build_window_index,
    index_digest,
    load_complete_members,
    load_splits,
)

CACHE = ROOT / "data" / "processed"
SPLITS = ROOT / "training" / "splits.json"
RUNS = ROOT / "training" / "runs"
CHECKPOINTS = ROOT / "training" / "checkpoints"
CSV_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_macro_f1", "val_accuracy", "lr", "epoch_seconds", "model", "commit", "cache",
]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: TrainStream, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )


def train_step(model, opt, scaler, sched, x, y, device, db_range) -> torch.Tensor:
    x = augment_batch(x.to(device, non_blocking=True).float(), db_range)
    y = y.to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        loss = F.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    sched.step()
    return loss.detach()


def bench_workers(args, dataset_args, device, db_range) -> None:
    """Time the real training step with 0 and with 2 data loader workers and print the windows per second."""
    for nw in (0, 2, 0, 2, 0, 2):
        seed_everything(args.seed)
        model = GestureCNN().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10_000)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        loader = make_loader(TrainStream(*dataset_args), args.batch_size, nw)
        model.train()
        it = iter(loader)
        for _ in range(20):  # warm-up: worker start, page cache, cuDNN
            train_step(model, opt, scaler, sched, *next(it), device, db_range)
        torch.cuda.synchronize()
        t0, n = time.perf_counter(), 100
        for _ in range(n):
            train_step(model, opt, scaler, sched, *next(it), device, db_range)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"num_workers={nw}: {n * args.batch_size / dt:.0f} windows/s ({dt / n * 1000:.1f} ms per batch).")
        del it, loader


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch-windows", type=int, default=40_000, help="sampled windows per epoch")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5, help="epochs without a better val macro-F1 before stopping")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--none-ratio", type=float, default=2.0, help="no-gesture share relative to one gesture class")
    ap.add_argument("--num-workers", type=int, default=2, help="data loader workers; 2 was faster than 0 in --bench-workers")
    ap.add_argument("--bench-workers", action="store_true", help="time num_workers 0 and 2, then exit")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)

    run = args.run_name or f"{MODEL_NAME}-seed{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    splits = load_splits(args.splits)
    members = load_complete_members(args.cache, splits)
    expected = {int(u.removeprefix("user")) for split in ("train", "val") for u in splits[split]}
    missing = sorted(expected - {m.user for m in members})
    if missing:
        print(f"ERROR: The cache has no complete member for the users {missing}. Finish the cache build first.", file=sys.stderr)
        return 2
    cache_id, commit = index_digest(members), build_commit()
    db_range = load_constants().db_max - load_constants().db_min

    wi = build_window_index(args.cache, members)
    train_rows, val_rows = wi.split_rows("train"), wi.split_rows("val")
    n_users = len(np.unique(train_rows["user"]))
    steps = (args.epoch_windows // n_users * n_users) // args.batch_size
    epoch_len = steps * args.batch_size
    sampler = BalancedUserSampler(
        train_rows["user"], per_user=args.epoch_windows // n_users, seed=args.seed,
        classes=train_rows["cls"], none_ratio=args.none_ratio,
    )
    dataset_args = (args.cache, members, train_rows, sampler, args.seed, epoch_len, args.max_epochs)
    print(f"Run {run}: commit {commit}, cache {cache_id}, device {device.type}, {steps} steps of {args.batch_size} windows per epoch.")

    model = GestureCNN().to(device)
    n_params = parameter_count(model)
    print(f"{MODEL_NAME} has {n_params} parameters. Train windows: {len(train_rows)}, validation windows: {len(val_rows)}.")
    share = np.bincount(train_rows["cls"][sampler.epoch_indices(0)[:epoch_len]], minlength=N_CLASSES) / epoch_len
    print("Class share of the sampled epoch 0:", ", ".join(f"{s:.3f}" for s in share))

    if args.bench_workers:
        bench_workers(args, dataset_args, device, db_range)
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs * steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loader = make_loader(TrainStream(*dataset_args), args.batch_size, args.num_workers)
    val_reader = CacheReader(args.cache, members)
    y_val = val_rows["cls"].astype(int)

    run_dir, ckpt_dir = RUNS / run, CHECKPOINTS / run
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "cache": str(args.cache), "splits": str(args.splits), "model": MODEL_NAME, "commit": commit, "cache_id": cache_id,
        "parameters": n_params, "steps_per_epoch": steps, "windows_per_epoch": epoch_len, "max_shift": MAX_SHIFT,
        "max_db": MAX_DB, "noise_std": NOISE_STD, "db_range": db_range, "device": device.type,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    writer = SummaryWriter(run_dir)
    writer.add_text("config", "```\n" + json.dumps(config, indent=2, default=str) + "\n```")

    best_f1, best_epoch, curve = -1.0, 0, []
    with open(run_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        csv_out = csv.DictWriter(f, CSV_FIELDS)
        csv_out.writeheader()
        it = iter(loader)
        for epoch in range(1, args.max_epochs + 1):
            t0 = time.perf_counter()
            model.train()
            loss_sum = torch.zeros((), device=device)
            for _ in range(steps):
                loss_sum += train_step(model, opt, scaler, sched, *next(it), device, db_range)
            train_loss = loss_sum.item() / steps

            logits = predict(model, val_reader, val_rows, device)
            pred = logits.argmax(axis=1)
            res = evaluate(y_val, pred)
            val_loss = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y_val)))
            seconds = time.perf_counter() - t0
            row = {
                "epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
                "val_macro_f1": round(res["macro_f1"], 5), "val_accuracy": round(res["accuracy"], 5),
                "lr": opt.param_groups[0]["lr"], "epoch_seconds": round(seconds, 1), "model": MODEL_NAME,
                "commit": commit, "cache": cache_id,
            }
            csv_out.writerow(row)
            f.flush()
            for k in ("train_loss", "val_loss", "val_macro_f1", "val_accuracy", "lr", "epoch_seconds"):
                writer.add_scalar(k, row[k], epoch)
            curve.append(row)

            improved = res["macro_f1"] > best_f1
            if improved:
                best_f1, best_epoch = res["macro_f1"], epoch
                torch.save(
                    {
                        "model_state": model.state_dict(), "model": MODEL_NAME, "epoch": epoch,
                        "val_macro_f1": best_f1, "commit": commit, "cache_id": cache_id, "config": config,
                    },
                    ckpt_dir / "best.pt",
                )
            print(
                f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  val macro-F1 {res['macro_f1']:.4f}  "
                f"val acc {res['accuracy']:.4f}  {seconds:.1f} s{'  *best*' if improved else ''}",
                flush=True,
            )
            if epoch - best_epoch >= args.patience:
                print(f"Early stopping: no better val macro-F1 for {args.patience} epochs.")
                break
    writer.close()

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    pred = predict(model, val_reader, val_rows, device).argmax(axis=1)
    users = per_user_accuracy(val_rows["user"], y_val, pred)
    mean_s = float(np.mean([r["epoch_seconds"] for r in curve]))
    print(f"Chosen epoch: {best_epoch} of {len(curve)} run. Best val macro-F1: {best_f1:.4f}.")
    print("Per-user accuracy at the chosen epoch:", ", ".join(f"user{u} {a:.4f}" for u, (a, _) in sorted(users.items())))
    print(f"Mean epoch time: {mean_s:.1f} s. Checkpoint: {ckpt_dir / 'best.pt'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
