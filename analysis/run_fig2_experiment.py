#!/usr/bin/env python3
"""Retrain one frozen exp041 Fig. 2 regional or single-band condition/fold."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from OURmodels.Base import base

BANDS = ((1, 4), (4, 7), (8, 13), (14, 29), (30, 47))
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
CONDITIONS = ("All", "region_F", "region_C", "region_P", "region_T", "region_O",
              "band_delta", "band_theta", "band_alpha", "band_beta", "band_gamma")
GRID = np.asarray([
    [-1, -1, -1, -1, 1, 33, 34, -1, -1, -1, -1],
    [-1, -1, -1, 2, 3, 37, 36, 35, -1, -1, -1],
    [-1, 7, 6, 5, 4, 38, 39, 40, 41, 42, -1],
    [-1, 8, 9, 10, 11, 47, 46, 45, 44, 43, -1],
    [-1, 15, 14, 13, 12, 48, 49, 50, 51, 52, -1],
    [-1, 16, 17, 18, 19, 32, 56, 55, 54, 53, -1],
    [24, 23, 22, 21, 20, 31, 57, 58, 59, 60, 61],
    [-1, -1, -1, 25, 26, 30, 63, 62, -1, -1, -1],
    [-1, -1, -1, -1, 27, 29, 64, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 28, -1, -1, -1, -1, -1],
], dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("KUL", "DTU"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_fig2"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true", help="Use two diagnostic epochs; not the paper protocol")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-local", action="store_true", help="Permit execution outside Slurm on a non-Huairou host")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reorder_trials(data, labels):
    ordered_data, ordered_labels, orders = np.empty_like(data), np.empty_like(labels), []
    for subject in range(data.shape[0]):
        trial_labels = labels[subject, :, 0].astype(int)
        zeros, ones = np.flatnonzero(trial_labels == 0), np.flatnonzero(trial_labels == 1)
        if len(zeros) != len(ones) or not np.all(labels[subject] == trial_labels[:, None]):
            raise ValueError("Fig. 2 requires balanced, constant-label trials")
        order = np.stack((zeros, ones), axis=1).reshape(-1)
        ordered_data[subject], ordered_labels[subject] = data[subject, order], labels[subject, order]
        orders.append(order.tolist())
    if data.shape[1] % 4:
        raise ValueError("Trial count must be divisible by four")
    return ordered_data, ordered_labels, orders


def split_trials(trials, fold):
    if trials % 4 or fold not in range(4):
        raise ValueError("Four-fold split requires divisible trial count and fold 0..3")
    quarter = trials // 4
    test = np.arange(fold * quarter, (fold + 1) * quarter)
    valid_start = ((fold + 1) % 4) * quarter
    valid = np.arange(valid_start, valid_start + quarter)
    train = np.setdiff1d(np.arange(trials), np.concatenate((test, valid)))
    return train, valid, test


def load_regions(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    regions = payload.get("regions", {})
    counts = {"F": 26, "C": 21, "P": 23, "T": 6, "O": 9}
    if set(regions) != set(counts) or any(len(regions[name]) != count for name, count in counts.items()):
        raise ValueError("regions.json does not match the frozen Fig. 2 masks")
    masks = {name: [int(channel) for channel in channels] for name, channels in regions.items()}
    if any(sorted(set(mask)) != sorted(mask) or min(mask) < 1 or max(mask) > 64 for mask in masks.values()):
        raise ValueError("Region masks must be unique, ascending 1-based channel IDs")
    return masks


def condition_masks(condition, regions):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    channels, bands = list(range(1, 65)), list(range(5))
    if condition.startswith("region_"):
        channels = regions[condition[-1]]
    elif condition.startswith("band_"):
        bands = [BAND_NAMES.index(condition[5:])]
    return channels, bands


def preprocess(data):
    result = np.empty((*data.shape[:2], 5, data.shape[2], data.shape[3]), dtype=np.float32)
    for index, (low, high) in enumerate(BANDS):
        b, a = butter(5, [low / 64.0, high / 64.0], btype="band")
        for subject in range(data.shape[0]):
            for trial in range(data.shape[1]):
                for channel in range(data.shape[3]):
                    result[subject, trial, index, :, channel] = filtfilt(b, a, data[subject, trial, :, channel])
    return result


def make_rows(subjects, trials, windows):
    return np.asarray([(subject, trial, window) for subject in range(subjects)
                       for trial in trials for window in range(windows)], dtype=np.int64)


def masked_grid_inputs(full_bands, index, channel_mask, grid_ids, grid_valid, band_ids):
    subject, trial, window = (index[:, col].long() for col in range(3))
    selected = full_bands[subject, trial]
    sample_ids = window[:, None] * 128 + torch.arange(128, device=window.device)[None, :]
    gather_ids = sample_ids[:, None, :, None].expand(-1, 5, -1, 64)
    values = torch.gather(selected, 2, gather_ids)
    values = values * channel_mask.reshape(1, 1, 1, 64)
    values = values[:, :, :, grid_ids].reshape(len(index), 5, 128, 10, 11) * grid_valid
    return tuple(values[:, band] if band in band_ids else torch.zeros_like(values[:, band]) for band in range(5))


def main(args=None):
    args = parse_args() if args is None else args
    if args.seed != 2025 or args.batch_size != 128 or args.epochs != (2 if args.smoke else 50):
        raise ValueError("Frozen settings require seed 2025, batch size 128, and 50 epochs (2 with --smoke)")
    if not args.allow_local and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Submit on Huairou through Slurm; use --allow-local only outside Huairou")
    set_seed(args.seed)
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    data_path = args.data_dir / f"{args.dataset}_1D.mat"
    with h5py.File(data_path, "r") as handle:
        raw, labels = np.asarray(handle["EEG"]).transpose().copy(), np.asarray(handle["ENV"]).transpose().copy()
    if raw.ndim != 4 or raw.shape[-1] != 64 or labels.shape != raw.shape[:3] or raw.shape[2] < 128 or raw.shape[2] % 128:
        raise ValueError("Expected EEG [subjects,trials,time,64], ENV [subjects,trials,time], with 128-sample windows")
    expected_subjects, expected_trials = (16, 8) if args.dataset == "KUL" else (21, 32)
    if not args.smoke and raw.shape[:2] != (expected_subjects, expected_trials):
        raise ValueError(f"{args.dataset} Fig. 2 requires {expected_subjects} subjects and {expected_trials} trials per subject")
    if not np.isfinite(raw).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("EEG must be finite and labels binary")
    data, labels, orders = reorder_trials(raw, labels)
    train_trials, valid_trials, test_trials = split_trials(data.shape[1], args.fold)
    windows = data.shape[2] // 128
    rows = {name: make_rows(data.shape[0], trial_ids, windows) for name, trial_ids in
            (("train", train_trials), ("validation", valid_trials), ("test", test_trials))}
    full_bands = torch.from_numpy(preprocess(data)).to(device=device)
    regions_path = ROOT / "configs" / "regions.json"
    channels, band_ids = condition_masks(args.condition, load_regions(regions_path))
    channel_mask = torch.zeros(64, device=device)
    channel_mask[torch.tensor(channels, device=device) - 1] = 1
    grid_ids = torch.as_tensor(np.maximum(GRID.reshape(-1) - 1, 0), dtype=torch.long, device=device)
    grid_valid = torch.as_tensor(GRID.reshape(-1) >= 0, dtype=torch.float32, device=device).reshape(1, 1, 10, 11)

    def inputs(index):
        return masked_grid_inputs(full_bands, index, channel_mask, grid_ids, grid_valid, band_ids)

    model = base(128).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != 30534:
        raise RuntimeError(f"Expected 30534 MDFNet parameters, got {parameter_count}")
    run_dir = args.output_dir / args.dataset / args.condition / f"seed{args.seed}" / f"fold{args.fold}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "exp041_fig2_1s_lto", "dataset": args.dataset, "condition": args.condition,
                "fold": args.fold, "seed": args.seed, "epochs": args.epochs, "batch_size": 128,
                "decision_window_samples": 128, "sample_rate_hz": 128, "optimizer": "Adam",
                "learning_rate": 0.001, "weight_decay": 0.01, "parameter_count": parameter_count,
                "resident_before_training": True, "determinism": "cudnn deterministic=False, benchmark=True; not bitwise reproduction",
                "data_sha256": sha256_file(data_path),
                "regions_json_sha256": sha256_file(regions_path),
                "base_sha256": sha256_file(ROOT / "OURmodels" / "Base.py"),
                "original_trial_order": orders,
                "trial_splits": {name: values.tolist() for name, values in
                                 zip(("train", "validation", "test"), (train_trials, valid_trials, test_trials))},
                "condition_mask": {"channels_1based": channels, "bands": [BAND_NAMES[i] for i in band_ids]}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    optimizer, loss_fn = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01), nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(rows["train"])), batch_size=128, shuffle=True, generator=generator)

    def evaluate(name):
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for (indices,) in DataLoader(TensorDataset(torch.from_numpy(rows[name])), batch_size=128):
                batch = indices.to(device)
                predictions.append(model(*inputs(batch)).argmax(1).cpu().numpy())
                cpu_batch = batch.cpu().numpy()
                targets.append(labels[cpu_batch[:, 0], cpu_batch[:, 1], cpu_batch[:, 2] * 128].astype(np.int64))
        return np.concatenate(predictions), np.concatenate(targets)

    history, best, best_epoch = [], -float("inf"), -1
    checkpoint = run_dir / "best.ckpt"
    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
        for (indices,) in train_loader:
            batch = indices.to(device)
            cpu = batch.cpu().numpy()
            target = torch.as_tensor(labels[cpu[:, 0], cpu[:, 1], cpu[:, 2] * 128], device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs(batch))
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(1) == target).sum())
            total += len(target)
        prediction, target = evaluate("validation")
        accuracy = float(np.mean(prediction == target))
        history.append({"epoch": epoch, "train_accuracy": correct / total, "validation_accuracy": accuracy})
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            torch.save(model.state_dict(), checkpoint)
        print(f"epoch={epoch} train_accuracy={correct / total:.6f} validation_accuracy={accuracy:.6f}", flush=True)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    prediction, target = evaluate("test")
    metrics = {"status": "complete", "dataset": args.dataset, "condition": args.condition,
               "fold": args.fold, "seed": args.seed, "best_epoch": best_epoch,
               "test_accuracy": float(np.mean(prediction == target)), "test_window_count": len(target),
               "parameter_count": parameter_count}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(run_dir / "predictions.npz", prediction=prediction, truth=target)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


if __name__ == "__main__":
    main()
