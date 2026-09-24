#!/usr/bin/env python3
"""Resident-memory 0.5-second KUL/DTU leave-one-subject-out runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from AADdataset import AADdataset
from baseline_csp import fit_baseline_csp
from get_model import MODEL_NAMES, get_model
from loso_utils import aggregate_metrics, reorder_trials, source_trial_split
from main import resident_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["KUL", "DTU"], required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, default="MDFNet")
    parser.add_argument("--data-dir", type=Path, default=Path(cfg.process_data_dir))
    parser.add_argument("--test-subject", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_loso"))
    parser.add_argument("--decision-window", type=int, choices=[64], default=64)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate completed subject metrics and exit")
    parser.add_argument("--allow-local", action="store_true", help="Permit local execution for smoke checks")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_trials(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["EEG"]).transpose().copy()
        labels = np.asarray(handle["ENV"]).transpose().copy()
    if data.ndim != 4 or data.shape[-1] != 64 or labels.shape != data.shape[:3]:
        raise ValueError("Expected EEG [subjects,trials,time,64] and ENV [subjects,trials,time]")
    if not np.isfinite(data).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("EEG must be finite and labels must be binary")
    if not np.all(labels == labels[..., :1]):
        raise ValueError("Each trial must have one constant label")
    return data, labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict | None:
    if args.aggregate:
        summary = aggregate_metrics(args.output_dir, args.dataset, args.model, args.seed)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary
    if not args.allow_local and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("LOSO training must be launched through Slurm; use --allow-local only for local checks")
    if args.seed != 2025 or args.epochs != 50 or args.batch_size != 128:
        raise ValueError("Frozen LOSO settings are seed 2025, 50 epochs, and batch size 128")
    if args.test_subject is None:
        raise ValueError("--test-subject is required unless --aggregate is specified")
    set_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 7))
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    data_path = args.data_dir / f"{args.dataset}_1D.mat"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    data, labels, orders = reorder_trials(*load_trials(data_path))
    subjects, trials, samples, channels = data.shape
    expected_trials = 8 if args.dataset == "KUL" else 32
    expected_subjects = 16 if args.dataset == "KUL" else 21
    if subjects != expected_subjects:
        raise ValueError(f"{args.dataset} LOSO requires {expected_subjects} subjects; found {subjects}")
    if trials != expected_trials:
        raise ValueError(f"{args.dataset} LOSO requires {expected_trials} trials per subject; found {trials}")
    if not 0 <= args.test_subject < subjects:
        raise ValueError("test-subject is outside the data")
    train_ids, valid_ids = source_trial_split(args.dataset)
    source_subjects = [index for index in range(subjects) if index != args.test_subject]

    def flatten(subject_ids: list[int], trial_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        return data[np.ix_(subject_ids, trial_ids)].reshape(-1, samples, channels), labels[np.ix_(subject_ids, trial_ids)].reshape(-1, samples)

    arrays = {
        "train": flatten(source_subjects, train_ids),
        "valid": flatten(source_subjects, valid_ids),
        "test": flatten([args.test_subject], list(range(trials))),
    }
    csp_pipe = fit_baseline_csp(args.model, *arrays["train"])
    datasets = {name: resident_dataset(AADdataset(x, y, 128, args.decision_window, args.dataset, csp_pipe), device)
                for name, (x, y) in arrays.items()}
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train", num_workers=0)
               for name, dataset in datasets.items()}
    model, _ = get_model(args.model, args.decision_window, len(arrays["train"][0]), device)
    run_dir = args.output_dir / args.dataset / args.model / "dw64" / f"seed{args.seed}" / f"subject{args.test_subject}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset, "model": args.model, "test_subject": args.test_subject,
        "window_samples": 64, "sample_rate_hz": 128, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "optimizer": "Adam", "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay, "source_train_trial_ids": train_ids,
        "source_valid_trial_ids": valid_ids, "target_trial_count": trials,
        "resident_before_training": True, "data_sha256": sha256(data_path), "original_trial_order": orders,
        "csp": ({"components": 64, "transform_into": "csp_space", "fit_trials": len(arrays["train"][0]),
                 "fit_scope": "train"} if csp_pipe is not None else None),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    best, history = -float("inf"), []
    checkpoint = run_dir / "best.ckpt"
    for epoch in range(args.epochs):
        model.train(loaders["train"], device, epoch, args.epochs)
        prediction, truth = model.test(loaders["valid"], device)
        accuracy = float(np.mean(prediction == truth))
        history.append({"epoch": epoch, "validation_accuracy": accuracy})
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), checkpoint)
        print(f"epoch={epoch + 1}/{args.epochs} validation_accuracy={accuracy:.6f}", flush=True)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    prediction, truth = model.test(loaders["test"], device)
    metrics = {"accuracy": float(np.mean(prediction == truth)), "n_windows": int(len(truth)),
               "dataset": args.dataset, "model": args.model, "test_subject": args.test_subject}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(run_dir / "predictions.npz", prediction=prediction, truth=truth)
    print(json.dumps(metrics), flush=True)
    return metrics


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
