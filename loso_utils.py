"""Utilities for the frozen MDFNet leave-one-subject-out protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def reorder_trials(data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    orders = []
    reordered_data = np.empty_like(data)
    reordered_labels = np.empty_like(labels)
    for subject in range(data.shape[0]):
        trial_labels = labels[subject, :, 0].astype(int)
        zeros = np.flatnonzero(trial_labels == 0).tolist()
        ones = np.flatnonzero(trial_labels == 1).tolist()
        if len(zeros) != len(ones):
            raise ValueError("LOSO requires balanced classes per subject")
        order = [trial for pair in zip(zeros, ones) for trial in pair]
        reordered_data[subject] = data[subject, order]
        reordered_labels[subject] = labels[subject, order]
        orders.append(order)
    return reordered_data, reordered_labels, orders


def source_trial_split(dataset: str) -> tuple[list[int], list[int]]:
    if dataset == "KUL":
        return list(range(6)), list(range(6, 8))
    if dataset == "DTU":
        return list(range(24)), list(range(24, 32))
    raise ValueError(f"Unsupported dataset: {dataset}")


def aggregate_metrics(metrics_root: Path, dataset: str, model: str, seed: int = 2025) -> dict:
    subject_dir = metrics_root / dataset / model / "dw64" / f"seed{seed}"
    rows = []
    paths = sorted(subject_dir.glob("subject*/metrics.json"))
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
    expected_subjects = 16 if dataset == "KUL" else 21 if dataset == "DTU" else None
    if expected_subjects is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not rows:
        raise FileNotFoundError(f"No LOSO subject metrics found under {subject_dir}")
    subject_ids = [int(row["test_subject"]) for row in rows]
    for path, subject_id in zip(paths, subject_ids):
        if path.parent.name != f"subject{subject_id}":
            raise ValueError(f"Metrics subject ID does not match directory: {path}")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Duplicate test_subject IDs in LOSO metrics")
    if any(subject_id < 0 or subject_id >= expected_subjects for subject_id in subject_ids):
        raise ValueError(f"Subject IDs must be in range 0..{expected_subjects - 1}")
    if set(subject_ids) != set(range(expected_subjects)):
        missing = sorted(set(range(expected_subjects)) - set(subject_ids))
        raise ValueError(f"Incomplete LOSO metrics; missing subjects: {missing}")
    for row in rows:
        if "dataset" in row and row["dataset"] != dataset:
            raise ValueError(f"Metrics dataset mismatch: expected {dataset}, found {row['dataset']}")
        if "model" in row and row["model"] != model:
            raise ValueError(f"Metrics model mismatch: expected {model}, found {row['model']}")
    accuracies = np.asarray([row["accuracy"] for row in rows], dtype=np.float64)
    if not np.isfinite(accuracies).all() or ((accuracies < 0) | (accuracies > 1)).any():
        raise ValueError("Subject accuracies must be finite values in [0, 1]")
    result = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "subjects": subject_ids,
        "n_subjects": int(len(rows)),
        "mean_accuracy": float(accuracies.mean()),
        "sample_sd_accuracy": float(accuracies.std(ddof=1)) if len(rows) > 1 else 0.0,
    }
    output_path = subject_dir / "aggregate_metrics.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
