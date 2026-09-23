#!/usr/bin/env python3
"""Validate and summarize complete four-fold Fig. 2 reruns."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

DATASETS = ("DTU", "KUL")
CONDITIONS = ("All", "region_F", "region_C", "region_P", "region_T", "region_O",
              "band_delta", "band_theta", "band_alpha", "band_beta", "band_gamma")
SEED = 2025
FOLDS = range(4)


def collect_results(input_dir):
    input_dir = Path(input_dir)
    values = {}
    for path in sorted(input_dir.glob("*/*/seed*/fold*/metrics.json")):
        relative = path.relative_to(input_dir).parts
        if len(relative) != 5:
            raise ValueError(f"Unexpected metrics path: {path}")
        dataset, condition, seed_dir, fold_dir, _ = relative
        try:
            seed, fold = int(seed_dir.removeprefix("seed")), int(fold_dir.removeprefix("fold"))
        except ValueError as error:
            raise ValueError(f"Invalid seed/fold directory in {path}") from error
        if dataset not in DATASETS or condition not in CONDITIONS or seed != SEED or fold not in FOLDS:
            raise ValueError(f"Unexpected run identity in {path}")
        key = (dataset, condition, fold)
        if key in values:
            raise ValueError(f"Duplicate run for {key}")
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read metrics JSON: {path}") from error
        expected = {"dataset": dataset, "condition": condition, "fold": fold, "seed": SEED}
        if metrics.get("status") != "complete" or any(metrics.get(name) != value for name, value in expected.items()):
            raise ValueError(f"Incomplete or mismatched metrics identity: {path}")
        manifest_path = path.with_name("manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Missing or invalid manifest: {manifest_path}") from error
        expected_manifest = {**expected, "protocol": "exp041_fig2_1s_lto", "epochs": 50,
                             "batch_size": 128, "decision_window_samples": 128,
                             "sample_rate_hz": 128, "parameter_count": 30534}
        if any(manifest.get(name) != value for name, value in expected_manifest.items()):
            raise ValueError(f"Mismatched or non-paper-protocol manifest: {manifest_path}")
        accuracy = metrics.get("test_accuracy")
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"test_accuracy must be a finite fraction in [0, 1]: {path}")
        values[key] = float(accuracy)

    expected_keys = {(dataset, condition, fold) for dataset in DATASETS for condition in CONDITIONS for fold in FOLDS}
    missing = sorted(expected_keys - values.keys())
    if missing:
        raise ValueError(f"Expected all 88 complete runs; missing {len(missing)}: {missing}")
    unexpected = sorted(values.keys() - expected_keys)
    if unexpected:
        raise ValueError(f"Unexpected runs: {unexpected}")

    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            scores = np.asarray([values[(dataset, condition, fold)] for fold in FOLDS])
            rows.append({"dataset": dataset, "condition": condition,
                         "mean_accuracy_percent": float(scores.mean() * 100),
                         "std_accuracy_percent": float(scores.std(ddof=1) * 100), "n_folds": 4})
    return rows


def write_summary(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fig2_summary.csv"
    fields = ("dataset", "condition", "mean_accuracy_percent", "std_accuracy_percent", "n_folds")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_summary(rows, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x_positions = np.arange(len(CONDITIONS))
    colors = ["#777777"] + ["#4c78a8"] * 5 + ["#f58518"] * 5
    for axis, dataset in zip(axes, ("DTU", "KUL")):
        selected = {row["condition"]: row for row in rows if row["dataset"] == dataset}
        means = [selected[name]["mean_accuracy_percent"] for name in CONDITIONS]
        deviations = [selected[name]["std_accuracy_percent"] for name in CONDITIONS]
        axis.bar(x_positions, means, color=colors, alpha=0.72, width=0.72)
        axis.errorbar(x_positions, means, yerr=deviations, fmt="o", color="black",
                      capsize=4, markersize=4, linewidth=1)
        axis.set_ylabel(f"{dataset} accuracy (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[-1].set_xticks(x_positions, CONDITIONS, rotation=30, ha="right")
    axes[0].set_title("Fig. 2 conditions — regenerated from new four-fold runs")
    figure.tight_layout()
    path = output_dir / "fig2_regenerated.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs_fig2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_results(args.input_dir)
    print(f"Wrote {write_summary(rows, args.output_dir)}")
    print(f"Wrote {plot_summary(rows, args.output_dir)}")


if __name__ == "__main__":
    main()
