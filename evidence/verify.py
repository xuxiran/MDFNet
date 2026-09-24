"""Check the published historical-result extracts without private run directories."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


EVIDENCE_DIR = Path(__file__).resolve().parent
PAPER_MODELS = {
    "CNN", "XANet", "DenseNet", "DARNet", "DBPNet", "ListenNet", "MHANet", "MDFNet"
}
PAPER_LTO_MATCHES = {
    ("DTU", "ListenNet", 0.5): (50.4, 1.1),
    ("DTU", "ListenNet", 1.0): (50.1, 1.5),
    ("DTU", "ListenNet", 2.0): (49.5, 0.7),
    ("DTU", "MHANet", 0.5): (52.9, 5.9),
    ("DTU", "MHANet", 1.0): (54.7, 6.3),
    ("DTU", "MHANet", 2.0): (56.8, 8.4),
    ("DTU", "MDFNet", 1.0): (69.3, 2.6),
    ("KUL", "ListenNet", 0.5): (77.1, 2.4),
    ("KUL", "ListenNet", 1.0): (77.7, 2.4),
    ("KUL", "ListenNet", 2.0): (78.7, 1.9),
    ("KUL", "MHANet", 0.5): (76.3, 2.5),
    ("KUL", "MHANet", 1.0): (76.7, 3.4),
    ("KUL", "MHANet", 2.0): (77.6, 2.9),
    ("KUL", "MDFNet", 1.0): (80.0, 3.7),
}
PAPER_LOSO_MATCHES = {
    ("DTU", "CNN"): (56.2, 6.2),
    ("DTU", "XANet"): (50.0, 0.0),
    ("DTU", "DenseNet"): (50.0, 0.0),
    ("DTU", "DBPNet"): (58.3, 7.2),
    ("DTU", "DARNet"): (58.0, 8.7),
    ("DTU", "ListenNet"): (51.1, 2.9),
    ("DTU", "MHANet"): (50.7, 1.7),
    ("DTU", "MDFNet"): (60.7, 10.1),
    ("KUL", "CNN"): (52.4, 3.2),
    ("KUL", "XANet"): (68.2, 17.3),
    ("KUL", "DenseNet"): (63.9, 14.4),
    ("KUL", "DBPNet"): (63.6, 10.6),
    ("KUL", "DARNet"): (64.3, 15.8),
    ("KUL", "ListenNet"): (65.2, 16.5),
    ("KUL", "MHANet"): (66.7, 14.4),
    ("KUL", "MDFNet"): (69.0, 17.9),
}


def read_csv(filename):
    with (EVIDENCE_DIR / filename).open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def fold_key(row):
    return row["dataset"], row["model"], float(row["window_s"]), int(row["fold"])


def check_lto():
    fold_rows = read_csv("lto_matched_folds.csv")
    metadata_rows = [
        json.loads(line)
        for line in (EVIDENCE_DIR / "lto_run_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    history_rows = read_csv("lto_validation_history.csv")
    assert len(fold_rows) == len(metadata_rows) == 56
    assert len(history_rows) == 56 * 50

    metadata_by_fold = {fold_key(row): row for row in metadata_rows}
    assert len(metadata_by_fold) == 56
    history_by_fold = defaultdict(list)
    for row in history_rows:
        history_by_fold[fold_key(row)].append(row)

    cells = defaultdict(list)
    seen_folds = set()
    for row in fold_rows:
        key = fold_key(row)
        assert key not in seen_folds
        seen_folds.add(key)
        assert int(row["seed"]) == 2025
        assert row["validation_status"] == "PASS"
        metadata = metadata_by_fold[key]
        assert metadata["source_run"] == row["source_run"]
        assert math.isclose(float(row["accuracy_percent"]), metadata["accuracy_percent"], abs_tol=1e-8)
        history = sorted(history_by_fold[key], key=lambda item: int(item["epoch"]))
        assert [int(item["epoch"]) for item in history] == list(range(50))
        assert all(int(item["seed"]) == 2025 for item in history)
        assert all(item["source_run"] == row["source_run"] for item in history)
        best_validation = max(float(item["validation_accuracy"]) for item in history)
        assert math.isclose(best_validation, metadata["best_validation_accuracy"], abs_tol=1e-8)
        assert metadata["epochs"] == 50
        train_trials = set(metadata["train_trials_reordered"])
        validation_trials = set(metadata["validation_trials_reordered"])
        test_trials = set(metadata["test_trials_reordered"])
        assert train_trials.isdisjoint(validation_trials)
        assert train_trials.isdisjoint(test_trials)
        assert validation_trials.isdisjoint(test_trials)
        cells[key[:3]].append((key[3], float(row["accuracy_percent"])))

    assert set(cells) == set(PAPER_LTO_MATCHES)
    for cell, folds in cells.items():
        assert sorted(fold for fold, _ in folds) == [0, 1, 2, 3]
        values = [accuracy for _, accuracy in sorted(folds)]
        displayed = round(statistics.mean(values), 1), round(statistics.stdev(values), 1)
        assert displayed == PAPER_LTO_MATCHES[cell], (cell, displayed)
    return len(cells), len(fold_rows)


def check_loso():
    subject_rows = read_csv("loso_0p5s_subjects.csv")
    summary_rows = read_csv("loso_0p5s_summary.csv")
    metadata_rows = [
        json.loads(line)
        for line in (EVIDENCE_DIR / "loso_run_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    history_rows = read_csv("loso_validation_history.csv")
    assert len(subject_rows) == 296
    assert len(summary_rows) == 16
    assert len(metadata_rows) == 296
    assert len(history_rows) == 296 * 50
    metadata_by_subject = {
        (row["dataset"], row["model"], row["test_subject_0based"]): row
        for row in metadata_rows
    }
    assert len(metadata_by_subject) == 296
    history_by_subject = defaultdict(list)
    for row in history_rows:
        key = row["dataset"], row["model"], int(row["test_subject_0based"])
        history_by_subject[key].append(row)
    subjects_by_group = defaultdict(dict)
    sources_by_group = defaultdict(set)
    for row in subject_rows:
        assert row["model"] in PAPER_MODELS
        assert row["validation_status"] == "PASS"
        assert not row["source_file"].startswith(("/", "\\"))
        assert int(row["seed"]) == 2025
        assert float(row["window_s"]) == 0.5
        group = row["dataset"], row["model"]
        subject = int(row["test_subject_0based"])
        assert subject not in subjects_by_group[group]
        key = row["dataset"], row["model"], subject
        metadata = metadata_by_subject[key]
        assert metadata["source_run"] == row["source"]
        assert math.isclose(metadata["accuracy_percent"], float(row["test_accuracy_percent"]), abs_tol=1e-8)
        assert metadata["epochs"] == 50
        assert metadata["seed"] == 2025
        assert metadata["test_subjects"] == [subject]
        assert subject not in metadata["train_subjects"]
        assert subject not in metadata["validation_subjects"]
        history = sorted(history_by_subject[key], key=lambda item: int(item["epoch"]))
        assert [int(item["epoch"]) for item in history] == list(range(50))
        assert all(item["source_run"] == row["source"] for item in history)
        best_validation = max(float(item["validation_accuracy"]) for item in history)
        assert math.isclose(best_validation, metadata["best_validation_accuracy"], abs_tol=1e-8)
        subjects_by_group[group][subject] = float(row["test_accuracy_percent"])
        sources_by_group[group].add(row["source"])

    expected_groups = {(dataset, model) for dataset in ("DTU", "KUL") for model in PAPER_MODELS}
    assert set(subjects_by_group) == expected_groups
    summary_by_group = {(row["dataset"], row["model"]): row for row in summary_rows}
    assert set(summary_by_group) == expected_groups
    for group, subject_values in subjects_by_group.items():
        count = 21 if group[0] == "DTU" else 16
        assert set(subject_values) == set(range(count))
        summary = summary_by_group[group]
        values = list(subject_values.values())
        assert int(summary["subjects"]) == count
        assert int(summary["seed"]) == 2025
        assert float(summary["window_s"]) == 0.5
        assert sources_by_group[group] == {summary["source"]}
        assert math.isclose(statistics.mean(values), float(summary["mean_accuracy_percent"]), abs_tol=1e-9)
        assert math.isclose(statistics.stdev(values), float(summary["subject_sample_sd_percent"]), abs_tol=1e-9)
        displayed = round(statistics.mean(values), 1), round(statistics.stdev(values), 1)
        assert displayed == PAPER_LOSO_MATCHES[group], (group, displayed)
    return len(summary_rows), len(subject_rows)


def main():
    lto_cells, lto_folds = check_lto()
    loso_cells, loso_subjects = check_loso()
    print(f"LTO: {lto_cells} matched cells, {lto_folds} folds, 50 epochs each")
    print(f"LOSO: {loso_cells} cells, {loso_subjects} subject results")


if __name__ == "__main__":
    main()
