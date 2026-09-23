"""Synthetic metrics-only tests for Fig. 2 aggregation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from aggregate_fig2 import CONDITIONS, DATASETS, collect_results, plot_summary, write_summary


class Fig2AggregateTests(unittest.TestCase):
    def make_runs(self, root):
        for dataset in DATASETS:
            for condition in CONDITIONS:
                for fold, accuracy in enumerate((0.5, 0.6, 0.7, 0.8)):
                    run_dir = root / dataset / condition / "seed2025" / f"fold{fold}"
                    run_dir.mkdir(parents=True)
                    (run_dir / "metrics.json").write_text(json.dumps({
                        "status": "complete", "dataset": dataset, "condition": condition,
                        "fold": fold, "seed": 2025, "test_accuracy": accuracy,
                    }), encoding="utf-8")
                    (run_dir / "manifest.json").write_text(json.dumps({
                        "protocol": "exp041_fig2_1s_lto", "dataset": dataset,
                        "condition": condition, "fold": fold, "seed": 2025,
                        "epochs": 50, "batch_size": 128, "decision_window_samples": 128,
                        "sample_rate_hz": 128, "parameter_count": 30534,
                    }), encoding="utf-8")

    def test_mean_sample_std_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            self.make_runs(root)
            rows = collect_results(root)
            self.assertEqual(len(rows), 22)
            expected = next(row for row in rows if row["dataset"] == "KUL" and row["condition"] == "All")
            self.assertAlmostEqual(expected["mean_accuracy_percent"], 65.0)
            self.assertAlmostEqual(expected["std_accuracy_percent"], (12.909944487 * 100 / 100), places=6)
            output = Path(directory) / "out"
            self.assertTrue(write_summary(rows, output).is_file())
            try:
                self.assertTrue(plot_summary(rows, output).is_file())
            except ImportError:
                self.skipTest("matplotlib is unavailable")

    def test_incomplete_runs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            self.make_runs(root)
            (root / "DTU" / "All" / "seed2025" / "fold0" / "metrics.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                collect_results(root)

    def test_invalid_accuracy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            self.make_runs(root)
            path = root / "KUL" / "All" / "seed2025" / "fold0" / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["test_accuracy"] = float("nan")
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite fraction"):
                collect_results(root)

    def test_smoke_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "inputs"
            self.make_runs(root)
            path = root / "KUL" / "All" / "seed2025" / "fold0" / "manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["epochs"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-paper-protocol manifest"):
                collect_results(root)


if __name__ == "__main__":
    unittest.main()
