"""CPU-only split, aggregation, and synthetic MDFNet forward checks."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from get_model import get_model
from loso_utils import aggregate_metrics, reorder_trials, source_trial_split


def main():
    data = np.arange(2 * 8 * 3 * 64).reshape(2, 8, 3, 64)
    labels = np.empty((2, 8, 3), dtype=np.int64)
    labels[:, :4] = 0
    labels[:, 4:] = 1
    reordered, reordered_labels, orders = reorder_trials(data, labels)
    assert orders[0] == [0, 4, 1, 5, 2, 6, 3, 7]
    assert reordered_labels[0, :, 0].tolist() == [0, 1] * 4
    assert reordered[0, 0, 0, 0] == data[0, 0, 0, 0]
    assert source_trial_split("KUL") == (list(range(6)), list(range(6, 8)))
    assert source_trial_split("DTU") == (list(range(24)), list(range(24, 32)))

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        subject_dir = root / "KUL" / "MDFNet" / "dw64" / "seed2025"
        for subject, accuracy in ((0, 0.75), (1, 0.5)):
            target = subject_dir / f"subject{subject}"
            target.mkdir(parents=True)
            (target / "metrics.json").write_text(json.dumps({"accuracy": accuracy, "test_subject": subject}))
        try:
            aggregate_metrics(root, "KUL", "MDFNet")
        except ValueError as error:
            assert "Incomplete LOSO metrics" in str(error)
        else:
            raise AssertionError("Incomplete subject coverage must fail")
        accuracies = np.linspace(0.4, 0.9, 16)
        for subject, accuracy in enumerate(accuracies):
            target = subject_dir / f"subject{subject}"
            target.mkdir(exist_ok=True)
            (target / "metrics.json").write_text(json.dumps({
                "accuracy": float(accuracy), "test_subject": subject, "dataset": "KUL", "model": "MDFNet",
            }))
        summary = aggregate_metrics(root, "KUL", "MDFNet")
        assert summary["n_subjects"] == 16
        assert np.isclose(summary["mean_accuracy"], accuracies.mean())
        assert np.isclose(summary["sample_sd_accuracy"], accuracies.std(ddof=1))

    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    model, _ = get_model("MDFNet", 64, 1, torch.device("cpu"))
    model.model.eval()
    features = [torch.randn(1, 64, 10, 11) for _ in range(5)]
    with torch.inference_mode():
        output = model.model(*features)
    assert output.shape == (1, 2) and torch.isfinite(output).all()
    print("LOSO split, aggregation, and synthetic MDFNet checks passed")


if __name__ == "__main__":
    main()
