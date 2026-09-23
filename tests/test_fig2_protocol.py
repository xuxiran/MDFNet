"""Synthetic-only checks for the frozen Fig. 2 masks and LTO protocol."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from run_fig2_experiment import (  # noqa: E402
    BAND_NAMES, CONDITIONS, GRID, condition_masks, load_regions, reorder_trials,
    split_trials, masked_grid_inputs,
)


class Fig2ProtocolTests(unittest.TestCase):
    def test_frozen_mask_sizes_and_condition_semantics(self):
        regions = load_regions(ROOT / "configs" / "regions.json")
        self.assertEqual({key: len(value) for key, value in regions.items()},
                         {"F": 26, "C": 21, "P": 23, "T": 6, "O": 9})
        self.assertTrue(all(condition_masks(f"region_{key}", regions)[1] == list(range(5))
                            for key in regions))
        for index, band in enumerate(BAND_NAMES):
            channels, active_bands = condition_masks(f"band_{band}", regions)
            self.assertEqual(channels, list(range(1, 65)))
            self.assertEqual(active_bands, [index])
        self.assertEqual(len(CONDITIONS), 11)

    def test_overlapping_regional_boundaries_are_frozen(self):
        regions = load_regions(ROOT / "configs" / "regions.json")
        self.assertTrue({9, 10, 11}.issubset(set(regions["F"]) & set(regions["C"])))
        self.assertTrue({17, 18, 19, 32}.issubset(set(regions["C"]) & set(regions["P"])))
        self.assertTrue({25, 26, 30, 62, 63}.issubset(set(regions["P"]) & set(regions["O"])))
        self.assertEqual(GRID.shape, (10, 11))
        self.assertEqual(set(GRID[GRID > 0]), set(range(1, 65)))

    def test_alternating_order_and_quarter_splits(self):
        labels = np.repeat(np.array([0, 1] * 4)[None, :, None], 3, axis=2)
        data = np.arange(8, dtype=np.int64)[None, :, None, None]
        ordered_data, ordered_labels, orders = reorder_trials(data, labels)
        self.assertEqual(orders, [[0, 1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(ordered_data.shape, data.shape)
        self.assertEqual(ordered_labels[0, :, 0].tolist(), [0, 1] * 4)
        for fold in range(4):
            train, valid, test = split_trials(8, fold)
            self.assertEqual((len(train), len(valid), len(test)), (4, 2, 2))
            self.assertEqual(set(train) | set(valid) | set(test), set(range(8)))
            self.assertFalse(set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test))
            self.assertEqual(valid[0], (test[-1] + 1) % 8)

    def test_invalid_region_config_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "regions.json"
            path.write_text(json.dumps({"regions": {"F": [1]}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_regions(path)

    def test_synthetic_masked_grid_shape_and_zeroing(self):
        bands = torch.ones((1, 4, 5, 128, 64), dtype=torch.float32)
        indices = torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.long)
        channel_mask = torch.zeros(64)
        channel_mask[[0, 32]] = 1
        grid_ids = torch.as_tensor(np.maximum(GRID.reshape(-1) - 1, 0), dtype=torch.long)
        grid_valid = torch.as_tensor(GRID.reshape(-1) >= 0, dtype=torch.float32).reshape(1, 1, 10, 11)
        features = masked_grid_inputs(bands, indices, channel_mask, grid_ids, grid_valid, [2])
        self.assertEqual(len(features), 5)
        self.assertTrue(all(feature.shape == (2, 128, 10, 11) for feature in features))
        self.assertTrue(torch.count_nonzero(features[0]) == 0)
        self.assertTrue(torch.count_nonzero(features[2]) > 0)
        self.assertTrue(torch.all(features[2][:, :, GRID < 0] == 0))


if __name__ == "__main__":
    unittest.main()
