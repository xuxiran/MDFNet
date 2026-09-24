"""Focused checks for train-only CSP baseline preprocessing."""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from baseline_csp import fit_baseline_csp


class BaselineCspTests(unittest.TestCase):
    def setUp(self):
        self.eeg = np.arange(3 * 16 * 64, dtype=np.float64).reshape(3, 16, 64)
        self.labels = np.repeat(np.array([[0], [1], [0]]), 16, axis=1)

    def test_fit_uses_trial_major_mne_shape_and_first_sample_labels(self):
        calls = {}

        class FakeCSP:
            def __init__(self, **kwargs):
                calls["kwargs"] = kwargs

            def fit(self, x, y):
                calls["x"] = x
                calls["y"] = y
                return self

        fake_mne = types.ModuleType("mne")
        fake_decoding = types.ModuleType("mne.decoding")
        fake_decoding.CSP = FakeCSP
        fake_mne.decoding = fake_decoding
        with patch.dict(sys.modules, {"mne": fake_mne, "mne.decoding": fake_decoding}):
            transformer = fit_baseline_csp("DARNet", self.eeg, self.labels)

        self.assertIsInstance(transformer, FakeCSP)
        self.assertEqual(calls["kwargs"], {
            "n_components": 64,
            "transform_into": "csp_space",
            "cov_est": "concat",
            "norm_trace": True,
            "log": None,
            "reg": None,
        })
        np.testing.assert_array_equal(calls["x"], self.eeg.transpose(0, 2, 1))
        np.testing.assert_array_equal(calls["y"], [0, 1, 0])

    def test_fit_receives_only_explicit_training_arrays(self):
        captured = {}

        class FakeCSP:
            def __init__(self, **kwargs):
                pass

            def fit(self, x, y):
                captured["trials"] = len(x)
                captured["labels"] = y.copy()
                return self

        fake_mne = types.ModuleType("mne")
        fake_decoding = types.ModuleType("mne.decoding")
        fake_decoding.CSP = FakeCSP
        fake_mne.decoding = fake_decoding
        with patch.dict(sys.modules, {"mne": fake_mne, "mne.decoding": fake_decoding}):
            fit_baseline_csp("DBPNet", self.eeg[:2], self.labels[:2])

        self.assertEqual(captured["trials"], 2)
        np.testing.assert_array_equal(captured["labels"], [0, 1])

    def test_non_csp_model_does_not_import_mne_or_fit(self):
        with patch.dict(sys.modules, {"mne": None, "mne.decoding": None}):
            self.assertIsNone(fit_baseline_csp("MDFNet", self.eeg, self.labels))

    def test_rejects_invalid_labels_and_shape(self):
        inconsistent = self.labels.copy()
        inconsistent[0, -1] = 1
        with self.assertRaisesRegex(ValueError, "constant class label"):
            fit_baseline_csp("DARNet", self.eeg, inconsistent)
        with self.assertRaisesRegex(ValueError, "64 input channels"):
            fit_baseline_csp("DBPNet", self.eeg[:, :, :32], self.labels)


if __name__ == "__main__":
    unittest.main()
