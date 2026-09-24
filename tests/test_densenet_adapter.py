import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from get_model import get_model


class DenseNetAdapterTests(unittest.TestCase):
    def test_forward_and_training_checkpoint_for_supported_windows(self):
        for window in (64, 128, 256):
            with self.subTest(window=window):
                device = torch.device("cpu")
                adapter, model_type = get_model("DenseNet", window, 1, device)
                self.assertEqual(model_type, "BASEmodels")
                torch.manual_seed(7)
                torch.cuda.manual_seed_all(7)
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                eeg = torch.randn(2, window, 10, 11)
                logits = adapter.model(eeg)
                self.assertEqual(tuple(logits.shape), (2, 2))

                before = next(adapter.model.parameters()).detach().clone()
                adapter.optimizer.zero_grad()
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                adapter.optimizer.step()
                self.assertFalse(torch.equal(before, next(adapter.model.parameters()).detach()))

                features = [torch.randn(2, window, 10, 11) for _ in range(11)]
                batches = [(features, torch.tensor([0, 1]), torch.tensor([0, 1]))]
                adapter.train(batches, device, 0, 1)
                predictions, truth = adapter.test(batches, device)
                self.assertEqual(predictions.shape, (2,))
                self.assertEqual(truth.tolist(), [0, 1])

                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "densenet.ckpt"
                    torch.save(adapter.state_dict(), checkpoint)
                    restored, _ = get_model("DenseNet", window, 1, device)
                    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                    restored.model.eval()
                    adapter.model.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(adapter.model(eeg), restored.model(eeg))


if __name__ == "__main__":
    unittest.main()
