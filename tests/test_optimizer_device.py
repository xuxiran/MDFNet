import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from get_model import MODEL_NAMES, get_model


class OptimizerDeviceTests(unittest.TestCase):
    def test_optimizer_tracks_inner_model_parameters_on_cpu(self):
        for model_name in MODEL_NAMES:
            with self.subTest(model_name=model_name):
                model, _ = get_model(model_name, 4, 1, torch.device("cpu"))
                optimizer_parameters = [
                    parameter
                    for group in model.optimizer.param_groups
                    for parameter in group["params"]
                ]
                inner_parameters = list(model.model.parameters())

                self.assertEqual(len(optimizer_parameters), len(inner_parameters))
                self.assertEqual(
                    {id(parameter) for parameter in optimizer_parameters},
                    {id(parameter) for parameter in inner_parameters},
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_optimizer_updates_inner_model_parameter(self):
        device = torch.device("cuda")
        model, _ = get_model("MDFNet", 4, 1, device)
        model.model.train()
        parameter = next(model.model.parameters())
        before = parameter.detach().clone()
        features = [torch.randn(1, 4, 10, 11, device=device) for _ in range(5)]

        model.optimizer.zero_grad()
        loss = model.model(*features).sum()
        loss.backward()
        model.optimizer.step()

        self.assertFalse(torch.equal(parameter.detach(), before))


if __name__ == "__main__":
    unittest.main()
