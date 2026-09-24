"""Shared loader adapter for the public baseline wrappers."""

import numpy as np
import torch
import tqdm


def logits_from_output(output):
    return output[-1] if isinstance(output, tuple) else output


class LoaderAdapter:
    """Provide the historical train(loader, device, epoch, N_epoch)/test API."""

    def _step_input(self, batch):
        raise NotImplementedError

    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        correct = total = 0
        for X, label, _ in tqdm.tqdm(train_loader, position=0, leave=True):
            target = label.to(device)
            logits = logits_from_output(self.model(self._step_input(X, device)))
            loss = torch.nn.functional.cross_entropy(logits, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += target.numel()
            correct += (logits.argmax(1) == target).sum().item()
        print(f"{self.__class__.__name__} Train Accuracy: {correct / total if total else 0.0}")

    def test(self, test_loader, device):
        self.model.eval()
        result, gt = [], []
        with torch.no_grad():
            for X, label, _ in tqdm.tqdm(test_loader, position=0, leave=True):
                logits = logits_from_output(self.model(self._step_input(X, device)))
                result.append(logits.argmax(1).cpu().numpy())
                gt.append(label.cpu().numpy())
        return np.concatenate(result) if result else np.array([]), np.concatenate(gt) if gt else np.array([])
