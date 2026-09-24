"""PyTorch reproduction of the temporal CNN baseline for AAD."""

import numpy as np
import torch
import torch.nn as nn
import tqdm

import config as cfg


class CNNFeatures(nn.Module):
    def __init__(self, decision_window):
        super().__init__()
        if decision_window < 1:
            raise ValueError("CNN requires a positive decision window")
        self.decision_window = decision_window
        self.conv_layer = nn.Conv2d(
            in_channels=1,
            out_channels=5,
            kernel_size=(17, 64),
            padding=(8, 0),
        )
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(decision_window, 1))
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=5, out_features=2)

    def forward(self, eeg):
        if eeg.ndim != 3 or eeg.shape[1:] != (self.decision_window, 64):
            raise ValueError(
                "CNN expects EEG shaped "
                f"[batch, {self.decision_window}, 64], got {tuple(eeg.shape)}"
            )
        features = self.conv_layer(eeg.unsqueeze(1))
        features = self.relu(features)
        features = self.avg_pool(features).flatten(start_dim=1)
        features = self.activation(self.fc1(features))
        return self.fc2(features)


class CNN(nn.Module):
    def __init__(self, device, decision_window):
        super().__init__()
        self.model = CNNFeatures(decision_window).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        correct = 0
        count = 0
        total_loss = 0.0
        for features, labels, _ in tqdm.tqdm(train_loader, position=0, leave=True):
            logits = self.model(features[0].to(device))
            labels = labels.to(device)
            loss = nn.functional.cross_entropy(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_size = labels.shape[0]
            count += batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total_loss += loss.detach().item() * batch_size
        accuracy = correct / count if count else float("nan")
        mean_loss = total_loss / count if count else float("nan")
        print(
            f"CNN epoch={epoch + 1}/{N_epoch} "
            f"train_loss={mean_loss:.6f} train_accuracy={accuracy:.6f}",
            flush=True,
        )

    def test(self, test_loader, device):
        self.model.eval()
        predictions = []
        labels_all = []
        with torch.no_grad():
            for features, labels, _ in tqdm.tqdm(test_loader, position=0, leave=True):
                logits = self.model(features[0].to(device))
                predictions.append(logits.argmax(dim=1).cpu().numpy())
                labels_all.append(labels.cpu().numpy())
        if not predictions:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(predictions), np.concatenate(labels_all)
