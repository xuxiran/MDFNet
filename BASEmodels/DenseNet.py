"""ASAD-DenseNet baseline adapter.

The feature extractor follows the MIT-licensed ASAD-DenseNet implementation
by Xiran Xu, with temporal pooling adjusted for this repository's windows.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import config as cfg


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm3d(in_channels + index * growth_rate),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels + index * growth_rate,
                    growth_rate,
                    kernel_size=(1, 3, 3),
                    padding=(0, 1, 1),
                ),
            )
            for index in range(num_convs)
        )
        self.out_channels = in_channels + num_convs * growth_rate

    def forward(self, features):
        for layer in self.layers:
            features = torch.cat((features, layer(features)), dim=1)
        return features


class DenseNet3D(nn.Module):
    def __init__(self, decision_window):
        super().__init__()
        pool_size = 5 if decision_window == 64 else 7
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        blocks = []
        channels = 64
        for index in range(4):
            block = DenseBlock(4, channels, 32)
            blocks.append(block)
            channels = block.out_channels
            if index < 3:
                blocks.append(
                    nn.Sequential(
                        nn.BatchNorm3d(channels),
                        nn.ReLU(),
                        nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1)),
                        nn.AvgPool3d(
                            kernel_size=(pool_size, 2, 2),
                            stride=(3, 1, 1),
                        ),
                    )
                )
                channels //= 2
        self.features = nn.Sequential(*blocks, nn.BatchNorm3d(channels), nn.ReLU())
        self.classifier = nn.Linear(channels, 2)

    def forward(self, eeg):
        if eeg.ndim != 4 or eeg.shape[-2:] != (10, 11):
            raise ValueError("DenseNet expects EEG shaped [batch, time, 10, 11]")
        features = self.stem(eeg.unsqueeze(1))
        features = self.features(features)
        features = F.avg_pool3d(features, kernel_size=features.shape[2:]).flatten(1)
        return self.classifier(features)


class DenseNet(nn.Module):
    def __init__(self, device, decision_window):
        super().__init__()
        self.model = DenseNet3D(decision_window).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def train(self, train_loader, device, epoch, N_epoch):
        del epoch, N_epoch
        self.model.train()
        correct = 0
        count = 0
        for features, labels, _ in tqdm.tqdm(train_loader, position=0, leave=True):
            logits = self.model(features[2].to(device))
            labels = labels.to(device)
            loss = F.cross_entropy(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            count += labels.shape[0]
            correct += (logits.argmax(dim=1) == labels).sum().item()
        if count:
            print(f"DenseNet Train Accuracy: {correct / count}")

    def test(self, test_loader, device):
        self.model.eval()
        predictions = []
        labels_all = []
        with torch.no_grad():
            for features, labels, _ in tqdm.tqdm(test_loader, position=0, leave=True):
                logits = self.model(features[2].to(device))
                predictions.append(logits.argmax(dim=1).cpu().numpy())
                labels_all.append(labels.cpu().numpy())
        if not predictions:
            return np.array([]), np.array([])
        return np.concatenate(predictions), np.concatenate(labels_all)
