"""XANet baseline adapter for the public MDFNet comparison package."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import config as cfg


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[-1])
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(query_size, num_hiddens * 10, bias=False)
        self.key = nn.Linear(key_size, num_hiddens * 10, bias=False)
        self.value = nn.Linear(value_size, num_hiddens, bias=False)
        self.attention = DotProductAttention(dropout)

    def _split_heads(self, tensor):
        batch, length, width = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, width // self.num_heads).permute(
            0, 2, 1, 3
        ).reshape(batch * self.num_heads, length, width // self.num_heads)

    def forward(self, queries, keys, values):
        query = self._split_heads(self.query(queries))
        key = self._split_heads(self.key(keys))
        value = self._split_heads(self.value(values))
        output = self.attention(query, key, value)
        batch_heads, length, width = output.shape
        output = output.reshape(-1, self.num_heads, length, width).permute(0, 2, 1, 3)
        return output.reshape(output.shape[0], output.shape[1], -1)


class XANetModel(nn.Module):
    def __init__(self, decision_window):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(decision_window)
        self.conv2d_left = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv2d_right = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.conv1d_left = nn.Conv1d(60, 5, kernel_size=3, padding=1)
        self.conv1d_right = nn.Conv1d(60, 5, kernel_size=3, padding=1)
        self.linear2 = nn.Linear(8, 64)
        self.attention = MultiHeadAttention(5, 5, 5, 5, 1, dropout=0.3)
        self.fc1 = nn.Linear(decision_window // 4 * 10, 2)

    def forward(self, eeg):
        if eeg.ndim != 4 or eeg.shape[-2:] != (10, 11):
            raise ValueError("XANet expects EEG shaped [batch, time, 10, 11]")
        left = eeg[..., :6].reshape(eeg.shape[0], eeg.shape[1], -1)
        right = eeg[..., 5:].reshape(eeg.shape[0], eeg.shape[1], -1)
        left = self.batchnorm(left).unsqueeze(1)
        right = self.batchnorm(right).unsqueeze(1)
        left = self.pooling(self.relu(self.conv2d_left(left))).squeeze(1).permute(0, 2, 1)
        right = self.pooling(self.relu(self.conv2d_right(right))).squeeze(1).permute(0, 2, 1)
        left = self.conv1d_left(left).permute(0, 2, 1).unsqueeze(1)
        right = self.conv1d_right(right).permute(0, 2, 1).unsqueeze(1)
        left = self.pooling(left).squeeze(1)
        right = self.pooling(right).squeeze(1)
        left_out = self.attention(right, left, left)
        right_out = self.attention(left, right, right)
        return self.fc1(torch.cat((left_out, right_out), dim=2).flatten(1))


class XANet(nn.Module):
    def __init__(self, device, decision_window):
        super().__init__()
        self.model = XANetModel(decision_window).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        correct = 0
        count = 0
        loss_total = 0.0
        for features, labels, _ in tqdm.tqdm(train_loader, position=0, leave=True):
            logits = self.model(features[2].to(device))
            labels = labels.to(device)
            loss = F.cross_entropy(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            count += labels.shape[0]
            correct += (logits.argmax(dim=1) == labels).sum().item()
            loss_total += loss.item() * labels.shape[0]
        accuracy = correct / count if count else 0.0
        mean_loss = loss_total / count if count else 0.0
        print(
            f"epoch={epoch + 1}/{N_epoch} XANet train_loss={mean_loss:.6f} "
            f"train_accuracy={accuracy:.6f}",
            flush=True,
        )

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
