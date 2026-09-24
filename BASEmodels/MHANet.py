import torch
import torch.nn as nn
import config as cfg

from BASEmodels.architectures.MHANet import MHANet as MHAArchitecture
from BASEmodels.training import LoaderAdapter


class MHANet(LoaderAdapter, nn.Module):
    def __init__(self, device, decision_window):
        nn.Module.__init__(self)
        self.model = MHAArchitecture(decision_window=decision_window, num_classes=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _step_input(self, batch, device):
        # The current comparison uses the raw five-band feature 0 (B,T,64).
        return batch[0].to(device)

    def forward(self, value):
        logits = self.model(value)
        return logits, logits
