import torch
import torch.nn as nn
import config as cfg

from BASEmodels.training import LoaderAdapter


class ListenNet(LoaderAdapter, nn.Module):
    def __init__(self, device, decision_window):
        nn.Module.__init__(self)
        try:
            from BASEmodels.architectures.ListenNet import ListenNet as ListenArchitecture
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("BASEmodels.architectures"):
                raise ImportError("ListenNet architecture is not bundled with this release") from error
            raise
        self.model = ListenArchitecture(chans=64, samples=decision_window, num_classes=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _step_input(self, batch, device):
        # Public loader feature 0 is B,T,64; ListenNet consumes B,1,64,T.
        return batch[0].to(device).permute(0, 2, 1).unsqueeze(1)

    def forward(self, value):
        return self.model(value)
