"""Internal, provenance-locked ListenNet architecture reproduction.

This module is for the exp036 fair comparison only. It follows the architecture
of Fan et al. (IJCAI 2025, DOI 10.24963/ijcai.2025/461), audited at upstream
commit 286b8ac4bddc7e6baee870940457809131249737. The upstream repository had no
root license at audit time, so this file must not be redistributed in a public
artifact without permission. The only semantic-preserving implementation fix
is device-neutral creation of ``channel_weight2``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Align(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        if channels_in > channels_out:
            self.projection = nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.channels_in > self.channels_out:
            return self.projection(value)
        if self.channels_in < self.channels_out:
            return F.pad(value, [0, 0, 0, 0, 0, self.channels_out - self.channels_in, 0, 0])
        return value


class DilatedInception(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dilation: int = 1) -> None:
        super().__init__()
        kernels = (1, 2, 3, 5)
        if channels_out % len(kernels):
            raise ValueError("channels_out must be divisible by four")
        branch_channels = channels_out // len(kernels)
        self.branches = nn.ModuleList(
            nn.Conv2d(channels_in, branch_channels, (1, kernel), dilation=(1, dilation))
            for kernel in kernels
        )
        # Retained for exact upstream parameter accounting, although the
        # published forward does not apply this layer.
        self.norm = nn.BatchNorm2d(channels_out)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branches = [branch(value) for branch in self.branches]
        width = branches[-1].shape[-1]
        return torch.cat([branch[..., -width:] for branch in branches], dim=1)


class CrossNestedAttention(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups
        group_channels = channels // groups
        self.softmax = nn.Softmax(-1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.height_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.width_pool = nn.AdaptiveAvgPool2d((1, None))
        self.norm = nn.GroupNorm(group_channels, group_channels)
        self.weight_projection = nn.Conv1d(1, 1, kernel_size=1, stride=17)
        self.coordinate_projection = nn.Conv2d(group_channels, group_channels, 1)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch, channels, height_first, width = first.shape
        height_second = second.shape[2]
        group_channels = channels // self.groups
        first_grouped = first.reshape(batch * self.groups, group_channels, height_first, width)
        second_grouped = second.reshape(batch * self.groups, group_channels, height_second, width)

        def coordinate_gate(value: torch.Tensor, height: int) -> torch.Tensor:
            height_value = self.height_pool(value)
            width_value = self.width_pool(value).permute(0, 1, 3, 2)
            joint = self.coordinate_projection(torch.cat([height_value, width_value], dim=2))
            gated_height, gated_width = torch.split(joint, [height, width], dim=2)
            return self.norm(value * gated_height.sigmoid() * gated_width.permute(0, 1, 3, 2).sigmoid())

        first_gated = coordinate_gate(first_grouped, height_first)
        second_gated = coordinate_gate(second_grouped, height_second)
        first_query = self.softmax(self.global_pool(first_gated).reshape(batch * self.groups, -1, 1).permute(0, 2, 1))
        second_query = self.softmax(self.global_pool(second_gated).reshape(batch * self.groups, -1, 1).permute(0, 2, 1))
        first_value = first_gated.reshape(batch * self.groups, group_channels, -1)
        second_value = second_gated.reshape(batch * self.groups, group_channels, -1)
        combined = torch.cat(
            [torch.matmul(first_query, second_value), torch.matmul(second_query, first_value)],
            dim=2,
        )
        weights = self.weight_projection(combined).reshape(batch * self.groups, 1, height_second, width)
        return (second_grouped * weights.sigmoid()).reshape(batch, channels, height_second, width)


class ListenNet(nn.Module):
    def __init__(
        self,
        chans: int = 64,
        samples: int = 1000,
        num_classes: int = 2,
        kernel: int = 8,
        depth: int = 16,
        average_pool: int = 10,
    ) -> None:
        super().__init__()
        self.channel_weight2 = nn.Parameter(torch.randn(depth, depth).float())
        nn.init.xavier_uniform_(self.channel_weight2.data)
        self.align = Align(chans, depth)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, depth, (1, 1), bias=False),
            nn.BatchNorm2d(depth),
            nn.Conv2d(depth, depth, (1, kernel), groups=depth, bias=False),
            nn.BatchNorm2d(depth),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(depth, depth, (1, 1), bias=False),
            nn.BatchNorm2d(depth),
            nn.Conv2d(depth, depth, (chans, 1), groups=depth, bias=False),
            nn.BatchNorm2d(depth),
            nn.GELU(),
        )
        self.multi_scale = DilatedInception(depth, depth, dilation=1)
        self.fusion_norm = nn.BatchNorm2d(depth)
        # Retained because both modules exist in the audited architecture and
        # contribute to its reported trainable-parameter count, even though
        # the upstream forward does not invoke them.
        self.skip_projection0 = nn.Conv2d(chans, depth, (1, 1))
        self.skip_projection1 = nn.Conv2d(depth, depth, (chans, 1), groups=depth, bias=False)
        self.skip = nn.Conv2d(depth, depth, (chans, 1), groups=depth, bias=True)
        self.dropout_probability = 0.65

        with torch.no_grad():
            probe = self.spatial(self.temporal(torch.ones(1, 1, chans, samples)))
        self.cross_nested_attention = CrossNestedAttention(probe.shape[1], probe.shape[1] // 2)
        pool_width = min(average_pool, probe.shape[-1])
        self.global_pool = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, pool_width)),
            nn.Dropout(p=0.65),
        )
        with torch.no_grad():
            feature_dimension = int(self.global_pool(probe).numel())
        self.classifier = nn.Linear(feature_dimension, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.temporal(value)
        spatial = self.spatial(temporal)
        multi_scale = self.multi_scale(temporal)
        skip = self.skip(F.dropout(multi_scale, self.dropout_probability, training=self.training))
        skip = F.interpolate(skip, size=(1, spatial.shape[-1]), mode="bilinear", align_corners=False)
        spatial_residual = self.fusion_norm(skip + spatial)
        aligned_temporal = self.align(temporal.permute(0, 2, 1, 3))
        aligned_temporal = torch.einsum("bdcw,sc->bdsw", aligned_temporal, self.channel_weight2)
        fused = self.cross_nested_attention(aligned_temporal, spatial_residual)
        feature = torch.flatten(self.global_pool(fused), 1)
        return feature, self.classifier(feature)
