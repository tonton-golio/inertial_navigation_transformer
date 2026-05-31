"""RoNIN-style 1D ResNet-18 for inertial navigation (channel-first).

Reimplements the ResNet-18 backbone used by RoNIN for regressing horizontal
velocity / displacement from a window of world-frame IMU features.

Model contract (shared by all position models):
    * ``__init__(self, in_channels=6, out_dim=2, **arch_kwargs)``
    * ``forward(x)`` takes ``x`` of shape ``(B, in_channels, L)`` (channel-first:
      3 world-frame accel + 3 world-frame gyro) and returns ``(B, out_dim)``.
    * Default ``out_dim=2`` (2D horizontal velocity). Runs on CPU.

Architecture:
    stem  : Conv1d(in_channels, 64, k=7, s=2, p=3) + BN + ReLU + MaxPool1d(3, 2, 1)
    stages: 4 x BasicBlock1D, planes [64, 128, 256, 512], blocks [2, 2, 2, 2],
            stride 2 at the start of stages 2-4
    head  : AdaptiveAvgPool1d(1) -> Flatten -> Linear(512, out_dim)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv3x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """3x1 convolution with padding (the 1D analogue of ResNet's 3x3 conv)."""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution (used for the downsampling skip projection)."""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock1D(nn.Module):
    """ResNet BasicBlock adapted to 1D: two 3x1 convs + BN + ReLU + skip."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x1(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """1D ResNet-18 backbone + linear regression head (channel-first).

    Args:
        in_channels: number of input channels (default 6: 3 accel + 3 gyro).
        out_dim: regression output dimension (default 2: 2D velocity).
        block_planes: channel width per stage.
        block_counts: number of BasicBlocks per stage.
        zero_init_residual: zero-init the last BN gamma in each block so each
            residual branch starts as identity (RoNIN/torchvision practice).
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 2,
        block_planes: tuple[int, ...] = (64, 128, 256, 512),
        block_counts: tuple[int, ...] = (2, 2, 2, 2),
        zero_init_residual: bool = True,
        **arch_kwargs,
    ) -> None:
        super().__init__()
        if len(block_planes) != len(block_counts):
            raise ValueError(
                "block_planes and block_counts must have the same length, "
                f"got {len(block_planes)} and {len(block_counts)}"
            )

        self.in_channels = in_channels
        self.out_dim = out_dim

        # Stem.
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages. Stage 1 keeps stride 1; stages 2-4 downsample (stride 2).
        self.inplanes = 64
        stages = []
        for i, (planes, count) in enumerate(zip(block_planes, block_counts)):
            stride = 1 if i == 0 else 2
            stages.append(self._make_stage(planes, count, stride=stride))
        self.layers = nn.Sequential(*stages)

        # Head.
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(block_planes[-1] * BasicBlock1D.expansion, out_dim)

        self._init_weights(zero_init_residual)

    def _make_stage(self, planes: int, count: int, stride: int) -> nn.Sequential:
        downsample = None
        out_planes = planes * BasicBlock1D.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, out_planes, stride=stride),
                nn.BatchNorm1d(out_planes),
            )

        blocks: list[nn.Module] = [
            BasicBlock1D(self.inplanes, planes, stride=stride, downsample=downsample)
        ]
        self.inplanes = out_planes
        for _ in range(1, count):
            blocks.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*blocks)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, L) channel-first.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)

        x = self.avgpool(x)          # (B, 512, 1)
        x = self.flatten(x)          # (B, 512)
        x = self.fc(x)               # (B, out_dim)
        return x


__all__ = ["ResNet1D", "BasicBlock1D", "conv3x1", "conv1x1"]
