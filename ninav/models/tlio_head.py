"""TLIO-style network: ResNet1D trunk with mean + log-std heads.

This is the probabilistic counterpart to the plain RoNIN-style regressor. A
1D ResNet trunk consumes a window of world-frame IMU features ``(B, 6, L)``
(3 accel + 3 gyro channels) and produces a pooled feature vector. Two
independent linear heads then map that vector to:

* ``mean``    -- the predicted displacement / velocity ``(B, out_dim)``
* ``log_std`` -- the LOG standard deviation ``(B, out_dim)`` of a diagonal
  Gaussian over the same target.

The network regresses LOG-std (never std or variance) for numerical
stability. The output pairs directly with
:func:`ninav.losses.regression.gaussian_nll_loss`, which expects
``(pred, target, log_std)`` and internally forms ``var = exp(2*log_std)``.

The package keeps its own small trunk here rather than importing
``ninav.models.resnet1d`` (which is not present) so this module is
self-contained and depends only on the public model contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _BasicBlock1d(nn.Module):
    """A pre-activation-free 1D residual block (two conv-bn-relu layers).

    Mirrors the RoNIN/TLIO ResNet basic block: two ``kernel_size=3`` convs
    with a projection shortcut when the channel count or stride changes.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class TLIONet(nn.Module):
    """ResNet1D trunk with separate mean and log-std heads (TLIO-style).

    Parameters
    ----------
    in_channels:
        Number of input channels. Defaults to 6 (3 world-frame accel +
        3 world-frame gyro), matching the model contract.
    out_dim:
        Output dimensionality of BOTH heads. Defaults to 2 (2D horizontal
        displacement / velocity).
    base_channels:
        Channel width of the first residual stage. Each subsequent stage
        doubles the channel count.
    layers:
        Number of basic blocks per stage. The number of stages is
        ``len(layers)``.

    forward
    -------
    ``forward(x)`` takes ``x`` of shape ``(B, in_channels, L)`` and returns a
    tuple ``(mean, log_std)`` each of shape ``(B, out_dim)``. The second
    element is LOG standard deviation, ready for
    :func:`ninav.losses.regression.gaussian_nll_loss`.
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 2,
                 base_channels: int = 64,
                 layers: tuple[int, ...] = (2, 2, 2, 2)):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Stem: widen channels and shrink the temporal axis once.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages. First stage keeps the stem stride; later stages
        # downsample by 2 and double the channel count.
        stages: list[nn.Module] = []
        cur = base_channels
        for stage_idx, n_blocks in enumerate(layers):
            out_ch = base_channels * (2 ** stage_idx)
            stride = 1 if stage_idx == 0 else 2
            blocks = [_BasicBlock1d(cur, out_ch, stride=stride)]
            for _ in range(1, n_blocks):
                blocks.append(_BasicBlock1d(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*blocks))
            cur = out_ch
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(1)
        feat_dim = cur

        # Two independent linear heads on the pooled features.
        self.mean_head = nn.Linear(feat_dim, out_dim)
        self.log_std_head = nn.Linear(feat_dim, out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        # Start log_std near 0 (std ~ 1) so early NLL / MSE warm-up is stable.
        nn.init.normal_(self.log_std_head.weight, std=1e-3)
        nn.init.zeros_(self.log_std_head.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature vector ``(B, feat_dim)`` for ``x``."""
        h = self.stem(x)
        h = self.stages(h)
        h = self.pool(h)            # (B, feat_dim, 1)
        return torch.flatten(h, 1)  # (B, feat_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        return mean, log_std
