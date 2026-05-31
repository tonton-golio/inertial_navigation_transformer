"""Recurrent and temporal-convolutional position regressors.

Both models follow the package model contract:

* ``__init__(self, in_channels=6, out_dim=2, **arch_kwargs)``
* ``forward(x)`` takes ``x`` of shape ``(B, in_channels, L)`` (channel-first;
  3 world-frame accel + 3 world-frame gyro) and returns ``(B, out_dim)``.
* Default ``out_dim=2`` (2D horizontal velocity / displacement).
* Runs on CPU.

``LSTMRegressor`` consumes the full window as a length-``L`` sequence (the
sequence dimension fed to the LSTM is ``L``, never 1) and regresses from the
last timestep's hidden state. ``TCNRegressor`` is a stack of dilated causal
residual blocks operating directly on ``(B, C, L)`` followed by global average
pooling and a linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LSTMRegressor", "TCNRegressor"]


class LSTMRegressor(nn.Module):
    """LSTM that regresses a target from the last timestep's hidden state.

    The input ``(B, C, L)`` is transposed to ``(B, L, C)`` so the recurrent
    sequence length is ``L`` (e.g. 200), not 1 -- the latter being the fatal
    bug in the legacy implementation this replaces.

    Parameters
    ----------
    in_channels:
        Number of input channels (default 6: 3 accel + 3 gyro).
    out_dim:
        Output dimension (default 2: horizontal velocity/displacement).
    hidden:
        LSTM hidden size.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers (only active when
        ``num_layers > 1``, per PyTorch semantics).
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 2,
        hidden: int = 100,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.hidden = hidden
        self.num_layers = num_layers

        # PyTorch only applies inter-layer dropout when num_layers > 1; passing
        # a non-zero dropout with a single layer emits a warning and is a no-op.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) channel-first -> (B, L, C) so the sequence dim is L.
        x = x.transpose(1, 2)
        # outputs: (B, L, hidden); we take the final timestep.
        outputs, _ = self.lstm(x)
        last = outputs[:, -1, :]
        return self.head(last)


class _TemporalBlock(nn.Module):
    """A dilated causal residual block (two causal conv layers + 1x1 skip)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_weight_norm: bool,
    ) -> None:
        super().__init__()
        # Left-pad amount for a causal convolution: no leakage from the future.
        self.pad = (kernel_size - 1) * dilation

        conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        if use_weight_norm:
            from torch.nn.utils.parametrizations import weight_norm

            conv1 = weight_norm(conv1)
            conv2 = weight_norm(conv2)
        self.conv1 = conv1
        self.conv2 = conv2

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        # 1x1 projection on the residual path when channel counts differ.
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.out_relu = nn.ReLU()

    def _causal_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        # Causal: pad only on the left so output[t] depends only on inputs <= t.
        return conv(F.pad(x, (self.pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self._causal_conv(x, self.conv1)))
        out = self.drop2(self.relu2(self._causal_conv(out, self.conv2)))
        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)


class TCNRegressor(nn.Module):
    """Temporal Convolutional Network regressor.

    A stack of dilated causal residual blocks (dilations 1, 2, 4, 8, ...)
    operating directly on ``(B, C, L)``, followed by adaptive global average
    pooling over time and a linear head.

    Parameters
    ----------
    in_channels:
        Number of input channels (default 6: 3 accel + 3 gyro).
    out_dim:
        Output dimension (default 2).
    channels:
        Output channels for each temporal block; its length is the network
        depth. Dilation doubles per block.
    kernel_size:
        Convolution kernel size.
    dropout:
        Dropout probability inside each block.
    use_weight_norm:
        Apply weight normalisation to the convolutions (default True).
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 2,
        channels=(64, 64, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        channels = tuple(channels)

        blocks = []
        prev = in_channels
        for i, ch in enumerate(channels):
            blocks.append(
                _TemporalBlock(
                    in_channels=prev,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                )
            )
            prev = ch
        self.network = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> conv stack -> (B, C_last, L)
        x = self.network(x)
        # global average pool over time -> (B, C_last, 1) -> (B, C_last)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
