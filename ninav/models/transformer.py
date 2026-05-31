"""Transformer-encoder regressor for fixed-length IMU windows.

Regresses a heading-agnostic, gravity-aligned 2D velocity / displacement from a
short IMU window (RoNIN / TLIO / IONet style), to be integrated downstream by
:mod:`ninav.reconstruct`.

Architecture (per the model contract -- inputs are channel-first ``(B, C, L)``):

    1. transpose ``(B, C, L) -> (B, L, C)`` so time is the sequence axis;
    2. ``Linear(in_channels -> d_model)`` input projection;
    3. ADD fixed sinusoidal positional encoding of shape ``(L, d_model)`` -- this
       is mandatory: a self-attention encoder is permutation-equivariant over the
       sequence axis, so without positional information shuffling the time axis
       would leave the (mean-pooled) output unchanged. The positional encoding
       breaks that symmetry (verified by a permutation test in ``__main__``);
    4. ``TransformerEncoder`` of ``num_layers`` pre-norm GELU encoder layers
       (``d_model=128, nhead=8, dim_feedforward=256, dropout=0.1``);
    5. MEAN POOL over the time axis to a single ``(B, d_model)`` summary;
    6. ``Linear(d_model -> out_dim)`` regression head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) sinusoidal positional encoding, ``(L, d_model)``.

    Uses the original Transformer formulation (Vaswani et al., 2017):

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    The table is registered as a (persistent-but-not-trained) buffer so it moves
    with ``.to(device)`` and is part of ``state_dict`` for reproducibility.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # (ceil(d_model/2),)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model: the cos slice may be one element shorter.
        pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape ``(B, L, d_model)``."""
        seq_len = x.shape[1]
        if seq_len > self.pe.shape[0]:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.pe.shape[0]}"
            )
        return x + self.pe[:seq_len].unsqueeze(0)


class TransformerRegressor(nn.Module):
    """Transformer-encoder regressor over fixed-length IMU windows.

    Parameters
    ----------
    in_channels:
        Number of input channels (default 6: 3 world-frame accel + 3 gyro).
    out_dim:
        Regression output dimension (default 2: 2D horizontal velocity).
    d_model:
        Transformer embedding width.
    nhead:
        Number of attention heads. Must divide ``d_model`` and be ``>= 4``.
    num_layers:
        Number of stacked encoder layers.
    dim_feedforward:
        Width of the per-layer feed-forward network.
    dropout:
        Dropout probability inside the encoder layers.
    max_len:
        Maximum supported window length for the positional table.

    Forward
    -------
    ``forward(x)`` takes ``x`` of shape ``(B, in_channels, L)`` (channel-first)
    and returns ``(B, out_dim)``.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
        assert nhead >= 4, f"nhead must be >= 4, got {nhead}"

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"expected 3D input (B, in_channels, L), got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {x.shape[1]}"
            )

        # (B, C, L) -> (B, L, C) so time is the sequence axis.
        x = x.transpose(1, 2)
        # Project channels to the model width, then inject position.
        x = self.input_proj(x)          # (B, L, d_model)
        x = self.pos_encoder(x)         # (B, L, d_model)
        x = self.encoder(x)             # (B, L, d_model)
        # Mean pool over the time axis.
        x = x.mean(dim=1)               # (B, d_model)
        return self.head(x)             # (B, out_dim)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = TransformerRegressor(in_channels=6, out_dim=2)
    model.eval()  # disable dropout for a deterministic permutation test

    x = torch.randn(4, 6, 200)
    with torch.no_grad():
        y = model(x)
    print(f"forward: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (4, 2), f"expected (4, 2), got {tuple(y.shape)}"

    # Permutation test: shuffling the time axis must change the output, proving
    # the positional encoding (not just attention) is in effect.
    perm = torch.randperm(x.shape[2])
    x_shuf = x[:, :, perm]
    with torch.no_grad():
        y_shuf = model(x_shuf)
    max_abs_diff = (y - y_shuf).abs().max().item()
    print(f"permutation test: max abs diff = {max_abs_diff:.6e}")
    assert max_abs_diff > 1e-5, (
        "time-shuffled input produced (near) identical output -- "
        "positional encoding is not working"
    )
    print("OK")
