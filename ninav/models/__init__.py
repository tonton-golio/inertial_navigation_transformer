"""Neural inertial-navigation position models and a name-based factory.

All position models follow a shared contract:

* ``__init__(self, in_channels=6, out_dim=2, **arch_kwargs)``
* ``forward(x)`` takes ``x`` of shape ``(B, in_channels, L)`` (channel-first:
  3 world-frame accel + 3 world-frame gyro) and returns ``(B, out_dim)``.
* Default ``out_dim=2`` (2D horizontal velocity / displacement). Runs on CPU.

``TLIONet`` is the probabilistic exception: its ``forward`` returns a tuple
``(mean, log_std)`` -- each of shape ``(B, out_dim)`` -- for use with
:func:`ninav.losses.regression.gaussian_nll_loss`.

Use :func:`build_model` to construct any of these by name.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from .lstm_tcn import LSTMRegressor, TCNRegressor
from .resnet1d import ResNet1D
from .tlio_head import TLIONet
from .transformer import TransformerRegressor

__all__ = [
    "ResNet1D",
    "TransformerRegressor",
    "LSTMRegressor",
    "TCNRegressor",
    "TLIONet",
    "MODEL_REGISTRY",
    "build_model",
]

# Map factory names to their model classes.
MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "resnet1d": ResNet1D,
    "transformer": TransformerRegressor,
    "lstm": LSTMRegressor,
    "tcn": TCNRegressor,
    "tlio": TLIONet,
}


def build_model(
    name: str,
    in_channels: int = 6,
    out_dim: int = 2,
    **kwargs,
) -> nn.Module:
    """Construct a position model by name.

    Parameters
    ----------
    name:
        One of ``{"resnet1d", "transformer", "lstm", "tcn", "tlio"}``.
    in_channels:
        Number of input channels (default 6: 3 world-frame accel + 3 gyro).
    out_dim:
        Regression output dimension (default 2: 2D horizontal velocity).
    **kwargs:
        Architecture-specific keyword arguments forwarded to the model
        constructor (e.g. ``d_model`` for the transformer, ``hidden`` for the
        LSTM, ``channels`` for the TCN).

    Returns
    -------
    torch.nn.Module
        An instantiated model following the package model contract.

    Raises
    ------
    KeyError
        If ``name`` is not a registered model.
    """
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"unknown model name {name!r}; valid names are: {valid}"
        ) from None
    return cls(in_channels=in_channels, out_dim=out_dim, **kwargs)
