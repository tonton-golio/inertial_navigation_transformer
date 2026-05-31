"""Regression losses for velocity / displacement targets."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the batch (the RoNIN default)."""
    return F.mse_loss(pred, target)


def gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor,
                      log_std: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian negative log-likelihood (TLIO).

    The network regresses ``log_std`` (NOT std or variance) for numerical
    stability. ``L = mean[ 0.5 * sum(2*log_std) + 0.5 * sum((d - mu)^2 / var) ]``
    with ``var = exp(2*log_std)``.

    Warm-start with :func:`mse_loss` for ~10 epochs; NLL from scratch diverges.
    """
    if pred.shape != target.shape or pred.shape != log_std.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)}, target {tuple(target.shape)}, "
            f"log_std {tuple(log_std.shape)}")
    inv_var = torch.exp(-2.0 * log_std)
    per_elem = log_std + 0.5 * (pred - target) ** 2 * inv_var
    return per_elem.sum(dim=-1).mean()
