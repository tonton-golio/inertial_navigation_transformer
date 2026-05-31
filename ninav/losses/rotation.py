"""Rotation/quaternion losses, double-cover aware and NaN-safe.

Fixes the old orientation-loss bugs:
    * phi2 collapsed the whole batch to one scalar and chose a single global
      ``q``/``-q`` sign (bug C10) -- here reductions are per-sample (dim=-1);
    * ``arccos(|cos_sim|)`` had an infinite gradient at the optimum (bug H6) --
      the recommended training losses (``1 - |<q1,q2>|`` and the chordal loss)
      are smooth there; the arccos form is provided only as a *metric* with a
      clamp away from +/-1.

All functions assume scalar-last unit-ish quaternions of shape ``(..., 4)`` and
normalize defensively. ``q`` and ``-q`` denote the same rotation, so every loss
is invariant to a per-sample sign flip.
"""
from __future__ import annotations

import torch


def _normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``1 - |<pred, target>|`` averaged over the batch (smooth, recommended)."""
    pred = _normalize(pred)
    target = _normalize(target)
    dot = (pred * target).sum(dim=-1).abs()
    return (1.0 - dot).mean()


def quat_chordal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample double-cover squared chordal distance ``min(|p-t|^2, |p+t|^2)``."""
    pred = _normalize(pred)
    target = _normalize(target)
    d_minus = ((pred - target) ** 2).sum(dim=-1)
    d_plus = ((pred + target) ** 2).sum(dim=-1)
    return torch.minimum(d_minus, d_plus).mean()


def quat_geodesic_loss(pred: torch.Tensor, target: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
    """Mean geodesic angle (radians) as a loss, with a clamp to keep grads finite.

    Prefer :func:`quat_cosine_loss` / :func:`quat_chordal_loss` for training;
    this is here for completeness and clamps ``|cos|`` to ``1 - eps``.
    """
    pred = _normalize(pred)
    target = _normalize(target)
    dot = (pred * target).sum(dim=-1).abs().clamp(max=1.0 - eps)
    return (2.0 * torch.arccos(dot)).mean()


@torch.no_grad()
def geodesic_angle_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample geodesic angle in DEGREES (a metric, not for backprop)."""
    pred = _normalize(pred.double())
    target = _normalize(target.double())
    dot = (pred * target).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return torch.rad2deg(2.0 * torch.arccos(dot))
