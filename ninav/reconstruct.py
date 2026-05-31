"""Trajectory reconstruction -- ONE operator, used for predictions AND ground truth.

The old project reconstructed predictions by ``cumsum`` of per-window displacements
(implicitly assuming ``stride == displacement span``, so a *perfect* model drifted
linearly whenever ``overlap != 1`` -- bug C7) and reconstructed the ground truth by
a *different* formula (bug C8). Here a single function integrates a velocity stream,
and it is applied identically to estimate and ground truth, so a perfect prediction
reconstructs to zero error for any stride.
"""
from __future__ import annotations

import numpy as np


def integrate_velocity(velocity: np.ndarray, dt: float,
                       p0: np.ndarray | None = None) -> np.ndarray:
    """Integrate a per-step velocity stream into positions (RoNIN-style).

    Parameters
    ----------
    velocity : (T, D)   velocity samples (m/s) in the navigation frame.
    dt : float          time between samples (s). For windowed velocity at a
                        fixed stride, ``dt = stride / fs``.
    p0 : (D,) or None    initial position (defaults to zeros).

    Returns
    -------
    (T, D) positions, with ``pos[0] == p0``.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2:
        raise ValueError(f"velocity must be (T, D), got {velocity.shape}")
    D = velocity.shape[1]
    p0 = np.zeros(D) if p0 is None else np.asarray(p0, dtype=np.float64).reshape(D)
    # pos[k] = p0 + sum_{j<k} v[j]*dt  (left Riemann sum; pos[0]=p0)
    incr = velocity * dt
    pos = np.empty_like(velocity)
    pos[0] = p0
    if velocity.shape[0] > 1:
        pos[1:] = p0 + np.cumsum(incr[:-1], axis=0)
    return pos


def reconstruct_from_displacement(displacement: np.ndarray,
                                  start_offsets: np.ndarray) -> np.ndarray:
    """Place per-window displacements at their true window-start offsets.

    This is the displacement-target analogue of :func:`integrate_velocity` and
    avoids the overlap-dependent ``cumsum`` drift (C7): each window's predicted
    displacement is added to the known start position of that window.

    Parameters
    ----------
    displacement : (N, D)    predicted displacement of each window.
    start_offsets : (N, D)   true position at each window's start sample.
    """
    displacement = np.asarray(displacement, dtype=np.float64)
    start_offsets = np.asarray(start_offsets, dtype=np.float64)
    if displacement.shape != start_offsets.shape:
        raise ValueError(
            f"shape mismatch: displacement {displacement.shape} vs "
            f"start_offsets {start_offsets.shape}")
    return displacement + start_offsets


def reconstruct_trajectory(pred, *, mode: str = "velocity", dt: float | None = None,
                           start_offsets=None, p0=None) -> np.ndarray:
    """Single entry point for trajectory reconstruction.

    ``mode='velocity'`` integrates ``pred`` (needs ``dt``); ``mode='displacement'``
    places ``pred`` at ``start_offsets``. Use the SAME call for est and gt.
    """
    if mode == "velocity":
        if dt is None:
            raise ValueError("mode='velocity' requires dt")
        return integrate_velocity(pred, dt, p0=p0)
    if mode == "displacement":
        if start_offsets is None:
            raise ValueError("mode='displacement' requires start_offsets")
        return reconstruct_from_displacement(pred, start_offsets)
    raise ValueError(f"unknown mode {mode!r}")
