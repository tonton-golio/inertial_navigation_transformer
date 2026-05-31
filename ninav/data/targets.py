"""Motion targets -- all RELATIVE, per-window, and IDENTICAL for train and test.

Fixes:
    * the old code rebased the target per window for train (``y -= y[:, :1, :]``)
      but NOT for test, then applied an ad-hoc global shift at plot time (bug C9);
    * the per-window target was reconstructed by a different formula than the
      ground truth (bug C8).

All targets are defined over the ``stride`` interval that follows each window
start, so that ``ninav.reconstruct.integrate_velocity`` (with ``dt = stride/fs``)
telescopes exactly back to the true positions at the window-start times.
By default we predict 2D horizontal motion (RoNIN evaluates in 2D).
"""
from __future__ import annotations

import numpy as np


def start_offsets(positions: np.ndarray, starts: np.ndarray, dims: int = 2) -> np.ndarray:
    """True position at each window's start sample -- the reconstruction anchor."""
    positions = np.asarray(positions, dtype=np.float64)
    return positions[starts, :dims].copy()


def velocity_target(positions: np.ndarray, starts: np.ndarray, stride: int,
                    dt: float, dims: int = 2) -> np.ndarray:
    """Average velocity over the ``stride`` interval after each window start.

    ``v[j] = (pos[s_j + stride] - pos[s_j]) / (stride * dt)`` in the nav frame.
    Integrating these with ``integrate_velocity(v, dt=stride*dt)`` reconstructs
    the true positions at the window-start times (perfect model -> 0 ATE).
    """
    positions = np.asarray(positions, dtype=np.float64)
    disp = positions[starts + stride, :dims] - positions[starts, :dims]
    return disp / (stride * dt)


def displacement_target(positions: np.ndarray, starts: np.ndarray, stride: int,
                        dims: int = 2) -> np.ndarray:
    """Displacement over the ``stride`` interval: ``pos[s+stride] - pos[s]``."""
    positions = np.asarray(positions, dtype=np.float64)
    return positions[starts + stride, :dims] - positions[starts, :dims]


def polar_target(positions: np.ndarray, starts: np.ndarray, stride: int,
                 dims: int = 2) -> np.ndarray:
    """IONet polar target ``(delta_l, delta_psi)`` for 2D motion.

    ``delta_l`` is the displacement magnitude over the stride; ``delta_psi`` is
    the change of heading (atan2 of the displacement) relative to the previous
    window's heading. Heading-invariant by construction.
    """
    if dims != 2:
        raise ValueError("polar_target is defined for 2D motion only")
    positions = np.asarray(positions, dtype=np.float64)
    disp = positions[starts + stride, :2] - positions[starts, :2]
    dl = np.linalg.norm(disp, axis=-1)
    psi = np.arctan2(disp[:, 1], disp[:, 0])
    dpsi = np.empty_like(psi)
    dpsi[0] = 0.0
    dpsi[1:] = np.arctan2(np.sin(psi[1:] - psi[:-1]), np.cos(psi[1:] - psi[:-1]))
    return np.stack([dl, dpsi], axis=-1)


def make_target(kind: str, positions: np.ndarray, starts: np.ndarray, stride: int,
                dt: float, dims: int = 2) -> np.ndarray:
    """Dispatch by target kind: ``'velocity' | 'displacement' | 'polar'``."""
    if kind == "velocity":
        return velocity_target(positions, starts, stride, dt, dims)
    if kind == "displacement":
        return displacement_target(positions, starts, stride, dims)
    if kind == "polar":
        return polar_target(positions, starts, stride, dims)
    raise ValueError(f"unknown target kind {kind!r}")
