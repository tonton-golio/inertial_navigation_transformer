"""Correct discrete-time attitude propagation from gyroscope readings.

This is the canonical replacement for the two contradictory, both-wrong
``Theta`` definitions in the old ``utils.py`` (bug C1):

    * the module-level one divided the ``sin`` term by ``|w|*dt/2`` instead of
      ``|w|`` -- wrong by ``2/dt = 400x`` at 200 Hz, and not norm-preserving;
    * the inner one (used to build the ``theta`` feature) baked ``0.5`` into
      ``Omega`` and so rotated at *half* the true rate.

The exact closed-form zeroth-order integrator for a constant body-rate ``w``
over ``dt`` is the rotation increment

    dq = [ (w/|w|) * sin(|w|*dt/2),  cos(|w|*dt/2) ]   (scalar-last, unit)

applied to the previous orientation. Equivalently, as a 4x4 matrix,

    Theta(w, dt) = cos(|w|*dt/2) * I_4  +  (sin(|w|*dt/2) / |w|) * Omega(w)

where ``Omega(w)`` is the full (un-halved) left-multiplication matrix of the
pure-vector quaternion ``[w, 0]``. ``Theta`` is then orthonormal (it is the
left-multiplication matrix of a *unit* quaternion), so it preserves quaternion
norm exactly -- which the original did not.
"""
from __future__ import annotations

import numpy as np

from .quaternion import quat_from_axis_angle, quat_multiply, quat_normalize

_EPS = 1e-12


def omega_matrix(w: np.ndarray) -> np.ndarray:
    """Full 4x4 left-multiplication matrix of the pure quaternion ``[w, 0]``.

    Scalar-last. ``omega_matrix(w) @ q == quat_multiply([wx,wy,wz,0], q)``.
    Note: NO factor of 1/2 here -- the half-angle lives in :func:`theta_matrix`.
    """
    w = np.asarray(w, dtype=np.float64).reshape(3)
    wx, wy, wz = w
    return np.array([
        [0.0, -wz, wy, wx],
        [wz, 0.0, -wx, wy],
        [-wy, wx, 0.0, wz],
        [-wx, -wy, -wz, 0.0],
    ])


def delta_quat(w: np.ndarray, dt: float) -> np.ndarray:
    """Unit rotation-increment quaternion for body-rate ``w`` over ``dt``."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(w) * dt
    return quat_from_axis_angle(w, angle)


def theta_matrix(w: np.ndarray, dt: float) -> np.ndarray:
    """Orthonormal 4x4 quaternion-update matrix (correct replacement for Theta).

    ``q_{k+1} = theta_matrix(w, dt) @ q_k``. Falls back to the identity for
    ``|w| -> 0`` (no NaN, unlike the original ``sin(0)/0``).
    """
    w = np.asarray(w, dtype=np.float64).reshape(3)
    wn = np.linalg.norm(w)
    if wn < _EPS:
        return np.eye(4)
    half = 0.5 * wn * dt
    return np.cos(half) * np.eye(4) + (np.sin(half) / wn) * omega_matrix(w)


def integrate_gyro(
    q0: np.ndarray,
    gyro: np.ndarray,
    dt: float,
    frame: str = "body",
) -> np.ndarray:
    """Integrate a gyro sequence into a sequence of orientations.

    Parameters
    ----------
    q0 : (4,) unit quaternion
        Initial orientation (scalar-last).
    gyro : (T, 3)
        Angular-velocity samples (rad/s).
    dt : float
        Sample period (s), e.g. 1/200.
    frame : {"body", "world"}
        ``"body"`` (default) integrates a strapdown gyro that measures angular
        velocity in the body frame: ``q_{k+1} = q_k (x) dq`` (right multiply).
        ``"world"`` applies the increment in the world frame (left multiply).

    Returns
    -------
    (T, 4) array of unit quaternions, with ``out[0] == normalize(q0)``.

    Each step renormalizes to fight floating-point drift.
    """
    gyro = np.asarray(gyro, dtype=np.float64)
    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError(f"gyro must be (T, 3), got {gyro.shape}")
    if frame not in ("body", "world"):
        raise ValueError(f"frame must be 'body' or 'world', got {frame!r}")
    T = gyro.shape[0]
    out = np.empty((T, 4), dtype=np.float64)
    q = quat_normalize(q0)
    out[0] = q
    for k in range(1, T):
        dq = delta_quat(gyro[k], dt)
        if frame == "body":
            q = quat_multiply(q, dq)
        else:
            q = quat_multiply(dq, q)
        q = quat_normalize(q)
        out[k] = q
    return out
