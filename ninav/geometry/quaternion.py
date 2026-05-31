"""Quaternion algebra with a single, asserted convention.

Convention (repo-wide, enforced by tests):
    * Quaternions are **scalar-last**: ``q = [x, y, z, w]``, matching
      ``scipy.spatial.transform.Rotation.from_quat`` / ``as_quat`` (default).
    * Hamilton product (right-handed), active rotations.
    * A unit quaternion ``q`` rotates a body-frame vector ``v`` into the world
      frame via ``R(q) @ v`` where ``R = quat_to_rotmat(q)``.

This module replaces the divergent, convention-mismatched helpers in the old
``utils.py`` (``rotation_matrix_2_quaternion`` was scalar-first while ``Omega``
was scalar-last -- see bug C3 in REVIEW.md). Everything here is scalar-last and
checked against scipy in the test-suite.
"""
from __future__ import annotations

import numpy as np

#: Index of the real (scalar) component under the scalar-last convention.
W = 3
#: Identity quaternion (no rotation), scalar-last.
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])

_EPS = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q`` scaled to unit norm along the last axis.

    Works on a single quaternion ``(4,)`` or a batch ``(..., 4)``. A zero
    quaternion maps to the identity rather than producing NaNs.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    safe = np.where(norm < _EPS, 1.0, norm)
    out = q / safe
    # Replace any quaternion that was ~zero with identity.
    zero = (norm < _EPS)[..., 0]
    if np.ndim(zero) == 0:
        if zero:
            out = IDENTITY.copy()
    elif zero.any():
        out[zero] = IDENTITY
    return out


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate ``[-x, -y, -z, w]`` (inverse for unit quaternions)."""
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p (x) q`` (scalar-last), batched over leading dims."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    px, py, pz, pw = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x = pw * qx + px * qw + py * qz - pz * qy
    y = pw * qy - px * qz + py * qw + pz * qx
    z = pw * qz + px * qy - py * qx + pz * qw
    w = pw * qw - px * qx - py * qy - pz * qz
    return np.stack([x, y, z, w], axis=-1)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a (batch of) unit quaternion(s) to rotation matrix/matrices.

    Returns shape ``(..., 3, 3)``. Input is normalized defensively.
    """
    q = quat_normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a scalar-last unit quaternion.

    Uses Shepperd's numerically stable 4-case method (picks the largest
    pivot to avoid the ``1/(4*q0)`` blow-up near 180 deg that broke the old
    ``rotation_matrix_2_quaternion`` -- bug H2). Single matrix only ``(3,3)``.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"rotmat_to_quat expects a single (3,3) matrix, got {R.shape}")
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(max(trace + 1.0, 0.0)) * 2.0  # s = 4*w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(max(1.0 + m00 - m11 - m22, 0.0)) * 2.0  # s = 4*x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(max(1.0 + m11 - m00 - m22, 0.0)) * 2.0  # s = 4*y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(max(1.0 + m22 - m00 - m11, 0.0)) * 2.0  # s = 4*z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w]))


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion for a rotation of ``angle`` (rad) about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < _EPS:
        return IDENTITY.copy()
    axis = axis / n
    half = 0.5 * angle
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic angle (radians) between two unit quaternions.

    Double-cover aware (``q`` and ``-q`` are the same rotation): uses
    ``2 * arccos(|<q1, q2>|)`` with the argument clamped for numerical safety.
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)
