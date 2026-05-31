"""Coordinate-frame construction and the gravity-aligned heading-agnostic frame.

Replaces the old ``get_body2world_rot`` (bugs C2 and H3):
    * the original set ``DOWN = +accelerometer``, but a stationary accelerometer
      measures specific force pointing **up** (``+g``), so the world frame was
      vertically inverted -- here ``DOWN = -acc/|acc|``;
    * the original was also called with arguments swapped ``(acc, mag)`` against
      a ``(m0, a0)`` signature -- this version takes explicit, named args.

The gravity-aligned heading-agnostic frame (HACF) is the input frame used by
RoNIN/TLIO: the IMU window is rotated so that Z is aligned with gravity, while
the (unobservable-from-IMU) heading about gravity is deliberately *not* pinned
to true north. We therefore align gravity only and leave yaw arbitrary, which
is exactly what random-yaw augmentation then exploits.
"""
from __future__ import annotations

import numpy as np

from .quaternion import quat_to_rotmat

_EPS = 1e-12
GRAVITY = 9.80665  # m/s^2


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < _EPS:
        raise ValueError("cannot normalize a near-zero vector")
    return v / n


def body_to_world_ned(mag: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Body->world (NED) rotation from one magnetometer + accelerometer sample.

    Parameters
    ----------
    mag : (3,)  magnetometer reading (body frame)
    acc : (3,)  accelerometer reading (body frame); at rest this points up (+g)

    Returns
    -------
    R : (3, 3)  rotation whose **columns** are the world NORTH, EAST, DOWN axes
        expressed in the body frame, with ``det(R) = +1``.

    A body-frame vector ``v_b`` is mapped to world (NED) coords via ``R.T @ v_b``
    (columns are world axes in body coords, so the transpose projects onto them).
    """
    acc = np.asarray(acc, dtype=np.float64).reshape(3)
    mag = np.asarray(mag, dtype=np.float64).reshape(3)
    down = -_unit(acc)                 # gravity (down) is opposite specific force
    east = _unit(np.cross(down, mag))  # east _|_ down and the magnetic vector
    north = np.cross(east, down)       # completes the right-handed NED triad
    R = np.column_stack([north, east, down])
    return R


def gravity_align_rotation(acc_mean: np.ndarray) -> np.ndarray:
    """Rotation that maps the body frame to a gravity-aligned frame (Z up).

    Builds the minimal rotation that sends the measured specific-force direction
    (``acc_mean``, points up at rest) onto ``+Z``. Heading about Z is left
    arbitrary (heading-agnostic). Returns a ``(3, 3)`` rotation matrix ``R`` such
    that ``R @ acc_dir ~= +Z``.
    """
    acc_dir = _unit(acc_mean)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(acc_dir, z)
    c = float(np.dot(acc_dir, z))
    s = np.linalg.norm(v)
    if s < _EPS:
        # Already (anti)aligned with Z.
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def yaw_rotation(angle_rad: float) -> np.ndarray:
    """3x3 rotation about the gravity (Z) axis -- used for heading augmentation."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_window(R: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Apply a single ``(3,3)`` rotation to every ``(...,3)`` vector in a window."""
    vecs = np.asarray(vecs, dtype=np.float64)
    return vecs @ R.T


def gravity_align_from_quat(window_acc: np.ndarray, window_gyro: np.ndarray,
                            q_start: np.ndarray):
    """Rotate accel+gyro of a window into the frame defined by ``q_start``.

    TLIO-style: use the orientation at the window start to express the whole
    window in a gravity-aligned, heading-from-start frame. ``q_start`` is a
    scalar-last unit quaternion (body->world). Returns ``(acc_w, gyro_w)``.
    """
    R = quat_to_rotmat(q_start)  # body->world
    return rotate_window(R, window_acc), rotate_window(R, window_gyro)
