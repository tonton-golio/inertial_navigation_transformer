"""Geometry: quaternion algebra, attitude propagation, and frame construction."""
from .quaternion import (
    IDENTITY,
    W,
    quat_angle,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    rotmat_to_quat,
)
from .propagate import delta_quat, integrate_gyro, omega_matrix, theta_matrix
from .frames import (
    GRAVITY,
    body_to_world_ned,
    gravity_align_from_quat,
    gravity_align_rotation,
    rotate_window,
    yaw_rotation,
)

__all__ = [
    "IDENTITY", "W", "quat_angle", "quat_conjugate", "quat_from_axis_angle",
    "quat_multiply", "quat_normalize", "quat_to_rotmat", "rotmat_to_quat",
    "delta_quat", "integrate_gyro", "omega_matrix", "theta_matrix",
    "GRAVITY", "body_to_world_ned", "gravity_align_from_quat",
    "gravity_align_rotation", "rotate_window", "yaw_rotation",
]
