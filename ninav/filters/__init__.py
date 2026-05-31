"""From-scratch navigation filters.

Two families:

* **Attitude (AHRS)** estimators in :mod:`ninav.filters.ahrs` -- Madgwick,
  Mahony, and a quaternion EKF -- each turn one recording's IMU stream into a
  sequence of body->world orientations (scalar-last quaternions).
* **Displacement fusion** in :mod:`ninav.filters.sc_ekf` -- a single-clone
  stochastic-cloning Kalman filter that fuses per-window 2D displacement
  predictions into a smooth trajectory.
"""
from .ahrs import GRAVITY, ekf_orientation, madgwick, mahony
from .sc_ekf import DisplacementEKF

__all__ = [
    "GRAVITY",
    "ekf_orientation",
    "madgwick",
    "mahony",
    "DisplacementEKF",
]
