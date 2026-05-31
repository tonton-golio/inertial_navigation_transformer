"""On-the-fly augmentation for IMU windows (applied during training only).

The single most important augmentation is **random yaw about gravity**, applied
to BOTH the input window and the target velocity together (RoNIN: ~+7%). Without
it, a non-equivariant net cannot become heading-invariant. The IMU window is in a
gravity-aligned frame, so a yaw rotation keeps Z (gravity) fixed.

All ops act on a single window: ``acc`` and ``gyro`` are ``(L, 3)``, the target
is 2D ``(2,)`` (horizontal velocity/displacement) or 3D ``(3,)``.
"""
from __future__ import annotations

import numpy as np

from ..config import AugmentCfg
from ..geometry.frames import yaw_rotation


def apply_yaw(acc, gyro, target, angle, rng=None):
    """Rotate gravity-aligned ``acc``/``gyro`` and the target by ``angle`` (rad).

    Gyro is a pseudovector but transforms like a vector under a proper rotation
    (det=+1), so the same matrix applies. Z (gravity) is unchanged.
    """
    R = yaw_rotation(angle)
    acc_r = acc @ R.T
    gyro_r = gyro @ R.T
    target = np.asarray(target, dtype=np.float64)
    if target.shape[-1] == 2:
        R2 = R[:2, :2]
        tgt_r = target @ R2.T
    else:
        tgt_r = target @ R.T
    return acc_r, gyro_r, tgt_r


def augment_window(acc, gyro, target, cfg: AugmentCfg, rng):
    """Apply the configured augmentations to one window. Returns (acc, gyro, target)."""
    acc = np.array(acc, dtype=np.float64, copy=True)
    gyro = np.array(gyro, dtype=np.float64, copy=True)
    target = np.array(target, dtype=np.float64, copy=True)

    if cfg.random_yaw:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        acc, gyro, target = apply_yaw(acc, gyro, target, angle, rng)

    if cfg.gravity_tilt_deg > 0:
        # small tilt about a random horizontal axis (bridges to noisier runtime ori)
        ax = rng.standard_normal(3); ax[2] = 0.0
        nrm = np.linalg.norm(ax)
        if nrm > 1e-8:
            ax /= nrm
            ang = np.deg2rad(rng.uniform(0.0, cfg.gravity_tilt_deg))
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rt = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
            acc = acc @ Rt.T
            gyro = gyro @ Rt.T

    if cfg.bias_acc > 0:
        acc = acc + rng.uniform(-cfg.bias_acc, cfg.bias_acc, size=3)
    if cfg.bias_gyro > 0:
        gyro = gyro + rng.uniform(-cfg.bias_gyro, cfg.bias_gyro, size=3)
    if cfg.gaussian_noise_acc > 0:
        acc = acc + rng.normal(0.0, cfg.gaussian_noise_acc, size=acc.shape)
    if cfg.gaussian_noise_gyro > 0:
        gyro = gyro + rng.normal(0.0, cfg.gaussian_noise_gyro, size=gyro.shape)

    return acc, gyro, target
