"""Physically self-consistent synthetic IMU + trajectory generator.

The real RoNIN HDF5 data is not in the repo, so the test-suite and smoke-train
run on synthetic recordings generated here. The generator is the *forward model*
of the navigation problem, so the pipeline (gravity removal, frame rotation,
double integration, velocity targets, reconstruction) can be inverted back to
the known ground truth and checked to ~0 error.

Conventions (match :mod:`ninav.geometry`):
    * world frame: ENU-like, Z up. Gravity acceleration vector ``g = [0,0,-G]``.
    * an accelerometer measures specific force ``f = a_world - g`` expressed in
      the BODY frame: ``acc_body = R(q)^T @ (a_world - g)`` (at rest -> +Z up, |g|).
    * gyro measures body-frame angular velocity ``omega_body`` (rad/s).
    * orientation ``q`` is body->world, scalar-last.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.propagate import integrate_gyro
from ..geometry.quaternion import quat_to_rotmat

G = 9.80665


@dataclass
class Recording:
    """One synthetic (or real) recording. Arrays are time-major ``(T, .)``."""
    rec_id: str
    fs: float
    time: np.ndarray        # (T,)
    gyro: np.ndarray        # (T, 3) body angular velocity (rad/s)
    acc: np.ndarray         # (T, 3) body specific force (m/s^2)
    mag: np.ndarray         # (T, 3) body magnetic field
    pos: np.ndarray         # (T, 3) world position (m)
    ori: np.ndarray         # (T, 4) body->world unit quaternion (scalar-last)
    vel: np.ndarray         # (T, 3) world velocity (m/s)


def generate_recording(rec_id: str = "synthetic", duration_s: float = 30.0,
                       fs: float = 200.0, seed: int = 0,
                       mag_world: np.ndarray | None = None) -> Recording:
    """Generate one smooth, physically consistent walking-like recording."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    dt = 1.0 / fs
    t = np.arange(n) * dt

    # --- world-frame trajectory: a few sinusoids in the horizontal plane + tiny z ---
    def smooth_signal(n_components, amp, fmax):
        sig = np.zeros(n)
        for _ in range(n_components):
            f = rng.uniform(0.05, fmax)
            ph = rng.uniform(0, 2 * np.pi)
            a = rng.uniform(0.3, 1.0) * amp
            sig += a * np.sin(2 * np.pi * f * t + ph)
        return sig

    px = smooth_signal(3, amp=6.0, fmax=0.3) + 0.6 * t   # net drift forward
    py = smooth_signal(3, amp=6.0, fmax=0.3)
    pz = smooth_signal(2, amp=0.05, fmax=0.4)            # small vertical bob
    pos = np.stack([px, py, pz], axis=1)
    pos -= pos[0]

    # velocity / world acceleration by central differences (consistent with cumsum)
    vel = np.gradient(pos, dt, axis=0)
    acc_world = np.gradient(vel, dt, axis=0)

    # --- body angular velocity: smooth, walking-like; integrate to orientation ---
    wx = smooth_signal(2, amp=0.4, fmax=0.5)
    wy = smooth_signal(2, amp=0.4, fmax=0.5)
    wz = smooth_signal(2, amp=0.8, fmax=0.3) + 0.15      # slow heading drift
    omega_body = np.stack([wx, wy, wz], axis=1)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    ori = integrate_gyro(q0, omega_body, dt, frame="body")

    # --- synthesize body-frame sensor readings from the world-frame physics ---
    g_vec = np.array([0.0, 0.0, -G])               # gravity acceleration (world)
    if mag_world is None:
        mag_world = np.array([20.0, 0.0, 40.0])    # arbitrary NED-ish field (uT)
    R = quat_to_rotmat(ori)                         # (T,3,3) body->world
    Rt = np.transpose(R, (0, 2, 1))                 # world->body
    specific_force_world = acc_world - g_vec        # f = a - g
    acc_body = np.einsum("tij,tj->ti", Rt, specific_force_world)
    mag_body = np.einsum("tij,j->ti", Rt, mag_world)

    return Recording(rec_id=rec_id, fs=fs, time=t, gyro=omega_body, acc=acc_body,
                     mag=mag_body, pos=pos, ori=ori, vel=vel)


def generate_dataset(n_recordings: int = 6, duration_s: float = 20.0,
                     fs: float = 200.0, seed: int = 0) -> list[Recording]:
    """A small set of distinct recordings for split/train/eval tests."""
    return [generate_recording(rec_id=f"syn_{i:02d}", duration_s=duration_s,
                               fs=fs, seed=seed + 1000 * i)
            for i in range(n_recordings)]
