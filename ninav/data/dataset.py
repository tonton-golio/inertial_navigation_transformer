"""Assemble recordings into model-ready windows + targets.

Canonical pipeline (correct version of the project's "position from IMU +
orientation" idea):

    1. Get per-sample orientation q_t (body->world): ground truth, or integrated
       from the gyro with the corrected propagator.
    2. Rotate body accel + gyro into the WORLD frame (gravity-aligned by
       construction). Gravity stays in accel as a constant z offset (RoNIN keeps
       it). These 6 channels are the network input.
    3. Build per-recording windows (no cross-recording seams) and a per-window
       2D velocity target over the stride interval (so reconstruction telescopes).
    4. Heading robustness comes from random-yaw augmentation at train time.

Everything is per-recording so windows never straddle two trajectories, the
train/test target definition is identical, and reconstruction uses the single
shared operator in :mod:`ninav.reconstruct`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import AugmentCfg, WindowCfg
from ..geometry.quaternion import quat_to_rotmat
from .augment import augment_window
from .synthetic import Recording
from .targets import make_target, start_offsets
from .windowing import create_windows


def _orientation(rec: Recording, source: str) -> np.ndarray:
    if source == "gt":
        return np.asarray(rec.ori, dtype=np.float64)
    if source == "gyro":
        from ..geometry.propagate import integrate_gyro
        return integrate_gyro(rec.ori[0], rec.gyro, 1.0 / rec.fs, frame="body")
    raise ValueError(f"orientation source must be 'gt' or 'gyro', got {source!r}")


def world_frame_imu(rec: Recording, orientation: str = "gt"):
    """Return ``(T, 6)`` world-frame [acc(3), gyro(3)] using per-sample orientation."""
    q = _orientation(rec, orientation)
    R = quat_to_rotmat(q)                       # (T,3,3) body->world
    acc_w = np.einsum("tij,tj->ti", R, rec.acc)
    gyro_w = np.einsum("tij,tj->ti", R, rec.gyro)
    return np.concatenate([acc_w, gyro_w], axis=1)


def prepare_recording(rec: Recording, cfg: WindowCfg, target_kind: str = "velocity",
                      orientation: str = "gt", dims: int = 2) -> dict:
    """Windows + targets + reconstruction anchors for ONE recording."""
    stream = world_frame_imu(rec, orientation)          # (T, 6)
    windows, starts = create_windows(stream, cfg.window_len, cfg.stride)
    pos = np.asarray(rec.pos, dtype=np.float64)
    target = make_target(target_kind, pos, starts, cfg.stride, cfg.dt, dims)
    return {
        "rec_id": rec.rec_id,
        "input": windows.astype(np.float32),            # (N, L, 6)
        "target": target.astype(np.float32),            # (N, dims)
        "starts": starts,                               # (N,)
        "p_start": start_offsets(pos, starts, dims),    # (N, dims)
        "pos": pos[:, :dims],                           # (T, dims) reference
    }


def build_dataset(recordings, cfg: WindowCfg, target_kind: str = "velocity",
                  orientation: str = "gt", dims: int = 2):
    """Prepare a list of recordings; return ``(per_recording, stacked)``.

    ``per_recording`` is a list of :func:`prepare_recording` dicts (for eval /
    reconstruction). ``stacked`` concatenates inputs/targets across recordings
    for training (with a ``rec_index`` mapping each window to its recording).
    """
    per = [prepare_recording(r, cfg, target_kind, orientation, dims) for r in recordings]
    per = [p for p in per if p["input"].shape[0] > 0]
    if not per:
        raise ValueError("no windows produced (recordings too short for the window/stride?)")
    inputs = np.concatenate([p["input"] for p in per], axis=0)
    targets = np.concatenate([p["target"] for p in per], axis=0)
    rec_index = np.concatenate([np.full(p["input"].shape[0], i) for i, p in enumerate(per)])
    return per, {"input": inputs, "target": targets, "rec_index": rec_index}


class WindowDataset(Dataset):
    """Torch dataset of IMU windows. Yields ``(x[C, L], y[dims])`` channel-first.

    With ``augment=True`` (training) the configured augmentations are applied
    per item, jointly to the input window and the target (e.g. random yaw).
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray,
                 augment_cfg: AugmentCfg | None = None, seed: int = 0):
        assert inputs.ndim == 3 and inputs.shape[2] == 6, inputs.shape
        self.acc = np.ascontiguousarray(inputs[..., :3], dtype=np.float32)   # (N,L,3)
        self.gyro = np.ascontiguousarray(inputs[..., 3:], dtype=np.float32)
        self.targets = np.ascontiguousarray(targets, dtype=np.float32)
        self.augment_cfg = augment_cfg
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.acc.shape[0]

    def __getitem__(self, i):
        acc, gyro, tgt = self.acc[i], self.gyro[i], self.targets[i]
        if self.augment_cfg is not None:
            acc, gyro, tgt = augment_window(acc, gyro, tgt, self.augment_cfg, self._rng)
        x = np.concatenate([acc, gyro], axis=1).T          # (6, L) channel-first
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), \
            torch.from_numpy(np.ascontiguousarray(tgt, dtype=np.float32))
