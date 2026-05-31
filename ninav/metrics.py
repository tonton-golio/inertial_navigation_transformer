"""RoNIN-compatible trajectory error metrics: ATE and RTE.

Fixes the old metric bugs:
    * ATE was ``mean(||error||)`` not an RMSE (bug M1);
    * RTE was either ``ATE * fraction-of-trajectory`` (meaningless, H11) or a
      windowed mean-ATE with an off-by-one and a div-by-zero (M2).

Definitions (2D position, navigation frame), per RoNIN ``metric.py``:

    ATE  = sqrt( mean_t || est_t - gt_t ||^2 )            (absolute RMSE)
    RTE  = sqrt( mean_t || (est_{t+d} - est_t) - (gt_{t+d} - gt_t) ||^2 )

where ``d`` is the number of samples in one minute (``pred_per_min``). For
trajectories shorter than one minute, RoNIN uses ``d = T-1`` and scales the
result by ``pred_per_min / T``.

Compute per recording, then average across recordings (report median/std too).
"""
from __future__ import annotations

import numpy as np


def compute_ate(est: np.ndarray, gt: np.ndarray) -> float:
    """Absolute Trajectory Error: RMSE of the per-point Euclidean position error."""
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if est.shape != gt.shape:
        raise ValueError(f"shape mismatch: est {est.shape} vs gt {gt.shape}")
    sq = np.sum((est - gt) ** 2, axis=-1)  # squared Euclidean error per point
    return float(np.sqrt(np.mean(sq)))


def compute_rte(est: np.ndarray, gt: np.ndarray, delta: int) -> float:
    """Relative Trajectory Error over a fixed sample lag ``delta``."""
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if est.shape != gt.shape:
        raise ValueError(f"shape mismatch: est {est.shape} vs gt {gt.shape}")
    if delta < 1:
        raise ValueError(f"delta must be >= 1, got {delta}")
    if delta >= est.shape[0]:
        raise ValueError(f"delta ({delta}) must be < trajectory length ({est.shape[0]})")
    err = (est[delta:] - est[:-delta]) - (gt[delta:] - gt[:-delta])
    sq = np.sum(err ** 2, axis=-1)
    return float(np.sqrt(np.mean(sq)))


def compute_ate_rte(est: np.ndarray, gt: np.ndarray,
                    pred_per_min: int = 12000) -> tuple[float, float]:
    """Return ``(ate, rte)`` with RoNIN's short-sequence handling.

    ``pred_per_min`` is the number of trajectory samples per minute (12000 for a
    200 Hz stream sampled every frame; for windowed predictions at stride ``s``
    it is ``60 * fs / s``).
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    T = est.shape[0]
    ate = compute_ate(est, gt)
    if T < 2:
        return ate, float("nan")
    if T <= pred_per_min:
        ratio = pred_per_min / T
        rte = compute_rte(est, gt, T - 1) * ratio
    else:
        rte = compute_rte(est, gt, pred_per_min)
    return ate, rte


def aggregate_metrics(per_recording: list[tuple[float, float]]) -> dict:
    """Aggregate ``(ate, rte)`` pairs across recordings (mean/median/std)."""
    arr = np.asarray(per_recording, dtype=np.float64).reshape(-1, 2)
    ate, rte = arr[:, 0], arr[:, 1]
    return {
        "ate_mean": float(np.nanmean(ate)), "ate_median": float(np.nanmedian(ate)),
        "ate_std": float(np.nanstd(ate)),
        "rte_mean": float(np.nanmean(rte)), "rte_median": float(np.nanmedian(rte)),
        "rte_std": float(np.nanstd(rte)),
        "n": int(arr.shape[0]),
    }
