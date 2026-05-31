"""Load-bearing tests for trajectory reconstruction + RoNIN error metrics.

These pin the contracts that the old project got wrong, and would silently
regress if the spine drifted:

* C7  -- predictions reconstructed by ``cumsum`` of per-window displacements
         assumed ``stride == displacement span``, so a *perfect* model drifted
         linearly whenever windows overlapped. Fixed by integrating a velocity
         stream with ``dt = stride/fs`` (telescopes to zero error for any stride).
* C8  -- ground truth reconstructed by a *different* formula than predictions.
         The single ``integrate_velocity`` / ``reconstruct_from_displacement``
         operator is applied identically to est and gt here.
* M1  -- ATE was ``mean(||error||)`` instead of an RMSE.
* M2  -- RTE short-sequence handling had an off-by-one / div-by-zero.
* H11 -- RTE was once ``ATE * fraction-of-trajectory`` (meaningless).

All synthetic data is tiny and CPU-only.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ninav.data.synthetic import generate_recording
from ninav.data.targets import start_offsets, velocity_target, displacement_target
from ninav.data.windowing import valid_starts
from ninav.metrics import compute_ate, compute_rte, compute_ate_rte
from ninav.reconstruct import integrate_velocity, reconstruct_from_displacement


FS = 200.0
DT = 1.0 / FS
WINDOW_LEN = 50


# ---------------------------------------------------------------------------
# (1) PERFECT-MODEL-ZERO-ATE: velocity targets reconstruct to ~0 ATE for any
#     stride. Guards C7 (overlap-dependent cumsum drift) and C8 (one operator
#     for est and gt). The old cumsum drifted linearly when overlap != 1.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stride", [1, 5, 10, 50])
def test_perfect_model_zero_ate_every_stride(stride):
    rec = generate_recording(rec_id="perfect", duration_s=5.0, fs=FS, seed=7)
    positions = rec.pos

    starts = valid_starts(positions.shape[0], WINDOW_LEN, stride)
    assert starts.size > 1, "need several windows to exercise integration"
    # windows are spaced exactly ``stride`` apart -- this is what makes the
    # velocity stream telescope back to the true positions.
    assert np.array_equal(starts, np.arange(0, starts[-1] + 1, stride))

    # "perfect model" == the exact velocity target for each window.
    v = velocity_target(positions, starts, stride, DT, dims=2)
    gt = start_offsets(positions, starts, dims=2)

    # SAME operator for est and gt; dt = stride/fs is the time between starts.
    est = integrate_velocity(v, dt=stride * DT, p0=gt[0])

    ate = compute_ate(est, gt)
    assert ate < 1e-6, f"stride={stride}: perfect model must give ~0 ATE, got {ate}"


def test_perfect_model_drift_is_zero_not_linear():
    """Explicitly contrast with the buggy cumsum: at an overlapping stride the
    reconstruction must NOT accumulate per-window displacement at every step."""
    stride = 5  # window_len=50 -> heavy overlap, the regime that broke C7
    rec = generate_recording(rec_id="drift", duration_s=4.0, fs=FS, seed=11)
    positions = rec.pos

    starts = valid_starts(positions.shape[0], WINDOW_LEN, stride)
    v = velocity_target(positions, starts, stride, DT, dims=2)
    gt = start_offsets(positions, starts, dims=2)
    est = integrate_velocity(v, dt=stride * DT, p0=gt[0])

    # final-point error stays at machine zero; a cumsum-with-stride-1 style bug
    # would scale the displacement and blow this up.
    final_err = np.linalg.norm(est[-1] - gt[-1])
    assert final_err < 1e-6, f"end drift {final_err} -- looks like C7 regressed"


# ---------------------------------------------------------------------------
# (2) ATE is the RMSE of the Euclidean error, NOT the mean. Guards M1.
# ---------------------------------------------------------------------------
def test_ate_is_rmse_not_mean():
    # errors {5, 0, 10} along one axis; the other coordinate is identical.
    gt = np.zeros((3, 2))
    est = np.array([[5.0, 0.0], [0.0, 0.0], [10.0, 0.0]])

    ate = compute_ate(est, gt)
    expected_rmse = math.sqrt((25.0 + 0.0 + 100.0) / 3.0)  # == 6.4549...
    mean_error = (5.0 + 0.0 + 10.0) / 3.0                   # == 5.0 (the bug)

    assert ate == pytest.approx(expected_rmse, rel=0, abs=1e-9)
    assert ate == pytest.approx(6.454972243679028, abs=1e-9)
    assert abs(ate - mean_error) > 1.0, "ATE collapsed to the mean (M1 regressed)"


def test_ate_uses_euclidean_norm_across_dims():
    """A point off by 3 in x and 4 in y contributes 5 (||.||), squared in RMSE."""
    gt = np.zeros((1, 2))
    est = np.array([[3.0, 4.0]])
    assert compute_ate(est, gt) == pytest.approx(5.0, abs=1e-12)


# ---------------------------------------------------------------------------
# (3) Constant offset: RTE ~ 0 (relative motion is identical) but ATE equals the
#     offset magnitude. Guards H11 (RTE != ATE-times-a-fraction) and the RTE
#     definition (relative displacements).
# ---------------------------------------------------------------------------
def test_constant_offset_zero_rte_nonzero_ate():
    rng = np.random.default_rng(0)
    T = 300
    gt = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    offset = np.array([3.0, 4.0])
    est = gt + offset  # rigidly shifted trajectory

    ate = compute_ate(est, gt)
    rte = compute_rte(est, gt, delta=50)

    assert ate == pytest.approx(np.linalg.norm(offset), abs=1e-9)  # == 5.0
    assert ate == pytest.approx(5.0, abs=1e-9)
    # a constant offset cancels in every (est_{t+d}-est_t) - (gt_{t+d}-gt_t).
    assert rte == pytest.approx(0.0, abs=1e-9)
    # H11 guard: RTE is NOT some fraction of ATE here -- it is genuinely ~0.
    assert rte < 1e-6 * max(ate, 1.0)


# ---------------------------------------------------------------------------
# (4) compute_ate_rte short-sequence branch: T < pred_per_min uses delta = T-1
#     and scales by pred_per_min / T, and stays finite. Guards M2 / H11.
# ---------------------------------------------------------------------------
def test_compute_ate_rte_short_sequence_branch():
    pred_per_min = 1200  # 60 * 200 / 10 (stride 10)
    T = 40               # well under one "minute" of predictions
    assert T < pred_per_min

    rng = np.random.default_rng(1)
    gt = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    est = gt + rng.normal(scale=0.1, size=(T, 2))

    ate, rte = compute_ate_rte(est, gt, pred_per_min=pred_per_min)

    # manual reproduction of the documented short-seq formula.
    expected_rte = compute_rte(est, gt, T - 1) * (pred_per_min / T)
    assert rte == pytest.approx(expected_rte, rel=1e-12)
    assert math.isfinite(ate) and math.isfinite(rte)
    assert ate == pytest.approx(compute_ate(est, gt), rel=1e-12)
    # scaling factor must be applied (ratio > 1 here), not silently dropped.
    assert rte > compute_rte(est, gt, T - 1)


def test_compute_ate_rte_long_sequence_uses_pred_per_min_delta():
    """T > pred_per_min: RTE uses delta == pred_per_min directly (no scaling)."""
    pred_per_min = 30
    T = 200
    assert T > pred_per_min

    rng = np.random.default_rng(2)
    gt = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    est = gt + rng.normal(scale=0.2, size=(T, 2))

    _, rte = compute_ate_rte(est, gt, pred_per_min=pred_per_min)
    assert rte == pytest.approx(compute_rte(est, gt, pred_per_min), rel=1e-12)
    assert math.isfinite(rte)


# ---------------------------------------------------------------------------
# (5) reconstruct_from_displacement places displacements at their true offsets
#     exactly. Guards C7 (no overlap-dependent cumsum) / C8 (anchored at known
#     start positions, same operator for est and gt).
# ---------------------------------------------------------------------------
def test_reconstruct_from_displacement_places_at_true_offsets():
    rec = generate_recording(rec_id="disp", duration_s=4.0, fs=FS, seed=21)
    positions = rec.pos
    stride = 10

    starts = valid_starts(positions.shape[0], WINDOW_LEN, stride)
    assert starts.size > 1

    disp = displacement_target(positions, starts, stride, dims=2)
    offsets = start_offsets(positions, starts, dims=2)

    est = reconstruct_from_displacement(disp, offsets)

    # By construction est[j] == pos[starts[j] + stride].
    expected = positions[starts + stride, :2]
    np.testing.assert_allclose(est, expected, atol=1e-9)

    # And as a trajectory vs the per-window start offsets it must be exact:
    # a perfect displacement prediction is anchored, never cumsum-drifted.
    assert compute_ate(est, expected) < 1e-9


def test_reconstruct_from_displacement_is_pure_addition():
    """Simple algebraic identity, independent of synthetic data."""
    disp = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    offsets = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    est = reconstruct_from_displacement(disp, offsets)
    np.testing.assert_allclose(est, disp + offsets, atol=1e-12)
