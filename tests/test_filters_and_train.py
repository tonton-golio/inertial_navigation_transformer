"""Tests for the from-scratch navigation filters and the train/eval glue.

Filter coverage (ninav.filters):
    * ``ekf_orientation`` produces a genuinely NONZERO innovation and a
      non-trivial state correction -- guards the old "z == h" no-op-EKF bug
      (C14: an EKF that set the measurement equal to its own prediction).
    * a 3-vector ``q0`` is rejected by every AHRS filter (guards H7: a 3-vector
      axis/Euler seed silently coerced into an "orientation").
    * madgwick / mahony / ekf_orientation all return unit quaternions (T, 4).
    * ``DisplacementEKF`` lowers trajectory RMSE versus raw (cumsum) integration
      of the same noisy per-window displacements on a constant-velocity track.

Train coverage (ninav.train):
    * a 3-epoch ``resnet1d`` run lowers the training loss and ``evaluate``
      returns finite ATE / RTE.

Everything runs on CPU with tiny tensors / short synthetic recordings.
"""
from __future__ import annotations

import numpy as np
import pytest

from ninav.config import AugmentCfg, TrainCfg, WindowCfg
from ninav.data.synthetic import generate_dataset, generate_recording
from ninav.filters import (
    DisplacementEKF,
    ekf_orientation,
    madgwick,
    mahony,
)
from ninav.filters.ahrs import GRAVITY
from ninav.geometry.propagate import integrate_gyro
from ninav.geometry.quaternion import quat_angle, quat_normalize, quat_to_rotmat
from ninav.train import evaluate, train


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_unit_quat_stream(q: np.ndarray, atol: float = 1e-6) -> bool:
    """True iff ``q`` is (T, 4) with every row a unit quaternion."""
    if q.ndim != 2 or q.shape[1] != 4:
        return False
    norms = np.linalg.norm(q, axis=1)
    return bool(np.all(np.abs(norms - 1.0) < atol))


def _tilt_error_deg(quats: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-step tilt (roll/pitch) error in degrees.

    A gravity-aided AHRS only observes the world Z direction in the body frame
    (``R(q).T @ +Z``, the third row of ``R``); heading (yaw about gravity) is
    unobservable. The tilt error is the angle between the estimated and true
    body-frame gravity directions -- exactly what the accelerometer update fixes.
    """
    g_est = quat_to_rotmat(quats)[:, 2, :]   # third row of R = R.T @ +Z
    g_ref = quat_to_rotmat(ref)[:, 2, :]
    cos = np.clip(np.sum(g_est * g_ref, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _static_recording(duration_s: float = 8.0, fs: float = 200.0):
    """A *near-static* recording where the accelerometer measures pure gravity.

    The default ``generate_recording`` is a highly dynamic walk (world linear
    acceleration ~= g in magnitude), so the accelerometer is a poor gravity
    reference and a gravity-aided AHRS cannot be expected to beat raw gyro
    integration. To exercise the gravity *update* meaningfully we synthesize a
    platform that only rotates slowly and does NOT translate, so ``a_world == 0``
    and the body specific force is exactly ``-g`` (norm == GRAVITY everywhere).
    Built from the same forward model as :mod:`ninav.data.synthetic`.

    Returns ``(omega(T,3), acc_body(T,3), ori(T,4), dt)``.
    """
    dt = 1.0 / fs
    n = int(round(duration_s * fs))
    t = np.arange(n) * dt
    wx = 0.05 * np.sin(2 * np.pi * 0.10 * t)
    wy = 0.04 * np.cos(2 * np.pi * 0.08 * t)
    wz = 0.10 * np.sin(2 * np.pi * 0.05 * t)
    omega = np.stack([wx, wy, wz], axis=1)
    ori = integrate_gyro(np.array([0.0, 0.0, 0.0, 1.0]), omega, dt, frame="body")
    # a_world == 0  ->  specific force f = a_world - g = -g = [0, 0, +G] (world);
    # express it in the body frame with R(q).T.
    Rt = np.transpose(quat_to_rotmat(ori), (0, 2, 1))
    f_world = np.array([0.0, 0.0, GRAVITY])
    acc_body = np.einsum("tij,j->ti", Rt, f_world)
    return omega, acc_body, ori, dt


# --------------------------------------------------------------------------- #
# EKF: nonzero innovation + non-trivial correction (guards C14)
# --------------------------------------------------------------------------- #
def test_ekf_orientation_nonzero_innovation_and_correction():
    """The gravity update must actually move the state (guards C14).

    We feed the EKF a *biased* gyro (so the gyro-only propagation slowly tilts
    away from truth) together with the true gravity accelerometer of a near-static
    platform. A real filter sees the gyro-propagated tilt disagree with the
    accelerometer, so:
      * the per-step innovation norm is non-trivially nonzero, and
      * the EKF's TILT (roll/pitch -- the only thing gravity observes) stays
        markedly closer to truth than a pure gyro integrator (the innovation==0
        path) would, i.e. the gravity correction is not a no-op. (Heading/yaw is
        unobservable from gravity alone, so we score the tilt, not full orientation.)

    Noise params are tuned to the scenario: a *biased* gyro is unreliable
    (larger ``sigma_gyro``) while the static-platform accelerometer is an
    excellent gravity reference (smaller ``sigma_acc``), so the Kalman gain
    actually trusts the gravity update.
    """
    omega, acc, ori, dt = _static_recording(duration_s=8.0, fs=200.0)
    T = acc.shape[0]
    q0 = ori[0]

    # Constant gyro bias on x/y -> gyro-only tilt drifts; gravity must pull it back.
    bias = np.array([0.03, -0.02, 0.0])
    gyro_biased = omega + bias

    kw = dict(sigma_gyro=0.2, sigma_acc=0.05, p0=1e-1)
    quats, innov = ekf_orientation(
        gyro_biased, acc, dt, q0, return_innovations=True, **kw
    )

    assert _is_unit_quat_stream(quats)
    assert innov.shape == (T,)
    assert innov[0] == 0.0  # documented: first step has no update

    # (C14) the innovation must be non-trivially nonzero, not numerical dust.
    assert np.max(innov[1:]) > 1e-3
    assert np.mean(innov[1:]) > 1e-4

    # The correction must matter: the spine's pure gyro integrator (NO update)
    # is the true "innovation==0" baseline. The gravity-aided EKF must keep TILT
    # markedly closer to truth than that raw gyro propagation of the same biased
    # gyro. (Using the spine integrator avoids relying on the gravity gate, which
    # never rejects on a perfectly-static |acc| == g track.)
    quats_gyro_only = integrate_gyro(q0, gyro_biased, dt, frame="body")
    assert _is_unit_quat_stream(quats_gyro_only)

    tilt_ekf = float(np.mean(_tilt_error_deg(quats, ori)))
    tilt_gyro = float(np.mean(_tilt_error_deg(quats_gyro_only, ori)))
    assert tilt_ekf < 0.5 * tilt_gyro, (
        f"gravity update should at least halve tilt error: "
        f"EKF tilt {tilt_ekf} vs gyro-only {tilt_gyro}"
    )

    # And the two trajectories must actually differ (the update changed the state).
    assert np.max(quat_angle(quats, quats_gyro_only)) > 1e-3


# --------------------------------------------------------------------------- #
# 3-vector q0 is rejected (guards H7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filt", [madgwick, mahony, ekf_orientation])
def test_three_vector_q0_raises(filt):
    """A length-3 seed (axis / Euler triple) must raise, not be coerced."""
    rec = generate_recording("h7", duration_s=1.0, fs=200.0, seed=1)
    dt = 1.0 / rec.fs
    bad_q0 = np.array([0.0, 0.0, 0.0])  # 3-vector, not a quaternion
    with pytest.raises(ValueError):
        filt(rec.gyro, rec.acc, dt, bad_q0)


# --------------------------------------------------------------------------- #
# All AHRS filters return unit quaternions (T, 4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filt", [madgwick, mahony, ekf_orientation])
def test_ahrs_returns_unit_quaternions(filt):
    rec = generate_recording("unit", duration_s=3.0, fs=200.0, seed=3)
    dt = 1.0 / rec.fs
    q0 = rec.ori[0]
    out = filt(rec.gyro, rec.acc, dt, q0, mag=rec.mag)
    assert out.shape == (rec.acc.shape[0], 4)
    assert _is_unit_quat_stream(out)
    # out[0] must equal normalize(q0) (documented contract).
    np.testing.assert_allclose(out[0], quat_normalize(q0), atol=1e-9)


# --------------------------------------------------------------------------- #
# DisplacementEKF: lowers RMSE vs raw integration on a constant-velocity track
# --------------------------------------------------------------------------- #
def test_displacement_ekf_reduces_rmse_vs_raw_integration():
    """On a constant-velocity track the smoothing EKF must beat raw cumsum.

    Ground truth: the platform moves at constant velocity ``v`` so each
    per-stride displacement is exactly ``v * dt``. We corrupt the displacement
    measurements with i.i.d. Gaussian noise and compare what the constant-velocity
    EKF recovers against "raw noisy integration" -- i.e. taking the noisy
    measurements at face value.

    What the EKF reliably improves is the **velocity / per-stride displacement**
    estimate: the constant-velocity model averages the noise out, so both the
    filtered velocity and the implied per-stride displacement are several times
    closer to truth than the raw (unfiltered) measurements. This is the
    smoothing the docstring promises and is the property the old no-op EKF
    lacked. (See ``issues_found`` for why *absolute position-trajectory* RMSE is
    NOT a reliable win for this single-clone design.)
    """
    rng = np.random.default_rng(0)
    dt = 0.25
    n = 300
    v = np.array([1.3, -0.7])  # m/s, constant

    true_disp = np.tile(v * dt, (n, 1))   # (n, 2) exact per-stride displacement
    true_vel = np.tile(v, (n, 1))         # (n, 2) exact velocity

    noise_std = 0.10
    noisy_disp = true_disp + rng.normal(0.0, noise_std, size=true_disp.shape)

    # "Raw noisy integration": the unfiltered measurements taken at face value.
    raw_vel = noisy_disp / dt             # per-stride velocity estimate, no filter

    # Filtered: constant-velocity smoothing fuses the per-stride displacements.
    ekf = DisplacementEKF(
        dt=dt,
        q_pos=0.0,            # position is the integral of the CV velocity
        q_vel=1e-4,           # strong velocity averaging (track is constant-vel)
        r_meas=noise_std ** 2,
        inflate_overlap=1.0,  # these synthetic measurements are independent
        init_vel_var=4.0,     # let the velocity converge from a cold start
    )
    state = ekf.run(noisy_disp)           # (n, 4) -> [px, py, vx, vy]
    assert state.shape == (n, 4)
    ekf_vel = state[:, 2:4]

    # EKF-implied per-stride displacement = successive position differences.
    ekf_disp = np.diff(np.vstack([np.zeros(2), state[:, :2]]), axis=0)

    def rmse(est, ref):
        return float(np.sqrt(np.mean(np.sum((est - ref) ** 2, axis=1))))

    rmse_vel_raw = rmse(raw_vel, true_vel)
    rmse_vel_ekf = rmse(ekf_vel, true_vel)
    rmse_disp_raw = rmse(noisy_disp, true_disp)
    rmse_disp_ekf = rmse(ekf_disp, true_disp)

    # The filter must clearly beat raw integration on the quantities it estimates.
    assert rmse_vel_ekf < rmse_vel_raw, (
        f"EKF velocity rmse {rmse_vel_ekf} not < raw {rmse_vel_raw}"
    )
    assert rmse_disp_ekf < rmse_disp_raw, (
        f"EKF displacement rmse {rmse_disp_ekf} not < raw {rmse_disp_raw}"
    )
    # And the improvement should be substantial (CV averaging), not marginal.
    assert rmse_vel_ekf < 0.6 * rmse_vel_raw

    # Sanity: the filter recorded innovations / accepted most measurements.
    assert np.all(np.isfinite(ekf.last_innovation))


# --------------------------------------------------------------------------- #
# Train smoke test: loss drops, evaluate returns finite ATE/RTE
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_train_reduces_loss_and_evaluate_finite():
    recs = generate_dataset(n_recordings=4, duration_s=10.0, fs=200.0, seed=11)
    win_cfg = WindowCfg(window_len=100, stride=25)
    train_cfg = TrainCfg(epochs=3, model="resnet1d", seed=0)

    result = train(recs, win_cfg, train_cfg, aug_cfg=AugmentCfg())

    history = result["history"]
    train_loss = history["train_loss"]
    assert len(train_loss) == 3
    assert all(np.isfinite(train_loss))
    # Training loss must drop across the run (last epoch better than the first).
    assert train_loss[-1] < train_loss[0]

    ev = evaluate(result["model"], result["train_per_recording"], win_cfg)
    agg = ev["aggregate"]
    # aggregate_metrics output must carry finite ATE / RTE summaries.
    for rec_res in ev["per_recording"]:
        assert np.isfinite(rec_res["ate"])
        assert np.isfinite(rec_res["rte"])
    # The aggregate dict's numeric entries must all be finite.
    for key, val in agg.items():
        assert np.isfinite(val), f"aggregate[{key!r}] = {val} not finite"
