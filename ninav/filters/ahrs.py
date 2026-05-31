"""From-scratch attitude (AHRS) filters: Madgwick, Mahony, and a quaternion EKF.

All three filters estimate a sequence of **body->world** orientations from one
recording's IMU stream and follow the repo-wide conventions enforced by
:mod:`ninav.geometry`:

    * quaternions are **scalar-last** ``q = [x, y, z, w]``;
    * ``q`` is a *body->world* rotation: ``quat_to_rotmat(q) @ v_body == v_world``;
    * the accelerometer reports **specific force** in the body frame, which at
      rest points *up* (``+g``). With the world gravity-acceleration vector
      ``g = [0, 0, -G]`` (see :mod:`ninav.data.synthetic`), the world specific
      force at rest is ``+Z`` and the body-frame measurement direction is the
      third row of the rotation matrix, i.e. ``R(q).T @ [0, 0, 1]``.

There is no dependency on the external ``ahrs`` package -- everything is plain
numpy built on :mod:`ninav.geometry.quaternion`.

These filters operate on a **single recording's** stream. Splitting across
train/val/test recordings is the caller's job; never concatenate recordings and
run one filter over the join (the orientation state would jump at the seam).

EKF note (the bug this replaces)
--------------------------------
An earlier EKF set the accelerometer *measurement* equal to its own
*prediction*, so the innovation ``y = z - h(x)`` was identically zero and the
filter never corrected -- it was a dressed-up gyro integrator. Here the
measurement is the actual (normalized) accelerometer sample and the predicted
measurement is ``R(q_pred).T @ zhat``; the innovation is genuinely nonzero
whenever the gyro-propagated tilt disagrees with what the accelerometer sees,
so the gravity update really pulls the estimate back. :func:`ekf_orientation`
also returns the per-step innovation norms (via ``return_innovations``) so a
test can assert the correction is non-trivial.
"""
from __future__ import annotations

import numpy as np

from ..geometry.quaternion import (
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
)
from ..geometry.propagate import delta_quat

_EPS = 1e-12
GRAVITY = 9.80665  # m/s^2, matches ninav.geometry.frames.GRAVITY


# --------------------------------------------------------------------------- #
# Shared input validation / helpers
# --------------------------------------------------------------------------- #
def _check_q0(q0: np.ndarray) -> np.ndarray:
    """Validate and normalize an initial orientation seed.

    ``q0`` MUST be a length-4 unit quaternion (scalar-last). A 3-vector seed
    (a common mistake -- passing an axis or Euler triple) is rejected with a
    clear error rather than being silently coerced.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    if q0.shape != (4,):
        raise ValueError(
            "q0 must be a length-4 unit quaternion (scalar-last [x, y, z, w]); "
            f"got shape {q0.shape}. A 3-vector (axis / Euler triple) is not a "
            "valid orientation seed."
        )
    if np.linalg.norm(q0) < _EPS:
        raise ValueError("q0 must be a non-zero (unit) quaternion, got ~zero norm.")
    return quat_normalize(q0)


def _check_stream(gyro: np.ndarray, acc: np.ndarray, mag: np.ndarray | None):
    """Coerce + shape-check the per-recording IMU stream. Returns (gyro, acc, mag, T)."""
    gyro = np.asarray(gyro, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError(f"gyro must be (T, 3), got {gyro.shape}")
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"acc must be (T, 3), got {acc.shape}")
    if gyro.shape[0] != acc.shape[0]:
        raise ValueError(
            f"gyro and acc must have the same length, got {gyro.shape[0]} vs {acc.shape[0]}"
        )
    if mag is not None:
        mag = np.asarray(mag, dtype=np.float64)
        if mag.ndim != 2 or mag.shape[1] != 3:
            raise ValueError(f"mag must be (T, 3) or None, got {mag.shape}")
        if mag.shape[0] != gyro.shape[0]:
            raise ValueError("mag must have the same length as gyro/acc")
    return gyro, acc, mag, gyro.shape[0]


def _safe_unit(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    if n < _EPS:
        return None
    return v / n


# World-frame reference directions (this repo's ENU-like, Z-up world; see
# ninav.data.synthetic). Gravity *specific force* at rest points up: +Z.
_UP_WORLD = np.array([0.0, 0.0, 1.0])


def _gravity_trust(acc_norm: float, accel_reject: float) -> float:
    """Down-weight the gravity correction when the accelerometer is dynamic.

    The accelerometer only equals gravity *at rest*; during motion it also
    contains linear acceleration, so it is a poor gravity reference. We compute a
    trust factor in ``[0, 1]`` that is 1 when ``|acc| == g`` and decays linearly
    to 0 once ``| |acc| - g | / g`` reaches ``accel_reject`` (a fractional band).
    ``accel_reject <= 0`` disables the gate (always full trust). This is the
    standard "gravity gate" that keeps gravity-aided AHRS from being dragged off
    by transient linear acceleration.
    """
    if accel_reject <= 0.0:
        return 1.0
    rel = abs(acc_norm - GRAVITY) / GRAVITY
    return float(np.clip(1.0 - rel / accel_reject, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Madgwick
# --------------------------------------------------------------------------- #
def madgwick(
    gyro: np.ndarray,
    acc: np.ndarray,
    dt: float,
    q0: np.ndarray,
    mag: np.ndarray | None = None,
    beta: float = 0.1,
    accel_reject: float = 0.15,
) -> np.ndarray:
    """Madgwick complementary filter (gradient-descent gravity correction).

    Parameters
    ----------
    gyro : (T, 3) body angular velocity (rad/s).
    acc  : (T, 3) body specific force (m/s^2); used only by direction.
    dt   : sample period (s).
    q0   : (4,) unit quaternion, scalar-last (body->world). A 3-vector is rejected.
    mag  : optional (T, 3); currently unused beyond shape-checking (IMU-only fusion
           -- this repo's synthetic world has no observable heading reference in
           the accelerometer, and the heading is intentionally left free).
    beta : gradient-descent gain (rad/s); larger trusts the accelerometer more.
    accel_reject : fractional gravity-gate band (see :func:`_gravity_trust`);
           the gravity correction is faded out as ``|acc|`` leaves the gravity
           band during motion. Set to 0 to always trust the accelerometer.

    Returns
    -------
    (T, 4) unit quaternions, ``out[0] == normalize(q0)``.

    Implementation notes
    --------------------
    The classic Madgwick paper uses a scalar-FIRST, world->body convention. Here
    everything is scalar-LAST and ``q`` is body->world, so we build the
    gravity-error objective directly from this repo's geometry:
    the predicted body-frame gravity direction is ``R(q).T @ +Z`` (= third row of
    ``R``), the measurement is the normalized accelerometer, and we descend the
    objective ``f = R(q).T @ zhat - acc_hat`` via its analytic Jacobian. The
    correction is applied to the gyro-derived quaternion rate.
    """
    gyro, acc, mag, T = _check_stream(gyro, acc, mag)
    q = _check_q0(q0)
    out = np.empty((T, 4), dtype=np.float64)
    out[0] = q

    for k in range(1, T):
        w = gyro[k]
        # Gyro-only quaternion rate: q_dot = 0.5 * q (x) [w, 0]   (body rate, right-mult).
        w_quat = np.array([w[0], w[1], w[2], 0.0])
        q_dot = 0.5 * quat_multiply(q, w_quat)

        acc_hat = _safe_unit(acc[k])
        if acc_hat is not None:
            trust = _gravity_trust(float(np.linalg.norm(acc[k])), accel_reject)
            if trust > 0.0:
                grad = _gravity_gradient(q, acc_hat)
                gn = np.linalg.norm(grad)
                if gn > _EPS:
                    q_dot = q_dot - (beta * trust) * (grad / gn)

        q = quat_normalize(q + q_dot * dt)
        out[k] = q
    return out


def _gravity_gradient(q: np.ndarray, acc_hat: np.ndarray) -> np.ndarray:
    """Gradient (wrt scalar-last q) of 0.5*||R(q).T @ +Z - acc_hat||^2.

    Predicted body gravity direction g_b(q) = R(q).T @ [0,0,1] = third ROW of R(q):
        gx = 2*(x*z - w*y)
        gy = 2*(y*z + w*x)
        gz = 1 - 2*(x^2 + y^2)
    f = g_b - acc_hat. Returns J^T @ f, where J = d g_b / d q (3x4), q=[x,y,z,w].
    """
    x, y, z, w = q
    f = np.array([
        2.0 * (x * z - w * y) - acc_hat[0],
        2.0 * (y * z + w * x) - acc_hat[1],
        (1.0 - 2.0 * (x * x + y * y)) - acc_hat[2],
    ])
    # Jacobian rows are d(gx,gy,gz)/d(x,y,z,w).
    J = np.array([
        [2 * z, -2 * w, 2 * x, -2 * y],   # d gx
        [2 * w, 2 * z, 2 * y, 2 * x],     # d gy
        [-4 * x, -4 * y, 0.0, 0.0],       # d gz
    ])
    return J.T @ f


# --------------------------------------------------------------------------- #
# Mahony
# --------------------------------------------------------------------------- #
def mahony(
    gyro: np.ndarray,
    acc: np.ndarray,
    dt: float,
    q0: np.ndarray,
    mag: np.ndarray | None = None,
    kp: float = 1.0,
    ki: float = 0.0,
    accel_reject: float = 0.15,
) -> np.ndarray:
    """Mahony explicit-complementary filter (PI feedback on the gravity error).

    Parameters
    ----------
    gyro, acc, dt : as in :func:`madgwick`.
    q0 : (4,) unit quaternion, scalar-last. A 3-vector is rejected.
    mag : optional (T, 3); shape-checked only (IMU-only fusion here).
    kp : proportional gain on the gravity-direction error.
    ki : integral gain (accumulates a gyro-bias estimate); 0 disables it.
    accel_reject : fractional gravity-gate band (see :func:`_gravity_trust`);
        fades the PI correction out during dynamic acceleration. 0 disables it.

    Returns
    -------
    (T, 4) unit quaternions, ``out[0] == normalize(q0)``.

    The error term is the cross product (in the BODY frame) between the measured
    gravity direction (normalized accelerometer) and the predicted one
    ``R(q).T @ +Z``; this vector is a small-angle tilt correction that is fed back
    onto the gyro rate (P term) and integrated into a bias estimate (I term).
    """
    gyro, acc, mag, T = _check_stream(gyro, acc, mag)
    q = _check_q0(q0)
    out = np.empty((T, 4), dtype=np.float64)
    out[0] = q
    bias = np.zeros(3)

    for k in range(1, T):
        w = gyro[k].copy()
        acc_hat = _safe_unit(acc[k])
        trust = _gravity_trust(float(np.linalg.norm(acc[k])), accel_reject) if acc_hat is not None else 0.0
        if acc_hat is not None and trust > 0.0:
            R = quat_to_rotmat(q)
            v_hat = R.T @ _UP_WORLD          # predicted body gravity direction
            # error: measured (x) predicted, in body frame (small-angle tilt vector)
            e = trust * np.cross(acc_hat, v_hat)
            if ki > 0.0:
                bias = bias + ki * e * dt    # integral term -> gyro-bias estimate
            w = w + bias + kp * e            # PI-corrected rate
        else:
            w = w + bias                     # still apply the learned bias estimate

        # Propagate with the corrected body rate (right-multiply, body frame).
        dq = delta_quat(w, dt)
        q = quat_normalize(quat_multiply(q, dq))
        out[k] = q
    return out


# --------------------------------------------------------------------------- #
# Quaternion EKF (with a real, nonzero accelerometer innovation)
# --------------------------------------------------------------------------- #
def ekf_orientation(
    gyro: np.ndarray,
    acc: np.ndarray,
    dt: float,
    q0: np.ndarray,
    mag: np.ndarray | None = None,
    sigma_gyro: float = 0.01,
    sigma_acc: float = 0.5,
    p0: float = 1e-3,
    accel_reject: float = 0.15,
    return_innovations: bool = False,
):
    """Multiplicative-style quaternion EKF with a gravity (accelerometer) update.

    State is the 4-vector quaternion ``q`` (body->world, scalar-last) with a 4x4
    covariance. Each step:

      1. **Predict** with the gyro: ``q_pred = q (x) dq(w*dt)`` and propagate the
         covariance through the linearized transition ``F``, adding process noise.
      2. **Update** with the accelerometer: the measurement is the *normalized*
         accelerometer sample ``z = acc/|acc|``; the predicted measurement is
         ``h(q_pred) = R(q_pred).T @ +Z`` (body gravity direction). The
         **innovation** ``y = z - h(q_pred)`` is genuinely nonzero whenever the
         propagated tilt disagrees with the accelerometer, and the Kalman gain
         corrects the quaternion accordingly. (This is the fix for the old EKF
         that set ``z = h`` and thus never corrected.)

    Parameters
    ----------
    gyro, acc, dt : as in :func:`madgwick`.
    q0 : (4,) unit quaternion, scalar-last. A 3-vector is rejected.
    mag : optional (T, 3); shape-checked only (gravity-only update here).
    sigma_gyro : gyro process-noise std (rad/s) -> Q.
    sigma_acc : accelerometer measurement-noise std (on the unit-direction) -> R.
    p0 : initial state covariance scale.
    accel_reject : fractional gravity-gate band (see :func:`_gravity_trust`);
        the measurement-noise covariance ``R`` is inflated (the update is
        down-weighted) as ``|acc|`` leaves the gravity band. 0 disables the gate.
        The innovation ``y`` is still *computed and reported* every step.
    return_innovations : if True, also return the (T,) per-step innovation norms
        (``out[0] == 0``); used by tests to assert a non-trivial state update.

    Returns
    -------
    (T, 4) unit quaternions, or ``(quats, innovation_norms)`` if
    ``return_innovations`` is True.
    """
    gyro, acc, mag, T = _check_stream(gyro, acc, mag)
    q = _check_q0(q0)

    P = np.eye(4) * p0
    Q = np.eye(4) * (sigma_gyro * dt) ** 2
    Rm = np.eye(3) * (sigma_acc ** 2)

    out = np.empty((T, 4), dtype=np.float64)
    out[0] = q
    innov = np.zeros(T, dtype=np.float64)

    for k in range(1, T):
        # ---- Predict (gyro) ----
        dq = delta_quat(gyro[k], dt)
        # q_pred = q (x) dq  ==  R_dq @ q  (right-multiply is a linear map on q).
        F = _right_mult_matrix(dq)
        q_pred = F @ q
        P = F @ P @ F.T + Q
        q_pred = quat_normalize(q_pred)

        # ---- Update (accelerometer gravity direction) ----
        acc_hat = _safe_unit(acc[k])
        if acc_hat is not None:
            h = _predicted_gravity_dir(q_pred)     # R(q_pred).T @ +Z
            y = acc_hat - h                        # innovation (nonzero in general)
            innov[k] = float(np.linalg.norm(y))    # reported regardless of gating
            H = _gravity_dir_jacobian(q_pred)      # 3x4
            # Gravity gate: inflate R (down-weight the update) during motion.
            trust = _gravity_trust(float(np.linalg.norm(acc[k])), accel_reject)
            if trust > 0.0:
                R_eff = Rm / trust                 # trust=1 -> Rm; trust->0 -> huge R
                S = H @ P @ H.T + R_eff
                K = P @ H.T @ np.linalg.inv(S)     # 4x3 Kalman gain
                q_upd = q_pred + K @ y
                P = (np.eye(4) - K @ H) @ P
                q = quat_normalize(q_upd)
            else:
                q = q_pred                         # fully rejected -> predict only
        else:
            q = q_pred

        out[k] = q

    if return_innovations:
        return out, innov
    return out


def _right_mult_matrix(r: np.ndarray) -> np.ndarray:
    """4x4 matrix M such that ``M @ q == quat_multiply(q, r)`` (scalar-last).

    Right-multiplication by a fixed quaternion is linear in ``q``; this is its
    matrix form, used as the EKF state-transition ``F``.
    """
    rx, ry, rz, rw = r
    return np.array([
        [rw, rz, -ry, rx],
        [-rz, rw, rx, ry],
        [ry, -rx, rw, rz],
        [-rx, -ry, -rz, rw],
    ])


def _predicted_gravity_dir(q: np.ndarray) -> np.ndarray:
    """Predicted body-frame gravity (specific-force) direction = R(q).T @ +Z.

    Equals the third ROW of R(q): [2(xz - wy), 2(yz + wx), 1 - 2(x^2 + y^2)].
    """
    x, y, z, w = q
    return np.array([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])


def _gravity_dir_jacobian(q: np.ndarray) -> np.ndarray:
    """Jacobian (3x4) of :func:`_predicted_gravity_dir` wrt q = [x, y, z, w]."""
    x, y, z, w = q
    return np.array([
        [2 * z, -2 * w, 2 * x, -2 * y],
        [2 * w, 2 * z, 2 * y, 2 * x],
        [-4 * x, -4 * y, 0.0, 0.0],
    ])
