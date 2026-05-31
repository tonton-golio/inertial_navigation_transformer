"""Tests for ``ninav.geometry`` — the attitude/frame correctness spine.

These guard the specific bugs called out in the spine docstrings:

* **C1** — ``theta_matrix`` rate/normalisation bug (old ``Theta`` rotated at the
  wrong rate and was not norm-preserving). Here we check orthonormality and that
  constant-rate integration recovers the *correct total angle*.
* **C2** — ``body_to_world_ned`` vertical inversion (old code set
  ``DOWN = +acc``). A flat resting phone measures specific force pointing up, so
  the DOWN column must be ``~ -acc_dir``.
* **C3** — scalar-first/scalar-last convention mismatch. The quat<->rotmat
  round-trip is checked against scipy (scalar-last) with an explicit index check
  that ``W == 3`` and that ``as_quat()[3]`` is the real part.
* **H1** — gyro propagation rate. ``integrate_gyro`` constant-rate must match
  scipy's exact rotation and stay unit-norm.
* **H2** — ``rotmat_to_quat`` near-180deg blow-up (the old ``1/(4*q0)`` form).
  Shepperd's method must stay finite and unit there.
* **H12** — ``gravity_align_rotation`` must map the measured acceleration
  direction onto ``+Z``.

The convention is scalar-last ``[x, y, z, w]``, matching
``scipy.spatial.transform.Rotation`` defaults, with ``quat_to_rotmat(q) @ v``
mapping a body-frame vector into the world frame.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ninav.geometry import frames, propagate, quaternion


# --------------------------------------------------------------------------- #
# Convention / index sanity (guards C3)
# --------------------------------------------------------------------------- #
def test_scalar_last_index_constants():
    """The real part lives at index 3 and IDENTITY is [0,0,0,1]."""
    assert quaternion.W == 3
    np.testing.assert_array_equal(quaternion.IDENTITY, np.array([0.0, 0.0, 0.0, 1.0]))


def test_scipy_scalar_last_index_check():
    """scipy's default as_quat() is scalar-last: a pure-Z rotation puts the
    sin(half) on index 2 and cos(half) on the real index 3, with x=y=0."""
    angle = 0.9
    q = Rotation.from_rotvec([0.0, 0.0, angle]).as_quat()  # scipy default = scalar-last
    assert q[quaternion.W] == pytest.approx(np.cos(angle / 2.0))
    assert q[2] == pytest.approx(np.sin(angle / 2.0))
    assert q[0] == pytest.approx(0.0, abs=1e-12)
    assert q[1] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# quat <-> rotmat round-trip vs scipy (guards C3)
# --------------------------------------------------------------------------- #
def _random_unit_quats(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    # Canonicalise sign so the double-cover does not confuse comparisons.
    q[q[:, quaternion.W] < 0] *= -1.0
    return q


def test_quat_to_rotmat_matches_scipy():
    """quat_to_rotmat must equal scipy's matrix for the same scalar-last quat."""
    quats = _random_unit_quats(50, seed=0)
    for q in quats:
        R_ours = quaternion.quat_to_rotmat(q)
        R_scipy = Rotation.from_quat(q).as_matrix()  # scalar-last input
        np.testing.assert_allclose(R_ours, R_scipy, atol=1e-12)


def test_rotmat_active_body_to_world():
    """R(q) @ v_body == scipy active rotation of v (sanity on the active sense)."""
    rng = np.random.default_rng(7)
    for q in _random_unit_quats(20, seed=3):
        R = quaternion.quat_to_rotmat(q)
        v = rng.standard_normal(3)
        np.testing.assert_allclose(R @ v, Rotation.from_quat(q).apply(v), atol=1e-12)


def test_quat_rotmat_quat_round_trip():
    """rotmat_to_quat(quat_to_rotmat(q)) recovers q (up to sign)."""
    for q in _random_unit_quats(50, seed=1):
        R = quaternion.quat_to_rotmat(q)
        q_back = quaternion.rotmat_to_quat(R)
        # double-cover: q and -q are the same rotation -> compare via angle.
        # quat_angle uses 2*arccos(|dot|), whose derivative diverges as dot->1,
        # so float64 round-trip error (~1e-15 in the components) shows up as
        # ~3e-8 rad here -- 1e-6 is a sound geodesic tolerance, not a bug.
        ang = quaternion.quat_angle(q, q_back)
        assert ang == pytest.approx(0.0, abs=1e-6)
        # and it must be unit norm
        assert np.linalg.norm(q_back) == pytest.approx(1.0, abs=1e-12)


def test_rotmat_to_quat_matches_scipy_indexwise():
    """rotmat_to_quat agrees with scipy's scalar-last quaternion (sign-aligned)."""
    for R in (Rotation.from_quat(q).as_matrix() for q in _random_unit_quats(40, seed=11)):
        q_ours = quaternion.rotmat_to_quat(R)
        q_scipy = Rotation.from_matrix(R).as_quat()  # scalar-last
        if np.dot(q_ours, q_scipy) < 0:
            q_scipy = -q_scipy
        np.testing.assert_allclose(q_ours, q_scipy, atol=1e-9)


# --------------------------------------------------------------------------- #
# theta_matrix orthonormality + identity at zero (guards C1)
# --------------------------------------------------------------------------- #
def test_theta_matrix_orthonormal():
    """theta_matrix is orthonormal (T @ T.T == I) with det == +1."""
    dt = 1.0 / 200.0
    rng = np.random.default_rng(5)
    for _ in range(20):
        w = rng.standard_normal(3) * rng.uniform(0.0, 20.0)
        T = propagate.theta_matrix(w, dt)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T @ T.T, np.eye(4), atol=1e-12)
        assert np.linalg.det(T) == pytest.approx(1.0, abs=1e-10)


def test_theta_matrix_zero_is_identity_no_nan():
    """theta_matrix(0) == I_4 and contains no NaN (old sin(0)/0 blew up)."""
    T = propagate.theta_matrix(np.zeros(3), 1.0 / 200.0)
    assert np.all(np.isfinite(T))
    np.testing.assert_array_equal(T, np.eye(4))


def test_theta_matrix_applies_correct_increment():
    """theta_matrix(w,dt) @ q must equal q (x) delta_quat(w,dt) (no half-rate bug)."""
    dt = 1.0 / 200.0
    w = np.array([0.3, -1.2, 0.7])
    q0 = quaternion.quat_normalize(np.array([0.1, 0.2, -0.3, 1.0]))
    # theta_matrix is the *left*-multiplication matrix of the increment quaternion,
    # i.e. theta_matrix @ q == delta_quat (x) q.
    expected = quaternion.quat_multiply(propagate.delta_quat(w, dt), q0)
    got = propagate.theta_matrix(w, dt) @ q0
    np.testing.assert_allclose(got, expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# integrate_gyro constant-rate (guards C1 + H1)
# --------------------------------------------------------------------------- #
def test_integrate_gyro_constant_rate_total_angle_and_unit():
    """Constant body-rate about a fixed axis recovers the correct TOTAL angle.

    integrate_gyro sets out[0] = normalize(q0) (no increment at step 0) and then
    applies T-1 increments, so the accumulated angle after T samples is
    (T-1)*|w|*dt. Verify the total angle, unit-norm at every step, and a match
    against scipy's exact rotation.
    """
    fs = 200.0
    dt = 1.0 / fs
    rate = 1.5  # rad/s
    axis = np.array([0.0, 0.0, 1.0])  # body-Z (single fixed axis)
    T = 201
    gyro = np.tile(axis * rate, (T, 1))
    q0 = quaternion.IDENTITY.copy()

    quats = propagate.integrate_gyro(q0, gyro, dt, frame="body")
    assert quats.shape == (T, 4)

    # unit norm at every step
    norms = np.linalg.norm(quats, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    # out[0] is exactly normalize(q0)
    np.testing.assert_allclose(quats[0], q0, atol=1e-12)

    # TOTAL accumulated angle after T samples = (T-1) increments.
    total_angle = (T - 1) * rate * dt
    ang = quaternion.quat_angle(q0, quats[-1])
    assert ang == pytest.approx(total_angle, abs=1e-9)

    # exact scipy reference (single fixed axis -> body == world, commutes)
    R_ref = Rotation.from_rotvec(axis * total_angle)
    q_ref = R_ref.as_quat()  # scalar-last
    if np.dot(q_ref, quats[-1]) < 0:
        q_ref = -q_ref
    np.testing.assert_allclose(quats[-1], q_ref, atol=1e-9)


def test_integrate_gyro_off_axis_matches_scipy():
    """A constant off-cardinal-axis body-rate still matches scipy exactly.

    For a single fixed axis the body- and world-frame increments commute, so the
    total rotation is the exact axis-angle rotation by (T-1)*|w|*dt.
    """
    dt = 1.0 / 200.0
    axis = np.array([1.0, -2.0, 0.5])
    axis = axis / np.linalg.norm(axis)
    rate = 2.3
    T = 64
    gyro = np.tile(axis * rate, (T, 1))
    q0 = quaternion.quat_normalize(np.array([0.0, 0.0, 0.0, 1.0]))

    quats = propagate.integrate_gyro(q0, gyro, dt, frame="body")
    np.testing.assert_allclose(np.linalg.norm(quats, axis=-1), 1.0, atol=1e-12)

    total_angle = (T - 1) * rate * dt
    q_ref = Rotation.from_rotvec(axis * total_angle).as_quat()
    if np.dot(q_ref, quats[-1]) < 0:
        q_ref = -q_ref
    np.testing.assert_allclose(quats[-1], q_ref, atol=1e-9)


# --------------------------------------------------------------------------- #
# body_to_world_ned static flat-phone (guards C2)
# --------------------------------------------------------------------------- #
def test_body_to_world_ned_flat_phone_down_column():
    """A flat resting phone: acc measures specific force UP (+g) along body Z.

    The DOWN column of the NED rotation must point opposite the accel direction
    (down ~ -acc_dir), the matrix must be a proper rotation (det = +1), and the
    columns must be orthonormal.
    """
    g = frames.GRAVITY
    acc = np.array([0.0, 0.0, g])          # specific force points up at rest
    mag = np.array([0.5, 0.0, -0.866])     # arbitrary horizontal-ish field
    R = frames.body_to_world_ned(mag, acc)

    assert R.shape == (3, 3)

    acc_dir = acc / np.linalg.norm(acc)
    down_col = R[:, 2]
    # DOWN column ~ -acc_dir (this is the C2 vertical-inversion guard)
    np.testing.assert_allclose(down_col, -acc_dir, atol=1e-12)

    # proper rotation: det == +1 and orthonormal columns
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_body_to_world_ned_right_handed_triad():
    """Columns N, E, DOWN form a right-handed triad: N x E == DOWN."""
    acc = np.array([0.1, -0.05, 9.8])
    mag = np.array([0.4, 0.2, -0.3])
    R = frames.body_to_world_ned(mag, acc)
    north, east, down = R[:, 0], R[:, 1], R[:, 2]
    np.testing.assert_allclose(np.cross(north, east), down, atol=1e-12)


# --------------------------------------------------------------------------- #
# rotmat_to_quat near-180deg finite + unit (guards H2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axis", [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
])
def test_rotmat_to_quat_near_180_finite_unit(axis):
    """Near a 180deg rotation (negative-trace regime) the result must stay finite
    and unit-norm, and match scipy. This is where the old 1/(4*q0) form blew up."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    for angle in (np.pi - 1e-7, np.pi, np.pi - 1e-4):
        R = Rotation.from_rotvec(axis * angle).as_matrix()
        q = quaternion.rotmat_to_quat(R)
        assert np.all(np.isfinite(q)), f"non-finite quat at angle {angle}"
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)
        # same rotation as scipy (compare via geodesic angle, double-cover safe)
        q_scipy = Rotation.from_matrix(R).as_quat()
        assert quaternion.quat_angle(q, q_scipy) == pytest.approx(0.0, abs=1e-6)


def test_rotmat_to_quat_exact_180_each_pivot():
    """Exact 180deg about each principal axis hits a different Shepperd pivot and
    must still round-trip back to the same rotation matrix, finitely."""
    for axis in (np.eye(3)):
        R = Rotation.from_rotvec(axis * np.pi).as_matrix()
        q = quaternion.rotmat_to_quat(R)
        assert np.all(np.isfinite(q))
        R_back = quaternion.quat_to_rotmat(q)
        np.testing.assert_allclose(R_back, R, atol=1e-9)


# --------------------------------------------------------------------------- #
# gravity_align_rotation maps acc -> +Z (guards H12)
# --------------------------------------------------------------------------- #
def test_gravity_align_rotation_maps_acc_to_plus_z():
    """R @ acc_dir == +Z for a generic resting orientation."""
    rng = np.random.default_rng(13)
    for _ in range(20):
        acc = rng.standard_normal(3)
        if np.linalg.norm(acc) < 1e-6:
            continue
        R = frames.gravity_align_rotation(acc)
        acc_dir = acc / np.linalg.norm(acc)
        np.testing.assert_allclose(R @ acc_dir, [0.0, 0.0, 1.0], atol=1e-12)
        # proper rotation
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_gravity_align_rotation_already_aligned():
    """acc already along +Z -> identity; along -Z -> still maps to +Z, finite."""
    R_up = frames.gravity_align_rotation(np.array([0.0, 0.0, 5.0]))
    np.testing.assert_allclose(R_up @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12)
    assert np.all(np.isfinite(R_up))

    R_down = frames.gravity_align_rotation(np.array([0.0, 0.0, -5.0]))
    np.testing.assert_allclose(R_down @ np.array([0.0, 0.0, -1.0]), [0.0, 0.0, 1.0], atol=1e-12)
    assert np.all(np.isfinite(R_down))
    assert np.linalg.det(R_down) == pytest.approx(1.0, abs=1e-10)
