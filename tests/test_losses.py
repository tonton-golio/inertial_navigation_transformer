"""Tests for ninav.losses (regression + rotation).

Guards two specific regressions called out in the rotation-loss docstring:
    * bug C10 -- the old phi2 loss collapsed the whole batch to one scalar and
      chose a SINGLE global q/-q sign, so a batch mixing samples that need +q
      with samples that need -q could not be driven to zero. The fixed losses
      reduce per-sample (dim=-1), so each sample picks its own sign.
    * bug H6 -- ``arccos(|cos_sim|)`` has an INFINITE gradient at the optimum
      (cos == 1). The recommended training surrogates are smooth there, and the
      geodesic loss clamps ``|cos|`` so its gradient stays finite even when
      ``pred == target``.

All tests are CPU-only, tiny tensors, double precision where exactness matters.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ninav.geometry.quaternion import quat_from_axis_angle
from ninav.losses.regression import gaussian_nll_loss, mse_loss
from ninav.losses.rotation import (
    geodesic_angle_deg,
    quat_chordal_loss,
    quat_cosine_loss,
    quat_geodesic_loss,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _rand_unit_quats(n: int, seed: int = 0) -> torch.Tensor:
    """A batch of (n, 4) random unit quaternions (scalar-last), float64."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return torch.from_numpy(q)  # float64


# --------------------------------------------------------------------------
# (1) DOUBLE-COVER per sample -- guards bug C10
# --------------------------------------------------------------------------
def test_double_cover_per_sample_cosine_and_chordal_are_zero():
    """Half the batch needs +q, half needs -q to match target.

    A correct per-sample double-cover loss is ~0 for the WHOLE batch. A loss
    that picks one global sign (the old phi2 / bug C10) cannot do this: it
    would be near-zero only for the half it agreed with and ~maximal for the
    other half, so the batch mean would be far from zero.
    """
    n = 8
    target = _rand_unit_quats(n, seed=1)
    pred = target.clone()
    # Flip the sign of the second half of the predictions: q and -q are the
    # same rotation, so geometrically pred still equals target everywhere.
    pred[n // 2:] = -pred[n // 2:]

    cos = quat_cosine_loss(pred, target)
    cho = quat_chordal_loss(pred, target)

    assert torch.isfinite(cos) and torch.isfinite(cho)
    assert cos.item() < 1e-10, f"cosine loss not ~0 across mixed signs: {cos.item()}"
    assert cho.item() < 1e-10, f"chordal loss not ~0 across mixed signs: {cho.item()}"


def test_double_cover_sign_flip_is_symmetric():
    """Flipping the sign of EVERY prediction must not change either loss."""
    target = _rand_unit_quats(6, seed=2)
    pred = _rand_unit_quats(6, seed=3)

    cos_a = quat_cosine_loss(pred, target)
    cos_b = quat_cosine_loss(-pred, target)
    cho_a = quat_chordal_loss(pred, target)
    cho_b = quat_chordal_loss(-pred, target)

    assert torch.allclose(cos_a, cos_b, atol=1e-12)
    assert torch.allclose(cho_a, cho_b, atol=1e-12)


def test_a_whole_batch_single_sign_loss_would_fail_this_case():
    """Sanity anchor for the C10 guard.

    Demonstrate that the buggy formulation (one global sign chosen by the SUM
    of dot products, then a single squared-distance term) is NOT ~0 on the
    mixed-sign batch the per-sample loss handles. This is a *characterization*
    of the bug, not a call into the spine -- it proves the test above has teeth.
    """
    n = 8
    target = _rand_unit_quats(n, seed=1)
    pred = target.clone()
    pred[n // 2:] = -pred[n // 2:]

    # Buggy global-sign chordal: pick one sign for the whole batch by the
    # aggregate dot, then a single (p-t)^2 summed over everything.
    total_dot = (pred * target).sum()
    sign = torch.sign(total_dot)
    buggy = ((pred * sign - target) ** 2).sum()
    # With an even split the aggregate dot is ~0 and whichever sign is chosen
    # leaves half the batch maximally wrong -> far from zero.
    assert buggy.item() > 1.0, (
        "characterization of the buggy global-sign loss should be large; "
        f"got {buggy.item()}"
    )


# --------------------------------------------------------------------------
# (2) FINITE gradient when pred == target -- guards bug H6
# --------------------------------------------------------------------------
def test_geodesic_loss_finite_gradient_at_optimum():
    """quat_geodesic_loss must have a finite gradient at pred == target.

    The naive ``arccos(|cos|)`` blows up to an infinite derivative at cos == 1
    (bug H6). The spine clamps ``|cos|`` to ``1 - eps``, so backprop through the
    identity case must yield finite (non-NaN, non-inf) grads.
    """
    target = _rand_unit_quats(5, seed=4)
    pred = target.clone().requires_grad_(True)

    loss = quat_geodesic_loss(pred, target)
    assert torch.isfinite(loss), f"geodesic loss not finite at optimum: {loss}"

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), (
        f"geodesic-loss grad not finite at pred==target: {pred.grad}")


def test_smooth_surrogates_finite_gradient_at_optimum():
    """The recommended training surrogates are smooth at pred == target too."""
    for loss_fn in (quat_cosine_loss, quat_chordal_loss):
        target = _rand_unit_quats(5, seed=5)
        pred = target.clone().requires_grad_(True)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all(), (
            f"{loss_fn.__name__} grad not finite at optimum: {pred.grad}")


def test_geodesic_loss_zero_at_optimum_and_positive_off_optimum():
    """Loss is ~0 (within the clamp floor) at the optimum and larger away."""
    target = _rand_unit_quats(4, seed=6)
    at_opt = quat_geodesic_loss(target.clone(), target)

    # Rotate every target by a fixed extra rotation -> strictly larger loss.
    extra = torch.from_numpy(quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.5))
    # Hamilton product (scalar-last) applied per row.
    def _qmul(p, q):
        px, py, pz, pw = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        x = pw * qx + px * qw + py * qz - pz * qy
        y = pw * qy - px * qz + py * qw + pz * qx
        z = pw * qz + px * qy - py * qx + pz * qw
        w = pw * qw - px * qx - py * qy - pz * qz
        return torch.stack([x, y, z, w], dim=-1)

    rotated = _qmul(target, extra)
    off_opt = quat_geodesic_loss(rotated, target)
    assert off_opt.item() > at_opt.item()


# --------------------------------------------------------------------------
# (3) geodesic_angle_deg endpoints
# --------------------------------------------------------------------------
def test_geodesic_angle_deg_identical_is_zero():
    q = _rand_unit_quats(7, seed=7)
    ang = geodesic_angle_deg(q, q.clone())
    assert ang.shape == (7,)
    assert torch.allclose(ang, torch.zeros_like(ang), atol=1e-6), ang


def test_geodesic_angle_deg_antipodal_rotation_is_180():
    """A 180-degree relative rotation yields a geodesic angle of ~180 deg.

    NOTE: q and -q are the SAME rotation (0 deg), so 'antipodal' here means a
    genuine 180-deg rotation of the target about some axis, not a sign flip.
    """
    target = _rand_unit_quats(5, seed=8)
    # 180-deg rotation about x as the relative rotation: pred = target (x) r.
    r = torch.from_numpy(quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi))

    def _qmul(p, q):
        px, py, pz, pw = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        x = pw * qx + px * qw + py * qz - pz * qy
        y = pw * qy - px * qz + py * qw + pz * qx
        z = pw * qz + px * qy - py * qx + pz * qw
        w = pw * qw - px * qx - py * qy - pz * qz
        return torch.stack([x, y, z, w], dim=-1)

    pred = _qmul(target, r)
    ang = geodesic_angle_deg(pred, target)
    assert torch.allclose(ang, torch.full_like(ang, 180.0), atol=1e-4), ang


def test_geodesic_angle_deg_sign_flip_is_zero():
    """A pure sign flip is the same rotation -> 0 deg (double-cover)."""
    q = _rand_unit_quats(6, seed=9)
    ang = geodesic_angle_deg(-q, q.clone())
    assert torch.allclose(ang, torch.zeros_like(ang), atol=1e-6), ang


# --------------------------------------------------------------------------
# (4) gaussian_nll_loss behaviour + mse matches torch
# --------------------------------------------------------------------------
def test_mse_loss_matches_torch():
    torch.manual_seed(0)
    pred = torch.randn(16, 2, dtype=torch.float64)
    target = torch.randn(16, 2, dtype=torch.float64)
    ours = mse_loss(pred, target)
    ref = F.mse_loss(pred, target)
    assert torch.allclose(ours, ref, atol=1e-12)


def test_gaussian_nll_finite():
    torch.manual_seed(1)
    pred = torch.randn(32, 2, dtype=torch.float64)
    target = torch.randn(32, 2, dtype=torch.float64)
    log_std = torch.zeros_like(pred)
    loss = gaussian_nll_loss(pred, target, log_std)
    assert torch.isfinite(loss)


def test_gaussian_nll_decreases_as_log_std_adapts():
    """Holding (pred, target) fixed, NLL is minimized at the optimal log_std.

    For a diagonal Gaussian the per-dimension NLL ``log_std + 0.5*(d/sigma)^2``
    is minimized at ``log_std* = log|d|``. Adapting log_std from a poor fixed
    value toward that optimum must DECREASE the loss; overshooting past it must
    increase it again -- i.e. the loss genuinely tracks the spread.
    """
    torch.manual_seed(2)
    pred = torch.randn(64, 2, dtype=torch.float64)
    target = pred + 0.3 * torch.randn(64, 2, dtype=torch.float64)
    residual = (pred - target).abs().clamp_min(1e-6)
    log_std_opt = torch.log(residual)  # per-element NLL optimum

    # Way-too-small variance (overconfident) -> large NLL.
    loss_too_small = gaussian_nll_loss(pred, target, log_std_opt - 2.0)
    # Way-too-large variance (underconfident) -> also large NLL.
    loss_too_large = gaussian_nll_loss(pred, target, log_std_opt + 2.0)
    # The optimum -> smallest NLL of the three.
    loss_opt = gaussian_nll_loss(pred, target, log_std_opt)

    assert torch.isfinite(loss_opt)
    assert loss_opt < loss_too_small, (loss_opt.item(), loss_too_small.item())
    assert loss_opt < loss_too_large, (loss_opt.item(), loss_too_large.item())


def test_gaussian_nll_decreases_via_gradient_descent():
    """A few SGD steps on log_std (pred,target fixed) strictly reduce the NLL."""
    torch.manual_seed(3)
    pred = torch.randn(48, 2, dtype=torch.float64)
    target = pred + 0.5 * torch.randn(48, 2, dtype=torch.float64)
    log_std = torch.full_like(pred, 1.5, requires_grad=True)  # start underconfident

    opt = torch.optim.SGD([log_std], lr=0.1)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = gaussian_nll_loss(pred, target, log_std)
        loss.backward()
        assert torch.isfinite(log_std.grad).all()
        opt.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0], f"NLL did not decrease: {losses[0]} -> {losses[-1]}"
