"""A compact, *correct* displacement-fusion EKF (a simplified analogue of TLIO's
stochastic-cloning EKF).

Why this exists
---------------
The old project shipped an "EKF" that did not actually filter: it propagated and
then "updated" against a measurement that was, by construction, identical to the
predicted measurement, so the innovation ``y = z - H @ x_pred`` was always (close
to) zero and the update never moved the state. A Kalman update with a zero
innovation is a no-op -- the "filter" just replayed the propagation. That class
of bug makes the filtered trajectory numerically equal to a plain integration of
the predictions, with none of the smoothing/outlier-rejection a real filter buys.

What this does instead
----------------------
We fuse the network's *per-window 2D displacement* predictions into a smooth 2D
trajectory with a textbook linear Kalman filter.

State (4-vector)::

    x = [px, py, vx, vy]

* **Propagation** -- constant-velocity model over the inter-window time ``dt``::

      x_pred = F @ x,    F = [[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1,  0],
                              [0, 0, 0,  1]]
      P_pred = F @ P @ F.T + Q

  ``Q`` is built from independent position and velocity process-noise spectral
  densities (``q_pos``, ``q_vel``) so the model is allowed to deviate from
  exact constant velocity.

* **Measurement** -- the network predicts the *displacement of the platform over
  the stride*, i.e. ``z ~= v * dt``. The measurement model that maps state to that
  observable is therefore::

      h(x) = (px - px_clone) ... -> in stochastic cloning you keep a "clone" of
      the position at the previous keyframe and measure the *difference* of the
      two position estimates against the network displacement.

  We implement exactly that with a single retained clone (stochastic cloning,
  one keyframe). Concretely the predicted measurement is the position change the
  filter expects between the previous and current keyframe; the innovation is the
  difference between the network's predicted displacement and that expected change.
  Because the network displacement is an *independent* sensor (not the filter's own
  propagation), the innovation is **generically non-zero** and the update genuinely
  corrects the state -- the property the old EKF lacked.

* **Outlier gate** -- a chi-square gate on the normalized innovation squared
  (NIS) ``y.T @ S^-1 @ y`` rejects measurements that disagree with the prediction
  by more than ``gate_chi2`` (default 13.8155 == 99.9th percentile of chi-square
  with 2 dof).

* **Measurement covariance inflation** -- consecutive windows overlap (stride <
  window_len), so their displacement errors are correlated; treating them as
  independent makes the filter overconfident. ``inflate_overlap`` (default x10)
  scales ``R`` to crudely de-weight that correlation, mirroring how TLIO inflates
  measurement noise for correlated cloned states.

This is intentionally a *single-clone* simplification of full stochastic cloning
(TLIO keeps the cloned covariance cross-terms); it is documented as such and is
correct for the constant-stride displacement-fusion use case.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DisplacementEKF"]


class DisplacementEKF:
    """Linear Kalman filter fusing per-window 2D displacement measurements.

    State ``x = [px, py, vx, vy]`` (navigation frame). Constant-velocity
    propagation between window keyframes; the measurement is the network's
    predicted displacement over one stride.

    Parameters
    ----------
    dt : float
        Time between consecutive window keyframes (``stride / fs``). Must be > 0.
    q_pos : float
        Position process-noise spectral density (m^2 / s). Allows the position
        to drift from the exact constant-velocity prediction.
    q_vel : float
        Velocity process-noise spectral density (m^2 / s^3). Larger => the filter
        trusts the displacement measurements more and adapts velocity faster.
    r_meas : float
        Per-axis measurement-noise variance of the network displacement (m^2).
    gate_chi2 : float, optional
        Chi-square gate threshold on the normalized innovation squared (2 dof).
        Default 13.8155 (99.9th percentile, 2 dof). Measurements above it are rejected.
    inflate_overlap : float, optional
        Multiplier applied to ``R`` to account for correlation between
        overlapping windows. Default 10.0 (i.e. "inflated x10"); set to 1.0 to
        disable.
    p0 : array-like (2,), optional
        Initial position. Default ``[0, 0]``.
    v0 : array-like (2,), optional
        Initial velocity. Default ``[0, 0]``.
    init_pos_var, init_vel_var : float, optional
        Initial state covariance (diagonal) for position / velocity.
    """

    def __init__(
        self,
        dt: float,
        q_pos: float,
        q_vel: float,
        r_meas: float,
        gate_chi2: float = 13.8155,
        inflate_overlap: float = 10.0,
        p0=None,
        v0=None,
        init_pos_var: float = 1.0,
        init_vel_var: float = 1.0,
    ):
        if not (dt > 0):
            raise ValueError(f"dt must be > 0, got {dt}")
        if q_pos < 0 or q_vel < 0 or r_meas <= 0:
            raise ValueError("q_pos, q_vel must be >= 0 and r_meas must be > 0")
        if gate_chi2 <= 0:
            raise ValueError(f"gate_chi2 must be > 0, got {gate_chi2}")
        if inflate_overlap <= 0:
            raise ValueError(f"inflate_overlap must be > 0, got {inflate_overlap}")

        self.dt = float(dt)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_meas = float(r_meas)
        self.gate_chi2 = float(gate_chi2)
        self.inflate_overlap = float(inflate_overlap)

        p0 = np.zeros(2) if p0 is None else np.asarray(p0, dtype=np.float64).reshape(2)
        v0 = np.zeros(2) if v0 is None else np.asarray(v0, dtype=np.float64).reshape(2)

        # State [px, py, vx, vy].
        self.x = np.concatenate([p0, v0]).astype(np.float64)
        self.P = np.diag(
            [init_pos_var, init_pos_var, init_vel_var, init_vel_var]
        ).astype(np.float64)

        # Constant-velocity transition matrix.
        dt = self.dt
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Process noise: independent position and velocity random walks.
        self.Q = np.diag(
            [
                self.q_pos * dt,
                self.q_pos * dt,
                self.q_vel * dt,
                self.q_vel * dt,
            ]
        ).astype(np.float64)

        # Effective per-axis measurement variance (inflated for window overlap).
        self.R = np.eye(2, dtype=np.float64) * (self.r_meas * self.inflate_overlap)

        # Stochastic-cloning keyframe: position at the previous accepted update.
        # The displacement measurement constrains (current position - clone).
        self._p_clone = self.x[:2].copy()

        # Diagnostics, recorded per step.
        self.last_innovation = np.zeros(2)
        self.last_nis = 0.0
        self.last_accepted = True
        self.n_rejected = 0

    # ------------------------------------------------------------------ #
    # Core filter step
    # ------------------------------------------------------------------ #
    def step(self, displacement_meas) -> np.ndarray:
        """Propagate one keyframe and fuse a single displacement measurement.

        Parameters
        ----------
        displacement_meas : array-like (2,)
            Network-predicted 2D displacement of the platform over this stride.

        Returns
        -------
        np.ndarray (4,)
            A copy of the posterior state ``[px, py, vx, vy]``.
        """
        z = np.asarray(displacement_meas, dtype=np.float64).reshape(2)

        # --- Predict ----------------------------------------------------
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Measurement model: the displacement observes (current position - clone).
        # Predicted measurement = predicted current position - retained clone.
        # H selects the position block (the clone is a constant offset, not a
        # filtered state in this single-clone simplification).
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        z_pred = x_pred[:2] - self._p_clone  # expected displacement since clone

        # --- Innovation -------------------------------------------------
        # NOTE: z (network displacement) is an INDEPENDENT sensor, so y is
        # generically non-zero -- this is the key property the old EKF lacked.
        y = z - z_pred
        S = H @ P_pred @ H.T + self.R
        S_inv = np.linalg.inv(S)

        # Chi-square gate on the normalized innovation squared (2 dof).
        nis = float(y @ S_inv @ y)
        self.last_innovation = y.copy()
        self.last_nis = nis

        if nis > self.gate_chi2:
            # Reject the outlier: keep the propagated state, advance the clone to
            # the predicted position so the *next* displacement is referenced
            # correctly, and do NOT apply a correction.
            self.x = x_pred
            self.P = P_pred
            self._p_clone = self.x[:2].copy()
            self.last_accepted = False
            self.n_rejected += 1
            return self.x.copy()

        # --- Update (Joseph form for covariance positivity) -------------
        K = P_pred @ H.T @ S_inv
        self.x = x_pred + K @ y
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        # Advance the stochastic-cloning keyframe to the just-estimated position.
        self._p_clone = self.x[:2].copy()
        self.last_accepted = True
        return self.x.copy()

    # ------------------------------------------------------------------ #
    # Batch helper
    # ------------------------------------------------------------------ #
    def run(self, displacements) -> np.ndarray:
        """Filter a whole sequence of per-window displacements.

        Parameters
        ----------
        displacements : array-like (T, 2)
            Per-window network displacement predictions.

        Returns
        -------
        np.ndarray (T, 4)
            Posterior state ``[px, py, vx, vy]`` after each window. The position
            columns ``[:, :2]`` form the filtered 2D trajectory.
        """
        disp = np.asarray(displacements, dtype=np.float64)
        if disp.ndim != 2 or disp.shape[1] != 2:
            raise ValueError(f"displacements must be (T, 2), got {disp.shape}")
        out = np.empty((disp.shape[0], 4), dtype=np.float64)
        for k in range(disp.shape[0]):
            out[k] = self.step(disp[k])
        return out
