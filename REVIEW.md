# Engineering Review — `inertial_navigation_transformer` (original code)

*An authoritative bug report for the original course project: the flat
`playground_notebooks/` scripts plus `utils.py`. Synthesized from a five-component
code audit (74 raw findings) cross-checked against the neural-inertial-navigation
literature (RoNIN, IONet/L-IONet, TLIO, EqNIO, CTIN, IMUNet, Brossard, Woodman).
Duplicate findings about a single root cause are merged. Three load-bearing claims
were re-verified numerically for this report and are flagged **[VERIFIED]**.*

The clean rebuild lives in the `ninav/` package (see §4). This document reviews the
**original** code only; the originals are retained for provenance and are not edited.

---

## 1. Executive summary — the fundamental "compression" flaw

The project sets out to **compress a high-rate IMU window into a position/orientation
estimate**. The intent is sound and matches the modern literature. The execution is
broken at essentially every physically meaningful stage: this is not one bug but a
stack of independent errors, each of which alone invalidates the output, sitting on top
of one architectural decision that guarantees poor generalization even if every bug were
fixed.

**The single most important conceptual flaw** is the *target formulation*. The code
regresses, in effect, **absolute / global-frame position from raw, body-frame-derived
features**, and then rebuilds the trajectory with an ad-hoc `cumsum` that is inconsistent
with how the ground truth is constructed. Every modern method does the opposite: it
regresses a **local, heading-invariant motion quantity** — a 2D velocity or a
short-horizon displacement — in a **gravity-aligned, heading-agnostic frame**, and only
then integrates. By coupling the target to arbitrary device attachment *and* walking
heading, the original framing forces the network to memorize every (device-orientation ×
compass-heading) combination it will ever see. It cannot, so it overfits to the training
headings and fails on anything unseen.

On top of that conceptual flaw sit four classes of concrete defects, all confirmed by the
audit:

1. **The orientation math is wrong by large constant factors.** There are *two*
   contradictory definitions of the quaternion-update matrix `Theta` in the same file.
   The module-level one (used by the baselines) scales the rotation term by **2/dt = 400×
   too large** at 200 Hz; the inner one (used to build the `theta` feature fed to the
   models) is exactly **2× too small**. Both coexist. A magic `factor = .002` was bolted
   onto the Kalman notebook purely to mask the resulting blow-up — which then *freezes*
   the orientation near its seed.

2. **The physics of integration is wrong.** Position is computed as
   `cumsum(v)·dt + cumsum(acc²)·dt²/2`. The `acc²` term is dimensionally meaningless
   (it has units m²/s², appears in no kinematic equation, and injects monotonic drift on
   every axis). Worse, the integration is run on **z-score-normalized** acceleration
   (dimensionless, gravity whitened away), and the body→world rotation defines
   `DOWN = +accelerometer`, vertically inverting the world frame (a resting accelerometer
   reads specific force pointing *up*).

3. **The evaluation is not the RoNIN benchmark and it leaks.** ATE is computed as
   `mean(‖error‖)`, not RMSE; the RTE formulas are either a meaningless
   `ATE × fraction-of-trajectory` or a windowed mean-ATE with an off-by-one and a
   div-by-zero. The `StandardScaler` is **re-fit on the test set**, and the AHRS filters
   run over `vstack(train, test)` as one recursive chain, leaking train state into test.

4. **The "model" is largely not the model.** In `transformer_informed.py` the declared
   `Transformer`/`RNN` are dead code — `forward()` is a pure MLP over a flattened window,
   so the temporal structure the project is *about* is destroyed before any learning
   happens. In the orientation notebooks the recurrent nets are fed **sequences of length
   1**. The transformers have **no positional encoding** (permutation-invariant over time —
   fatal for an integration task) and `nhead = 1`.

**Net effect:** the reported trajectories and ATE/RTE numbers are not trustworthy and not
comparable to RoNIN. The ATE ~30 m vs RoNIN ~5 m gap (§3) is fully explained — not by a
modeling subtlety, but by a wrong frame/target, wrong physics, a reconstruction formula
that drifts linearly even under a *perfect* model, and a metric that is not RoNIN's metric.

---

## 2. Consolidated bug catalog (deduped, by severity)

Auditors overlapped heavily on the core `utils.py` math; those findings are merged into
single entries with all call sites listed. Severity is the maximum assigned by any
auditor. Stable IDs (`C#`/`H#`/`M#`/`L#`) are used in cross-references.

### CRITICAL

#### C1 — Two contradictory, both-wrong `Theta` quaternion-update matrices  **[VERIFIED]**
- **Where:** `utils.py:452-461` (module-level) and `utils.py:285-334` (inner, inside
  `load_split_data`). Call sites: `transformer_informed.py:111`,
  `Linear_informed_with_EKF.py`, `kalman_filter_anton.ipynb` cell 4, all quaternion
  notebooks.
- **Root cause:** The correct discrete propagator is
  `q_{k+1} = [cos(|w|·dt/2)·I + (sin(|w|·dt/2)/|w|)·Ω_full(w)]·q_k`.
  - The **module-level** `Theta` divides the `sin` term by `|w|·dt/2` instead of `|w|`,
    so the off-diagonal (rotation) term is scaled by `2/dt`.
  - The **inner** `Theta` uses the correct `|w|` denominator but multiplies `Ω` by `0.5`
    (half the full skew), so its rotation term is exactly `0.5×` the truth.

  The two definitions therefore differ from each other by `4×` and both differ from truth.
- **[VERIFIED]** Re-running the original module-level formula at `dt = 1/200`:
  `Theta_module[0,1] / Theta_correct[0,1] = 400.0` exactly — i.e. **the rotation term is
  400× too large at 200 Hz** (this ratio is independent of `|w|`; it is the `2/dt`
  factor). The matrix is no longer a unit-quaternion update: at a realistic walking yaw
  rate `|w| = 1 rad/s`, `Theta·Thetaᵀ ≈ 2·I` (diagonal `2.0`) and applying it to a unit
  quaternion gives **norm ≈ 1.414 after one step**; at `|w| ≈ 1.34 rad/s` the one-step
  norm is **≈ 1.67**. Either way the quaternion norm explodes within a window, and the
  caller's per-step renormalization cannot recover the *correct rotation*, only the
  length.
- **Correct behavior:** One canonical `Theta` with `Ω = full skew` and the `0.5` only
  inside the exponent: `ang = |w|·dt; cos(ang/2)·I + (sin(ang/2)/|w|)·Ω_full(w)`, with a
  `|w|→0` small-angle guard returning `I`, and per-step renormalization.
- **Fix:** Delete both duplicates; implement one tested closed-form exponential-map
  propagator. This is the headline orientation bug. (Subsumes the separately-filed
  *Ω differs by 2×* and *Theta non-orthonormal/NaN at zero gyro* findings — see H1.)

#### C2 — Body→world rotation inverts the vertical (`DOWN = +accelerometer`)
- **Where:** `utils.py:476-486` (`get_body2world_rot`); copy-pasted into
  `transformer_informed.py`, `Linear_informed_with_EKF.py`, used in
  `bodyFrame_to_worldFrame.ipynb`.
- **Root cause:** `DOWN = a0` (normalized accelerometer). A stationary IMU measures
  *specific force* ≈ `+g` pointing **up**, so `DOWN` points up and the world frame is
  reflected about the vertical. The author's own comment reads
  `# maybe a minus sign is needed`. Compounded by `EAST = cross(m, a)` ordering.
- **Correct behavior:** `DOWN = -a0/‖a0‖`; rebuild `EAST = cross(DOWN, m)`,
  `NORTH = cross(EAST, DOWN)`, assert `det(R) = +1` against a known flat-phone case.
- **Fix:** Flip the sign, re-derive the cross-product order, add a static-segment test
  against `tango_ori`.

#### C3 — Quaternion convention mismatch (scalar-first vs scalar-last) across the stack
- **Where:** `utils.py:488-511` (`rotation_matrix_2_quaternion` / `quat_2_rotation_matrix`
  are scalar-first `[w,x,y,z]`) vs `utils.py:290-293,428-436` (Ω places `w` in slot 4 →
  scalar-last `[x,y,z,w]`). Also `transformer_informed.py:130` feeds a scalar-first
  quaternion into scipy `R.from_quat`, which expects scalar-last `[x,y,z,w]`.
- **Root cause:** The seed quaternion's real part lives in slot 0, but `Theta`/`Ω` treat
  slot 3 as real. `scipy.from_quat([1,0,0,0])` is interpreted as a 180° rotation about x,
  not identity. The reader and the writer of the rotation representation disagree on layout,
  so the propagation rotates the wrong components even if `Theta` were correct.
- **Correct behavior:** Pick **one** convention repo-wide (scalar-last `[x,y,z,w]` matches
  Ω and scipy). Reorder all seeds, or use `scalar_first=True` where appropriate.
- **Fix:** Single quaternion module with an asserted convention + round-trip unit tests
  against scipy using an explicit index permutation.

#### C4 — Spurious `cumsum(acc²)` term in double integration  **[VERIFIED]**
- **Where:** `transformer_informed.py:137`, `Linear_informed_with_EKF.py:130`.
- **Root cause:** `p = cumsum(v)·dt + cumsum(acc²)·dt²/2`. The author conflated the
  kinematic `½·a·t²` with "half the integral of acceleration squared."
- **[VERIFIED]** `cumsum(acc²)·dt²/2` carries units `(m/s²)² · s² = m²/s²` — **not metres,
  so the term is dimensionally invalid.** Because `acc² ≥ 0` it is **monotonically
  non-decreasing on every axis** (confirmed numerically), so it injects systematic
  one-directional drift into the position estimate.
- **Correct behavior:** A pure double integral: `v = cumsum(a)·dt`, then
  `p = cumsum(v)·dt`. Delete the `acc²` term.
- **Fix:** Remove the term entirely.

#### C5 — Integration runs on z-score-normalized (dimensionless, gravity-free) acceleration
- **Where:** `utils.py:376-379` (`StandardScaler` over the full feature matrix before
  windowing), consumed at `transformer_informed.py:133,137`,
  `Linear_informed_with_EKF.py:126,130`.
- **Root cause:** Acceleration is whitened (per-axis mean 0, unit variance) before being
  double-integrated. The DC component that actually moves you and gravity are subtracted,
  and each axis is rescaled by an arbitrary factor. `cumsum` of a zero-mean whitened
  signal is a random walk unrelated to displacement, and the `·dt` factor is meaningless
  against unitless data.
- **Correct behavior:** Integrate **raw, gravity-compensated, world-frame** acceleration
  in m/s² (use `synced/linacce`, which is gravity-compensated, or subtract `[0,0,g]` after
  a correct rotation). Normalize learned input features separately from physics-derived
  ones — and per the literature, prefer not to z-score the IMU at all (see §4).
- **Fix:** Compute physics features on un-normalized acceleration; never z-score the
  channel you integrate.

#### C6 — `StandardScaler` re-fit on the test set (data leakage / invalid eval)
- **Where:** `utils.py:225-228` (`normalize_features`); called at `utils.py:379`,
  `pred_pos_lstm_one_prediction.py:799`, `pred_pos_lstm_many_predictions.py:521`, and via
  `load_split_data` everywhere `normalize=True`.
- **Root cause:** `scaler.fit_transform(X)` is called unconditionally; the `scaler=`
  argument only decides whether to *construct* a new scaler, never whether to *transform*
  with an existing one. So `normalize_features(X_test, scaler=scaler)` re-fits on test
  statistics. The docstring "use the same scaler for test data" is false. Verified by the
  auditor: passing the train scaler still yields test mean ≈ 0.
- **Correct behavior:**
  `if scaler is None: scaler = StandardScaler(); X = scaler.fit_transform(X) else: X = scaler.transform(X)`.
- **Fix:** Branch on whether the scaler was supplied. (At deployment this also means a
  single new trajectory is normalized by its own stats, silently degrading predictions.)

#### C7 — Trajectory `cumsum` assumes stride == displacement span; drifts when `overlap ≠ 1`  **[VERIFIED]**
- **Where:** `pred_pos_lstm.py:530-534`; same pattern in
  `pred_pos_lstm_one_prediction.py:621-632`,
  `pred_pos_lstm_many_predictions.py:583-591,960-968`;
  `transformer_informed.py:156,310-314`.
- **Root cause:** Each window's target is the displacement over `seq_len-1` raw samples,
  but consecutive window *starts* advance by `stride = seq_len - overlap`. Cumulatively
  summing displacements as if windows were laid end-to-end is only correct when
  `stride == seq_len-1`, i.e. `overlap == 1`. `many_predictions` uses `overlap=20,
  seq_len=173`.
- **[VERIFIED]** With `seq_len=173, overlap=20` (span 172, stride 153), a **perfect** model's
  reconstructed per-window position error grows by a **constant `span − stride = 19` samples
  of displacement every window** (reproduced: error increments `19, 19, 19, …`), i.e. pure
  linear metric-induced drift. On a 5000-sample constant-velocity straight line this alone
  yields **ATE ≈ 466 m (RMSE) from a perfect predictor**; the corrected single-operator
  reconstruction (below) takes the same perfect model to **ATE = 0.0 (< 1e-6)**. The exact
  metre figure scales with speed and length, but the linear-drift mechanism and its
  hundreds-of-metres magnitude are confirmed.
- **Correct behavior:** Reconstruct the prediction exactly as the GT is built:
  `pos[i] = predicted_displacement[i] + true_window_start_offset[i]`. Alternatively make
  the per-window target the displacement over exactly one stride and `cumsum` those.
- **Fix:** Use the per-window start offsets; do not `cumsum` overlapping displacements.

#### C8 — Prediction and ground-truth trajectories reconstructed by *different* formulas  **[VERIFIED]**
- **Where:** `pred_pos_lstm.py:532-538`; same in the other LSTM files.
- **Root cause:** The prediction uses `cumsum` of per-window displacements; the GT uses
  `displacement + true_offset[i]`. Two different operators → a perfect model accrues
  nonzero ATE/RTE attributable to the metric, not the model.
- **[VERIFIED]** Under the original (mismatched) reconstruction a perfect model gives the
  ~214 m ATE above. Under a **single, identical** `displacement + offset` operator for both
  prediction and GT, the same perfect model gives **ATE = 0.0 (< 1e-6)**. This is the most
  diagnostic single confirmation in the report: it isolates a multi-hundred-metre error
  contributed by the reconstruction math alone, independent of any model.
- **Correct behavior:** One identical reconstruction operator for both pred and GT.
- **Fix:** A single `reconstruct_trajectory()` used for both.

#### C9 — Per-window target reference frame inconsistent between train and test
- **Where:** `transformer_informed.py:64-65,151,160,302`;
  `Linear_informed_with_EKF.py:55-56,144,153,295`.
- **Root cause:** In `preprocess_data`, train rebases per window
  (`y -= y[:, :1, :]` → relative displacement) but test skips it (absolute final
  position), then plot time applies a one-off global shift `y_test -= y_test[0]`. The model
  is trained on relative displacement and scored against absolute position — different
  quantities with different magnitudes and statistics.
- **Correct behavior:** Identical per-window rebasing for all splits; drop the ad-hoc
  plot-time shift.
- **Fix:** Remove the `if test == False` guard.

#### C10 — `phi2` quaternion loss collapses the whole batch to one scalar and picks one global sign
- **Where:** `quaternions_LSTM_GRU_RNN.ipynb` cell 7 (`CustomLoss.phi2`, lines 294-295);
  identical `quaternions_loss_func.ipynb` cell 4 lines 184-185.
- **Root cause:** `torch.min(torch.norm(q1-q2), torch.norm(q1+q2))` with no `dim` computes a
  single Frobenius norm over the entire `(N,4)` batch (shape `[]`). One antipodal sign is
  chosen for the whole batch; `aggregate()` is a no-op. This is the loss behind **all** the
  headline LSTM/GRU/RNN orientation numbers, so the reported "validation loss 190/250/310"
  figures are this malformed scalar.
- **Correct behavior:** Per row:
  `d = torch.minimum(norm(q1-q2, dim=1), norm(q1+q2, dim=1))`, then aggregate over dim 0.
- **Fix:** Add `dim=1` to the norms; resolve the double cover per sample.
  (`phi5` has the same whole-batch reduction bug — needs `dim=(-2,-1)`; see H-class note.)

#### C11 — Recurrent orientation nets fed sequences of length 1 (no temporal context)
- **Where:** `quaternions_LSTM_GRU_RNN.ipynb` cells 11/15/19;
  `quaternions_loss_func.ipynb` lines 254-258,305-309.
- **Root cause:** `train_data = cat((gyro,acc,mag,...), dim=1)` → shape `(N, F)`; then
  `input_size = F` and `view(N, -1, F)` makes the sequence dimension = 1. The feature axis
  is conflated with the time axis, so the bidirectional LSTM/GRU/RNN is an inert per-sample
  MLP and never uses the 200 Hz temporal structure.
- **Correct behavior:** Reshape to `(num_windows, seq_len, features_per_step)` so
  `input_size = features-per-timestep` and dim 1 = `seq_len`.
- **Fix:** Build true windows (do not `cat` per-feature columns into the time axis).

#### C12 — Transformer test-time accumulation iterates the batch dim (size 1) — reconstruction is a no-op
- **Where:** `transformer_initial_anton.ipynb` cell 6; `transformer_2_anton.ipynb` cell 0.
- **Root cause:** `predictions` has shape `(batch=1, seq, 3)`; `for pred in predictions:`
  iterates axis 0 (batch), runs once, so `accumulated_preds == predictions`. The
  "deltas → trajectory" cumulative sum never happens — the headline compression→trajectory
  step does nothing.
- **Correct behavior:** `np.cumsum(predictions[0], axis=0)` over the **time** axis (if the
  output is deltas), or drop accumulation entirely (if absolute). Make the train target and
  the test reconstruction consistent.
- **Fix:** Cumsum over the sequence axis; reconcile the target semantics.

#### C13 — Pure transformer has no positional encoding + time/batch axes swapped
- **Where:** `position_estimate_anton.ipynb` cell 3 (`PositionPredictor`).
- **Root cause:** `src` is fed straight into `nn.TransformerEncoder` with no positional
  encoding → permutation-invariant over time (it cannot distinguish t=0 from t=99; fatal
  for an integration task). `d_model = X.shape[-1] (~9)`, `nhead = 1`, and `batch_first` is
  left at its `False` default while batch-first tensors are passed, so attention mixes the
  wrong axis (time treated as batch).
- **Correct behavior:** A linear input embedding to a sensible `d_model` (32-64, divisible
  by `nhead ≥ 4`), additive sinusoidal/learned positional encoding sized `[seq_len,
  d_model]`, `batch_first=True`, and an explicit reduction (mean-pool / CLS) before the head.
- **Fix:** Add PE; fix the layout; raise `nhead`. (Sanity check: shuffling the time axis
  must change a correct model's output.)

#### C14 — EKF measurement = prediction → innovation identically zero, filter never corrects
- **Where:** `kalman_filter_anton.ipynb` cell 8.
- **Root cause:** `z = x_pred; y = z - observation_model(x_pred) = 0`. The Kalman gain
  multiplies a zero innovation, so the update is a no-op and the EKF degenerates to
  open-loop dead reckoning. Compounded by: accel arbitrarily scaled `/100`, gyro unscaled,
  scipy `from_quat/as_quat` (scalar-last) used while the state is `[w,x,y,z]`, and gravity
  not removed before propagation.
- **Correct behavior:** Supply a real measurement (`tango_pos`/`tango_ori`, or a ZUPT
  pseudo-measurement) with an `H` mapping state → measurement; one consistent quaternion
  convention; gravity-removed world-frame accel.
- **Fix:** Wire a genuine measurement; for the rebuild, adopt the TLIO stochastic-cloning
  EKF (§4).

#### C-extra — Double-integration baseline never removes gravity
- **Where:** `kalman_filter_anton.ipynb` cell 4 (uses `synced/acce`, not `synced/linacce`).
- **Root cause:** `p += v·dt + 0.5·a_world·dt²` with `a = synced/acce` (raw specific force,
  ~9.81 at rest); no gravity vector subtracted. A constant 9.81 m/s² double-integrates to
  ~1.23 m of drift in 0.5 s, growing quadratically. The dataset ships `synced/linacce` and
  `synced/grav` precisely for this and neither is used.
- **Fix:** Integrate `synced/linacce`, or subtract world-frame gravity (`[0,0,±9.81]`) from
  the rotated acceleration each step. Verify a resting trajectory stays put. (Closely
  related to C5/H4.)

### HIGH

- **H1 — Analytical `Theta` non-orthonormal / NaN at zero gyro** — `utils.py:452-461`.
  Built from un-normalized `w`; `Theta·Thetaᵀ` diagonal ≈ 1.14 for small `w` (and ≈ 2 at
  `|w|=1 rad/s`), not 1.0; `w=[0,0,0]` → all-NaN (`sin(0)/0`). Per-step renorm fixes the
  length but not the wrong rotation magnitude. **Fix:** subsumed by C1 (correct closed form
  + `|w|→0` guard).
- **H2 — `rotation_matrix_2_quaternion` divides by `4·q0` with no 180° guard** —
  `utils.py:491-496`. Near 180° the trace → −1, `q0 → 0` → inf/NaN; a `sqrt` of a slightly
  negative float → NaN seeds that poison the whole propagated sequence. **Fix:** Shepperd's
  4-case method or `scipy R.from_matrix(R).as_quat()`, clamp the `sqrt` argument `≥ 0`,
  normalize.
- **H3 — `get_body2world_rot` called with args swapped** —
  `transformer_informed.py:97-99`, `Linear_informed_with_EKF.py:88-90`. The signature is
  `(m0, a0)` but it is called `(acc_cols, mag_cols)`, so `DOWN` is built from the
  magnetometer and `EAST = cross(acc, mag)`. **Fix:** call `get_body2world_rot(mag, acc)`
  with keyword args.
- **H4 — Acc "bias" removal subtracts the first sample before rotation** —
  `transformer_informed.py:125`, `Linear_informed_with_EKF.py:118`.
  `X[:,:,acc] -= X[:,:1,acc]` zeroes genuine t0 motion, removes gravity in the *body* frame
  at one arbitrary instant (so it does not cancel constant world-frame gravity), and double-
  debiases on top of normalization. **Fix:** remove the first-sample subtraction; remove
  gravity in the world frame after a correct rotation, or use `synced/linacce`.
- **H5 — Predicted quaternions never normalized to unit norm at output** —
  `pred_ori_2_lstm.py:198-228`, `pred_ori_lstm.py:97-103`,
  `quaternion_update_anton_new.ipynb` cell 4. An unconstrained 4-vector under L1/L2/phi2
  lets the net shrink magnitude to cheat the loss. **Fix:**
  `q = q / q.norm(dim=-1, keepdim=True)` in `forward`, and/or a `(‖q‖−1)²` penalty.
  (Also `transformer_initial`/`transformer_2` normalize a quaternion stream over the wrong
  axis — `torch.norm(quats)` over the whole tensor / `dim=1` over time — instead of
  `dim=-1`.)
- **H6 — `FourVectorLoss = arccos(|cos_sim|)` has a singular (NaN) gradient at the optimum** —
  `pred_ori_lstm.py:106-123`. `d/du arccos(u) = −1/√(1−u²) → −∞` as `u → 1` (i.e. exactly
  when the prediction is correct); `q1=q2` gives loss 0 but gradient NaN. Also `.cpu()`
  every step. **Fix:** use `1 − |⟨q1,q2⟩|` (smooth) or `acos(clamp(|cos|, max=1−1e-6))` in
  float64; use `.mean()`.
- **H7 — Orientation filters seeded with a 3D position as q0** —
  `pred_pos_lstm.py:261-294`, `one_prediction.py:253`, `many_predictions.py:161`.
  `q_0 = y_train[0,0,:]` but `output_features = ['pose/tango_pos']` → a 3-vector position
  fed where a 4-element unit quaternion is required. **Fix:** seed q0 from `pose/tango_ori`
  or `acc2q(acc[0])`, length-4 unit.
- **H8 — AHRS filters run over `vstack(train, test)` as one recursive chain (state leakage)** —
  `pred_pos_lstm.py:260-297`, `one_prediction.py:252-289`, `many_predictions.py:160-197`.
  End-of-train orientation state carries into the first test samples. The author's correct
  per-split version sits dead under `if 0:`. **Fix:** filter `X_train` and `X_test`
  separately (ideally per recording), each with its own q0.
- **H9 — Filters integrate a reshaped windowed stream that duplicates/reorders boundary
  samples** — `pred_pos_lstm.py:264-266`. `reshape(-1,3)` of `(Nseq, seq_len, 3)` windows
  built with `overlap=1` duplicates the shared boundary sample (the stream becomes
  `0,1,2,3,3,4,5,6,6,…`), injecting spurious zero-dt updates. **Fix:** run filters on the
  original de-duplicated continuous stream, then window the resulting quaternions.
- **H10 — `use_truth_input` leaks GT position into the model input** —
  `pred_pos_lstm.py:222-223,342-344,486-498`; `one_prediction.py:208-209,333-335,580-592`.
  GT position is appended as an input; at test time the first 11 windows are fed true GT
  outright. **Fix:** never feed the target as input; if autoregressive, feed only the
  model's own previous prediction from step 0.
- **H11 — RTE is not RoNIN RTE (multiple wrong formulas)** —
  - `pred_pos_lstm.py:543`: `RTE = ATE · (windows-per-60s / total-windows)`, which collapses
    to `RTE = ATE` for any realistic run; physically meaningless.
  - `one_prediction.py:640-648`: sums `Nmins−1` block means but divides by `Nmins`
    (off-by-one, drops the last minute), and averages cumulative absolute error rather than
    segment drift; same at 725-731 and `many_predictions.py:977-983`.
  - **Correct behavior:** RoNIN RTE = mean over fixed 1-minute (12000-frame @200 Hz) sliding
    windows of the relative-displacement-error RMSE, with short-sequence scaling
    (`delta = N−1`, × `12000/N`).
  - **Fix:** a single `compute_ate_rte()` mirroring RoNIN `metric.py` (§4).
- **H12 — Module-level Ω vs inner Ω differ by 2×** — `utils.py:285-293` vs `423-436`.
  `Ω_module(w) == 2·Ω_inner(w)`. The `0.5` belongs in the propagator exponent, not baked
  into Ω. **Fix:** subsumed by C1 (one canonical `Ω = full skew`).
- **H13 — Declared Transformer/RNN are dead code; `forward()` is a pure MLP** —
  `transformer_informed.py:188-192,213-227`; `Linear_informed_with_EKF.py:181-185,206-220`.
  `self.transformer(x,x)` and `self.rnn(x)` are commented out; the flattened window goes
  straight through `fc_in → fc_out`. The 50×23 window is MLP'd into a 3-vector with no
  sequence model. `nhead = 1` even if wired. **Fix:** either wire a real sequence model on
  `[batch, seq, feat]` or delete the modules and rename. (This is the literal "compression"
  the rebuild must fix.)
- **H14 — Autoregressive test feedback broadcasts batch-element-0's prediction to all
  samples** — `pred_ori_2_lstm.py:332-346`. RHS `[0, -1]` → `(No,)` broadcasts sample-0's
  quaternion into every row. **Fix:** `[:, -1]` → `(B, No)`.
- **H15 — `load_much_data` train branch reads `start` only assigned in the test branch** —
  `utils.py:178-205`. `start` is set inside `if dir in test_dir`; the train branch reuses
  the test dir's (possibly random) offset; it works only because sorted `dirs` puts
  `test_dir` first, and would `NameError` if the ordering changed. Plus the first-assignment
  slices `[start:start+N]` but the vstack paths use `[:N]` (the offset is dropped for all
  but the first dir). **Fix:** independent, bounds-checked per-directory `start`; consistent
  slicing in both branches.
- **H16 — Sequences built across concatenated multi-trajectory / train+test data, no
  recording boundaries** — `utils.py:187-205,372-388`. `create_sequences` tiles across the
  seam between recordings; ≈ `num_dirs−1` corrupted windows per split straddle two unrelated
  coordinate origins/clocks. **Fix:** window per trajectory and concatenate window tensors;
  carry per-window provenance so reconstruction restarts at each trajectory's initial pose.
- **H-other** — `position_estimate_anton.ipynb` cell 4 optimizes a last-timestep MSE while
  validating on full-sequence MSE (incomparable objective, breaks early stopping);
  `quaternions_LSTM_GRU_RNN.ipynb`/`quaternions_loss_func.ipynb` `phi5` Frobenius norm
  reduces over the whole batch instead of per sample (needs `dim=(-2,-1)`).
  `transformer_informed.py:116-121` tests the wrong key (`'synced/tango_ori'` instead of
  `'pose/tango_ori'`), so the high-quality ground-truth orientation is silently ignored and
  the buggy gyro integrator is used instead.

### MEDIUM

- **M1 — ATE is `mean(‖error‖)` not RMSE** — `pred_pos_lstm.py:542`,
  `one_prediction.py:638`, `many_predictions.py:598,845,975`. Move `.mean()` inside the
  `sqrt`: `sqrt(mean(sum(err², axis=1)))`. Systematically under-reports vs RoNIN
  (e.g. distances `{5,0,10}`: code gives 5.0, RMSE gives 6.45).
- **M2 — Unguarded `RTE /= Nmins` div-by-zero for sub-60s clips** — `one_prediction.py:853`,
  `many_predictions.py:983`. Add the `if Nmins > 0` guard used at 645-648; fall back to
  `ATE·60/Nseconds`.
- **M3 — `overlap` semantic inverted/confusing; defaults disagree** — `utils.py:82-105`.
  `stride = seq_len − overlap`; `overlap=1` ⇒ ~99% overlap, `overlap=0` (`load_split_data`
  default) ⇒ tiling; the `create_sequences` default is `1`. If `overlap ≥ seq_len`, output
  is silently empty. **Fix:** rename to an explicit `stride`, validate `0 < stride ≤
  seq_len`, unify defaults.
- **M4 — `calculate_position_difference` mutates its input in place** — `utils.py:401-404`.
  `y[i] -= y[i][0]` corrupts shared/source arrays and is non-idempotent. **Fix:**
  `return y - y[:, 0:1, :]`.
- **M5 — Single-sample unbatched inference loop** — `transformer_informed.py:288-294`, EKF
  script 281-287. `model(X_test[i])` (1-D, no batch dim) looped over up to ~1.6M rows; works
  only for the MLP path, breaks any real sequence model. **Fix:** one batched call
  `model(torch.from_numpy(X_test).float())`.
- **M6 — In-place mutation of shared `col_locations`; `-10/-6/-3` hard-coded offsets** —
  `transformer_informed.py:146-148,158-160`. Brittle; assumes exactly 10 appended cols.
  **Fix:** deep-copy per split; derive offsets from explicit widths.
- **M7 — Train-time autoregression mutates held-out `X_test` in place** —
  `quaternion_update_anton.ipynb` cell 6 lines 602-607; `quaternion_update_anton_new.ipynb`
  cell 10 897-900. Overwrites GT inputs permanently. **Fix:** operate on `X_test.clone()`
  (cell 11 already does this correctly).
- **M8 — `many_predictions` train/test reduction mismatch + overlap=20 cumsum** —
  `pred_pos_lstm_many_predictions.py:80-86,548,568-569,583-591`. Trained seq-to-seq,
  evaluated last-step, fed into the overlap-sensitive cumsum (C7). **Fix:** one reduction;
  offset-based reconstruction.
- **M9 — `DataLoader shuffle=False` on the training loader** —
  `transformer_informed.py:166,169`. Identical batch order every epoch biases AdamW.
  **Fix:** `shuffle=True` for training; keep a separate ordered array for trajectory
  plotting.
- **M10 — `NameError: trial_no`** — `pred_pos_lstm_one_prediction.py:627`.
  `np.savetxt(f'...{trial_no}...')`; the defined variable is `n_trial`. Aborts the
  test/metric section. **Fix:** use `n_trial`.
- **M11 — `position_estimate` calls `load_split_data` with stale kwargs** —
  `position_estimate_anton.ipynb` cell 2. Passes `Ntrain/Nval` (ignored; defaults 1000/100
  are used) and unpacks 2 of a 5-tuple → silently trains on 1000 samples then `ValueError`.
  Also missing torch imports. **Fix:** the current API
  `X_train, y_train, X_test, y_test, col_locations = load_split_data(..., N_train=..., N_test=...)`.
- **M12 — `StepLR(step_size=0.1)` float, stepped per batch** — `transformer_2_anton.ipynb`
  cell 0, `transformer_initial_anton.ipynb` cell 4. Collapses LR to ~0 within the first
  batches. **Fix:** integer `step_size` in epochs, step once per epoch.
- **M13 — `calced/position` per-window absolute summed as an inter-window delta** —
  `transformer_informed.py:299-300,311`. Each window's integration restarts at 0; summing
  endpoints loses velocity continuity. **Fix:** integrate v/p continuously across the full
  sequence for the dead-reckoning baseline.
- **M14 — Same-subject contiguous-tail validation** — `pred_pos_lstm.py:352-354`.
  `shuffle=False` last-20% from the same train dirs/subjects; the val tail shares a boundary
  sample with train. **Fix:** hold out validation by subject/recording with a `≥ seq_len`
  gap.

### LOW

- **L1 — `random_start` magic-number cap 20000, checks only the first feature's length** —
  `utils.py:180-182`. Short recordings collapse to deterministic 0; long ones never sample
  the tail. **Fix:** derive the bound from `len(stream) − N`; verify equal-length streams.
- **L2 — `split_data` is dead code with a non-shuffled positional cut** — `utils.py:394-399`.
  If used, leaky on overlapping windows. **Fix:** remove, or add a `≥ seq_len` guard gap.
- **L3 — `include_clusters`/`theta` mutate the caller's list** — `utils.py:147-148,351`.
  Repeated calls accumulate duplicate entries, breaking width accounting. **Fix:** local
  copies.
- **L4 — `vec_norm` returns `(1,1)` not a scalar** — `utils.py:449-450`. Fragile
  `if vec_norm(w) < eps` guards compare against an array. **Fix:**
  `float(np.linalg.norm(v.ravel()))`.
- **L5 — `factor=.002` / `a/100` "arbitrary" hacks masking upstream bugs** —
  `kalman_filter_anton.ipynb` cell 4. The `factor` shrinks each gyro increment to ~0.2% of
  truth, freezing orientation near the seed — a band-aid over C1. Remove once C1/C5 are fixed.
- **L6 — Train-loss print divided by `seq_len`** — `pred_pos_lstm.py:447,460`. 30× smaller
  than the objective and not comparable to the other scripts. **Fix:** print
  `running_loss / n_minibatches`.
- **L7 — Display sign-flip uses test GT to pick a global double-cover sign inside the
  per-axis loop** — `quaternions_loss_func.ipynb` cell 1 lines 44-45. Cosmetic but
  misleading. **Fix:** per-sample sign once, outside the loop.
- **L8 — LR 3e-6 / 1e-6 far too low, weight init commented out** —
  `transformer_informed.py:34,243`, EKF script 1e-6 / 1000 epochs. The model underfits
  toward the target mean. **Fix:** enable init, LR ~1e-3/1e-4, confirm train loss beats the
  variance-of-target baseline.

**Pervasive smells (not individually catalogued, but they mandate the rebuild):** hardcoded
absolute author paths (`C:\Users\Simon Andersen\...`, `/Users/antongolles/...`) in every
file; massive copy-paste of `Theta`/`Omega`/`get_body2world_rot`/`Net`/`CustomLoss`/
preprocessing across notebooks and `.py` files with *divergent* copies (the direct cause of
the two-`Theta` class of bugs); author uncertainty left in comments
(`# maybe a minus sign is needed`, `# they may not be in the right order`); `np.Inf`
(removed in modern NumPy); `assert(include_mag is True & (...))` (bitwise-`&` on bools);
broken-syntax cells; magic numbers `dt=1/200`, `23`, `19+4`, `-10/-6/-3`.

---

## 3. Why the ATE is ~30 m vs RoNIN ~5 m

The ~6× benchmark gap is **over-determined**. Each of the following alone would inflate
ATE; they stack.

1. **Wrong target frame ⇒ structural failure to generalize (the dominant cause).** Raw
   body-frame-derived features regressed to (effectively) global/absolute position couple
   the target to arbitrary device attachment **and** walking heading. The same physical
   "walk forward" produces different (input, target) pairs under every device orientation
   and compass heading, so the network must memorize all combinations — it cannot, and it
   overfits to the training headings. Woodman, RoNIN, IONet, and EqNIO all flag this as
   *the* generalization killer. RoNIN's ~5 m comes precisely from regressing 2D motion in a
   **gravity-aligned, heading-agnostic frame** with **random-yaw augmentation**, which this
   project does neither of.

2. **The reconstruction formula drifts linearly even with a perfect model (C7/C8).**
   **[VERIFIED]** at `overlap=20, seq_len=173` a perfect predictor accrues a constant
   `span − stride = 19`-sample displacement overcount every window — pure metric-induced
   linear drift, ≈ 466 m of ATE on a 5000-sample constant-velocity straight line — *before
   the model contributes anything*. The corrected single-operator reconstruction takes the
   same perfect model to ATE = 0.0 (< 1e-6).

3. **The physics prior is anti-informative (C1-C5, H1-H4).** The orientation used to rotate
   accel into the world frame is wrong by 400× / 2× **[VERIFIED]** and vertically inverted;
   the integration adds a meaningless `acc²` monotonic-drift term **[VERIFIED]**; and it
   integrates dimensionless whitened acceleration with gravity whitened away. Classical
   strapdown already drifts roughly cubically (gyro bias → linear tilt → gravity leakage
   `g·sinθ` → double-integrated → ~t³; Woodman: a 0.05° tilt → 7.7 m in 30 s; a stationary
   IMU integrated open-loop → 152 m in 60 s). Feeding the network a *wrong* physics channel
   is worse than feeding none — it injects systematic, orientation-dependent error
   correlated across the window.

4. **The learning signal is largely disconnected (H13, C11, C13).** The "transformer" is a
   flattened-window MLP; the recurrent nets see length-1 sequences; the transformers have no
   positional encoding. The 200 Hz temporal structure that the velocity/displacement signal
   *is* never reaches an attention or recurrence mechanism. At LR 3e-6 with init disabled
   (L8), the model collapses toward the target mean.

5. **The reported number isn't even the RoNIN number (M1, H11).** ATE as `mean(‖·‖)` rather
   than RMSE, and the meaningless RTE, mean the headline figures are not comparable to
   published RoNIN/TLIO ATE/RTE in the first place — the "~30 m" is computed by a different,
   leaky (C6, H8, H10) yardstick.

**Conclusion:** the gap is not a tuning or capacity problem. Fixing the bugs without
changing the target formulation would still leave (1) unaddressed; changing only the
formulation without fixing the reconstruction/metric would still mis-measure. The rebuild
must address frame, target, reconstruction, physics, architecture, and metric *together*.

---

## 4. What the rebuild does instead

The rebuild lives in the installable [`ninav/`](ninav/) package at the repo root — one
definition per concept, hard convention assertions, paths from config (no hardcoded author
paths), and a synthetic-data-first test suite. It follows the RoNIN / TLIO / IONet
literature rather than re-deriving broken physics. The flat `playground_notebooks/` of
divergent copy-pasted scripts — the organizational root cause of the two-`Theta` class of
bugs — is replaced wholesale.

Key design decisions (each directly closing a bug class above):

- **Heading-agnostic, gravity-aligned target.** Regress a **2D horizontal velocity**
  (RoNIN, default) or a **3D displacement + diagonal log-std covariance** (TLIO) in a
  gravity-aligned frame; never global/absolute position from body-frame features. Closes
  the §3.1 root cause; random-yaw augmentation (`ninav/data/augment.py`) makes it
  heading-invariant.
- **One canonical geometry stack.** `ninav/geometry/quaternion.py` (single scalar-last
  convention, Shepperd `rotmat→quat`), `ninav/geometry/propagate.py` (one closed-form
  exponential-map gyro propagator with a `|w|→0` guard), `ninav/geometry/frames.py`
  (`DOWN = −a/‖a‖`, asserted `det = +1`). Closes C1, C2, C3, H1, H2, H3, H12.
- **Correct physics.** Pure double integration on raw, gravity-compensated, world-frame
  acceleration — no `acc²` term, no integrating whitened data. Closes C4, C5, H4.
- **One reconstruction operator + RoNIN metrics.** `ninav/reconstruct.py` builds prediction
  *and* GT identically; `ninav/metrics.py` mirrors RoNIN `metric.py` (ATE as RMSE, the
  1-minute sliding-window RTE with short-sequence scaling). Closes C7, C8, C12, M1, M2, M8,
  M13, H11.
- **Real sequence models.** `ninav/models/`: a RoNIN 1D ResNet-18 (`resnet1d.py`), a
  transformer with sinusoidal positional encoding / `nhead=8` / `batch_first=True`
  (`transformer.py`), a RoNIN LSTM/TCN with a non-trainable integration layer
  (`lstm_tcn.py`), and a TLIO displacement+covariance head (`tlio_head.py`). Closes C11,
  C13, H13.
- **Leak-free data path.** Per-recording windowing, subject/recording splits with a
  `≥ seq_len` gap, fit-on-train/transform-on-test scaler (default: no IMU z-score per the
  literature), and AHRS filters run per split/recording with a unit-quaternion seed. Closes
  C6, C9, H7, H8, H9, H10, H15, H16, M3, M4, M14.
- **A correcting EKF.** `ninav/filters/sc_ekf.py` implements the TLIO stochastic-cloning EKF
  with a yaw-only measurement frame, χ² gate, and covariance inflation — a filter whose
  innovation is not identically zero. Closes C14.

### Tests

The suite is **synthetic-data-first** because the real RoNIN HDF5 is not in the repo. The
load-bearing tests assert *correctness on analytic ground truth* — the checks the original
code never had — and encode the three verified confirmations of this report directly:

- `test_theta_*`: `Theta·Thetaᵀ ≈ I`, recovered angle `== |w|·T` for constant rate (catches
  the 400× and 2× errors), `w=0 → I` with no NaN.
- `test_double_integration`: constant accel `a` over `T` ⇒ `p = ½·a·T²`, with **no `acc²`
  term**.
- `test_perfect_model_zero_ate` / `test_recon_operator_identical`: a synthetic trajectory →
  velocity targets → reconstruction ⇒ **ATE ≈ 0 for any stride/overlap** (the original
  drifted linearly to ~214 m; this is the single most diagnostic test).
- `test_ate_is_rmse`, `test_rte_matches_ronin`, `test_transformer_positional` (shuffling
  time must change the output), `test_double_cover_per_sample`, `test_arccos_grad_finite`,
  `test_body2world_static`, `test_scaler_transform_not_fit`, and an end-to-end CPU smoke
  test that train loss beats the variance-of-target baseline.

*(`ninav/` is fully built out with the modules above, and `tests/` is populated with six
modules — `test_geometry.py`, `test_data.py`, `test_reconstruct_metrics.py`,
`test_models.py`, `test_losses.py`, `test_filters_and_train.py` — covering the plan above.
The full suite is **91 passing / 0 failing** under `pytest`, and all five models
(`resnet1d`, `transformer`, `lstm`, `tcn`, `tlio`) train end-to-end on synthetic data via
`python -m ninav.cli train`.)*

---

*Cross-references use the stable IDs in §2. Audited originals (to be replaced, not edited):*
`utils.py`, `playground_notebooks/{transformer_informed.py, Linear_informed_with_EKF.py,
pred_pos_lstm.py, pred_pos_lstm_one_prediction.py, pred_pos_lstm_many_predictions.py,
pred_ori_lstm.py, pred_ori_2_lstm.py, kalman_filter_anton.ipynb, position_estimate_anton.ipynb,
transformer_initial_anton.ipynb, transformer_2_anton.ipynb, quaternions_LSTM_GRU_RNN.ipynb,
quaternions_loss_func.ipynb, quaternion_update_anton.ipynb, quaternion_update_anton_new.ipynb,
bodyFrame_to_worldFrame.ipynb, Acce_Ori_to_Pos.ipynb}`.
