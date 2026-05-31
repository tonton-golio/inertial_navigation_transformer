The structure confirms the audit's file references. I have everything needed. Producing the brief now as my final response.

---

# Findings & Design Brief — Neural Inertial Navigation Rebuild

*Single source of truth for the clean rebuild + bug report. Synthesized from a 5-component code audit (`utils.py`, the LSTM position pipeline, the "informed" position pipeline, the orientation/quaternion-loss pipeline, and the transformer/EKF baselines) and a literature corpus (RoNIN, IONet/L-IONet, TLIO, EqNIO, CTIN, IMUNet, Brossard gyro-denoising, Woodman strapdown, SO(3) loss survey).*

---

## 1. Executive summary — the fundamental flaw

The project tries to **"compress" a high-rate IMU window into a position/orientation estimate**, but it does so in a way that is broken at essentially every physically meaningful stage. There is not one bug; there is a stack of independent errors that each individually invalidate the output, plus one architectural choice that guarantees poor generalization even if every bug were fixed.

The single most important conceptual flaw: **the project regresses (effectively) absolute / global-frame position from raw, body-frame-derived features, and reconstructs the trajectory by an ad-hoc `cumsum` that is inconsistent with how the ground truth is built.** Every modern method (RoNIN, IONet, TLIO, EqNIO) does the opposite — it regresses a **local, heading-invariant motion quantity** (2D velocity or short-horizon displacement) **in a gravity-aligned, heading-agnostic frame**, then integrates. The project's framing forces the network to memorize device orientation × walking heading combinations it cannot learn, so it overfits and fails on anything unseen.

On top of that conceptual flaw sit four classes of concrete defects, all confirmed numerically by the auditors:

1. **The orientation math is wrong by large constant factors.** There are *two* contradictory definitions of the quaternion-update matrix `Theta` in the same file. The module-level one (used by the baselines) scales the rotation term by `2/dt = 400×` too large (`Theta@Theta.T ≈ 2·I`, norm inflates ~40%/step); the inner one (used to build the `theta` feature fed to the models) is exactly `2×` too small (half the true angular rate). Both coexist. A magic `factor=.002` was bolted on in the Kalman notebook purely to mask the blow-up, which then *freezes* the orientation.

2. **The physics of integration is wrong.** Position is computed as `cumsum(v)·dt + cumsum(acc²)·dt²/2` — the `acc²` term is dimensionally meaningless (no kinematic equation contains it; it injects monotonic drift on every axis). Integration is run on **z-score-normalized** acceleration (dimensionless, gravity removed by whitening), and gravity is never properly removed in the world frame. The body→world rotation sets `DOWN = +accelerometer`, but a resting accelerometer reads specific force pointing **up**, so the world frame is vertically inverted.

3. **The evaluation is not the RoNIN benchmark and leaks.** ATE is computed as `mean(‖error‖)` not RMSE; RTE formulas are either a meaningless `ATE × fraction-of-trajectory` or a windowed mean-ATE with an off-by-one and a div-by-zero. The `StandardScaler` is **re-fit on the test set** (`fit_transform` is called unconditionally even when a fitted scaler is passed), so train and test live in different normalized spaces. AHRS filters are run over `vstack(train, test)` as one recursive chain, leaking train state into test.

4. **The "model" isn't the model.** In `transformer_informed.py` the declared `Transformer`/`RNN` are dead code — `forward()` is a pure MLP over a flattened window, so the temporal structure the project is about is destroyed before any learning happens. In the LSTM/GRU/RNN orientation notebooks the recurrent nets are fed **sequences of length 1** (feature axis conflated with time axis), so the recurrence is inert. The transformers have no positional encoding (permutation-invariant over time — fatal for an integration task) and `nhead=1`.

**Net effect:** the reported trajectories and ATE/RTE numbers are not trustworthy and not comparable to RoNIN. Their ATE (~30 m) vs RoNIN (~5 m) is fully explained — not by a modeling subtlety, but by (a) wrong frame/target, (b) wrong physics, (c) a reconstruction formula that drifts linearly even with a *perfect* model, and (d) a metric that isn't RoNIN's metric.

---

## 2. Consolidated bug catalog (deduped, by severity)

Auditors overlapped heavily on the core `utils.py` math; those are merged into single entries below with all call sites listed. Severity is the max assigned by any auditor.

### CRITICAL

---

**C1 — Two contradictory, both-wrong `Theta` quaternion-update matrices**
`utils.py:285–334` (inner, in `load_split_data`) and `utils.py:423–461` (module-level). Call sites: `transformer_informed.py:111`, `Linear_informed_with_EKF.py`, `kalman_filter_anton.ipynb` cell 4, all quaternion notebooks.

- **Root cause:** The correct discrete propagator is `q_{k+1} = [cos(|w|dt/2)·I + (sin(|w|dt/2)/|w|)·Ω_full(w)]·q_k`. The **module-level** `Theta` divides the `sin` term by `|w|·dt/2` instead of `|w|` → off-diagonal scaled by `2/dt = 400×` at 200 Hz (verified `Theta_mod[0,1]/Theta_correct[0,1] ≈ 400`; `Theta@Theth.T ≈ 2·I`, applying to a unit quaternion gives norm ≈ 1.4). The **inner** `Theta` uses the correct `|w|` denominator but multiplies Ω by `0.5` (so Ω is half the full skew) → rotation term exactly `0.5×` too small (half the true angular rate). The two definitions differ by 2× from each other and both differ from truth.
- **Correct behavior:** One canonical `Theta` with `Ω = full skew` and the `0.5` only inside the exponent: `ang = |w|·dt; cos(ang/2)·I + (sin(ang/2)/|w|)·Ω_full(w)`, with a `|w|→0` small-angle guard returning `I`, and per-step renormalization.
- **Fix:** Delete both duplicates; implement one tested closed-form exponential-map propagator (see §4/§5). This is the headline orientation bug.

---

**C2 — Body→world rotation inverts the vertical (`DOWN = +accelerometer`)**
`utils.py:476–486` (`get_body2world_rot`); copy-pasted into `transformer_informed.py`, `Linear_informed_with_EKF.py`, used in `bodyFrame_to_worldFrame.ipynb`.

- **Root cause:** `DOWN = a0` (normalized accelerometer). A stationary IMU measures specific force ≈ `+g` pointing **up**. So `DOWN` points up; the world frame is reflected about the vertical. Author's own comment: `# maybe a minus sign is needed`. Compounded by `EAST = cross(m, a)` ordering.
- **Correct behavior:** `DOWN = -a0/‖a0‖`; rebuild `EAST = cross(DOWN, m)`, `NORTH = cross(EAST, DOWN)`, assert `det(R)=+1` against a known flat-phone case.
- **Fix:** Flip the sign, re-derive cross-product order, add a static-segment test vs `tango_ori`.

---

**C3 — Quaternion convention mismatch (scalar-first vs scalar-last) across the whole stack**
`utils.py:488–511` (`rotation_matrix_2_quaternion`/`quat_2_rotation_matrix` are scalar-first `[w,x,y,z]`) vs `utils.py:290–293,428–436` (Ω places `w` in slot 4 → scalar-last `[x,y,z,w]`). Also `transformer_informed.py:130` feeds a scalar-first quaternion into `scipy R.from_quat` which expects scalar-last `[x,y,z,w]`.

- **Root cause:** The seed quaternion's real part is in slot 0, but `Theta`/`Ω` treat slot 3 as real; `scipy.from_quat([1,0,0,0])` is interpreted as a 180° rotation about x, not identity. Reader and writer of the rotation representation disagree on layout.
- **Correct behavior:** Pick **one** convention repo-wide (scalar-last `[x,y,z,w]` matches Ω and scipy). Reorder all seeds / use `scalar_first=True` where appropriate.
- **Fix:** Single quaternion module with asserted convention + round-trip unit tests against scipy with an explicit index permutation (§5).

---

**C4 — Spurious `cumsum(acc²)` term in double integration**
`transformer_informed.py:137`, `Linear_informed_with_EKF.py:130`.

- **Root cause:** `p = cumsum(v)·dt + cumsum(acc²)·dt²/2`. The author conflated the kinematic `½·a·t²` with "half the integral of acceleration squared." `acc²` has units `(m/s²)²`, is always ≥0 (monotonic drift), and is not in any kinematic equation.
- **Correct behavior:** `p = cumsum(cumsum(a)·dt)·dt` (pure double integral), i.e. `p = cumsum(v)·dt` with `v = cumsum(a)·dt`. Delete the `acc²` term.
- **Fix:** Remove the term entirely.

---

**C5 — Integration runs on z-score-normalized (dimensionless, gravity-free) acceleration**
`utils.py:376–379` (`StandardScaler` over full feature matrix before windowing), consumed at `transformer_informed.py:133,137`, `Linear_informed_with_EKF.py:126,130`.

- **Root cause:** Acceleration is whitened (mean 0, unit variance per axis) before being double-integrated. The DC component that actually moves you and gravity are subtracted; each axis is rescaled by an arbitrary factor. `cumsum` of a zero-mean whitened signal is a random walk unrelated to displacement; the `·dt` factor is meaningless against unitless data.
- **Correct behavior:** Integrate **raw, gravity-compensated, world-frame** acceleration in m/s² (use `synced/linacce` which is gravity-compensated, or subtract `[0,0,g]` after a correct rotation). Normalize learned input features separately from physics-derived ones.
- **Fix:** Compute physics features on un-normalized acceleration; never z-score the channel you integrate.

---

**C6 — `StandardScaler` re-fit on the test set (data leakage / invalid eval)**
`utils.py:225–228` (`normalize_features`); called at `utils.py:379`, `pred_pos_lstm_one_prediction.py:799`, `pred_pos_lstm_many_predictions.py:521`, and via `load_split_data` everywhere `normalize=True`.

- **Root cause:** `scaler.fit_transform(X)` is called unconditionally; the `scaler=` argument only decides whether to *construct* a new one, never to *transform*. So `normalize_features(X_test, scaler=scaler)` re-fits on test stats. The docstring "use the same scaler for test data" is false.
- **Correct behavior:** `if scaler is None: scaler = StandardScaler(); X = scaler.fit_transform(X) else: X = scaler.transform(X)`.
- **Fix:** Branch on whether the scaler was supplied. (Note: per the literature, the rebuild should likely **not** z-score IMU at all — see §4 normalization.)

---

**C7 — Trajectory reconstruction `cumsum` assumes stride == displacement span; drifts whenever `overlap ≠ 1`**
`pred_pos_lstm.py:530–534`; same pattern `pred_pos_lstm_one_prediction.py:621–632`, `pred_pos_lstm_many_predictions.py:583–591,960–968`; `transformer_informed.py:156,310–314`.

- **Root cause:** Each window's target is displacement over `seq_len-1` raw samples, but consecutive window *starts* advance by `stride = seq_len - overlap`. Cumulatively summing displacements as if windows were laid end-to-end is only correct when `stride == seq_len-1`, i.e. `overlap == 1`. `many_predictions` uses `overlap=20, seq_len=173`. Verified: a *perfect* model drifts linearly (0, 0, −19, −38, −57, …) purely from the formula.
- **Correct behavior:** Reconstruct prediction the same way as GT: `pos[i] = predicted_displacement[i] + true_window_start_offset[i]`. Or make the per-window target the displacement over exactly one stride and `cumsum` those.
- **Fix:** Use the per-window start offsets; do not `cumsum` overlapping displacements.

---

**C8 — Prediction and ground-truth trajectories reconstructed by *different* formulas**
`pred_pos_lstm.py:532–538`; same in the other LSTM files.

- **Root cause:** Prediction uses `cumsum` of per-window displacements; GT uses `displacement + true_offset[i]`. Two different operators → a perfect model gets nonzero ATE/RTE attributable to the metric, not the model.
- **Correct behavior:** Apply one identical reconstruction operator to both pred and GT.
- **Fix:** Single `reconstruct_trajectory()` used for both (§5).

---

**C9 — Per-window target reference frame inconsistent between train and test**
`transformer_informed.py:64–65,151,160,302`; `Linear_informed_with_EKF.py:55–56,144,153,295`.

- **Root cause:** In `preprocess_data`, train rebases per window (`y -= y[:, :1, :]` → relative displacement) but test skips it (absolute final position), then plot time applies a one-off global shift `y_test -= y_test[0]`. The model is trained on relative displacement and scored against absolute position — different quantities, different magnitudes.
- **Correct behavior:** Identical per-window rebasing for all splits; drop the ad-hoc plot-time shift.
- **Fix:** Remove the `if test==False` guard.

---

**C10 — phi2 quaternion loss collapses the whole batch to one scalar and picks one global sign**
`quaternions_LSTM_GRU_RNN.ipynb` cell 7 (`CustomLoss.phi2`, lines 294–295); identical `quaternions_loss_func.ipynb` cell 4 lines 184–185.

- **Root cause:** `torch.min(torch.norm(q1-q2), torch.norm(q1+q2))` with no `dim` → single Frobenius norm over the entire `(N,4)` batch (shape `[]`). One antipodal sign is chosen for the whole batch; `aggregate()` is a no-op. This is the loss behind **all** the headline LSTM/GRU/RNN orientation numbers.
- **Correct behavior:** Per-row: `d = torch.minimum(norm(q1-q2,dim=1), norm(q1+q2,dim=1))`, then aggregate over dim 0.
- **Fix:** Add `dim=1` to the norms; resolve double-cover per sample. (phi5 has the same whole-batch reduction bug — `dim=(-2,-1)` needed.)

---

**C11 — Recurrent orientation nets fed sequences of length 1 (no temporal context)**
`quaternions_LSTM_GRU_RNN.ipynb` cells 11/15/19; `quaternions_loss_func.ipynb` lines 254–258,305–309.

- **Root cause:** `train_data = cat((gyro,acc,mag,...), dim=1)` → shape `(N, F)`; then `input_size = F` and `view(N, -1, F)` makes the seq dim = 1. The feature axis is conflated with the time axis. The bidirectional LSTM/GRU/RNN is effectively a per-sample MLP.
- **Correct behavior:** Reshape to `(num_windows, seq_len, features_per_step)` so `input_size = features-per-timestep` and dim 1 = `seq_len`.
- **Fix:** Build true windows (don't `cat` per-feature columns into the time axis).

---

**C12 — Transformer test-time accumulation iterates the batch dim (size 1) — reconstruction is a no-op**
`transformer_initial_anton.ipynb` cell 6; `transformer_2_anton.ipynb` cell 0.

- **Root cause:** `predictions` is `(batch=1, seq, 3)`; `for pred in predictions:` iterates axis 0 (batch), runs once, so `accumulated_preds == predictions`. The "deltas → trajectory" cumulative sum never happens.
- **Correct behavior:** `np.cumsum(predictions[0], axis=0)` over the **time** axis (if deltas), or drop accumulation (if absolute). Make train target and test reconstruction consistent.
- **Fix:** Cumsum over the sequence axis; reconcile target semantics.

---

**C13 — Pure transformer has no positional encoding + time/batch axes swapped**
`position_estimate_anton.ipynb` cell 3 (`PositionPredictor`).

- **Root cause:** `src` fed straight into `nn.TransformerEncoder` with no positional encoding → permutation-invariant over time (cannot tell t=0 from t=99; fatal for integration). `d_model = X.shape[-1] (~9)`, `nhead=1`, `batch_first` left at default `False` while batch-first tensors are passed → attention mixes the wrong axis.
- **Correct behavior:** Linear input embedding to a sensible `d_model` (32–64, divisible by `nhead≥4`), additive sinusoidal/learned positional encoding sized `[seq_len, d_model]`, `batch_first=True`, explicit reduction (mean-pool / CLS) before the head.
- **Fix:** Add PE; fix layout; raise `nhead`. (Sanity check: shuffling the time axis must change a correct model's output.)

---

**C14 — EKF measurement = prediction → innovation identically zero, filter never corrects**
`kalman_filter_anton.ipynb` cell 8.

- **Root cause:** `z = x_pred; y = z - observation_model(x_pred) = 0`. The Kalman gain multiplies a zero innovation → update is a no-op → open-loop dead reckoning. Plus: accel arbitrarily scaled `/100`, gyro unscaled, scipy `from_quat/as_quat` (scalar-last) used while state is `[w,x,y,z]`, gravity not removed.
- **Correct behavior:** Supply a real measurement (`tango_pos`/`tango_ori`, or ZUPT pseudo-measurement) with an `H` mapping state→measurement; consistent quaternion convention; gravity-removed world-frame accel.
- **Fix:** Wire a genuine measurement; or, for the rebuild, adopt the TLIO stochastic-cloning EKF (§4).

### HIGH

---

**H1 — Analytical `Theta` is non-orthonormal / NaN at zero gyro** — `utils.py:452–461`. Built from un-normalized `w`; `Theta@Theth.T` diagonal 1.14 not 1.0; `w=[0,0,0]` → all-NaN (`sin(0)/0`). Per-step renorm fixes length but not the wrong rotation magnitude. **Fix:** subsumed by C1 (correct closed form + `|w|→0` guard).

**H2 — `rotation_matrix_2_quaternion` divides by `4·q0` with no 180° singularity guard** — `utils.py:491–496`. Near 180°, `trace→−1`, `q0→0` → inf/NaN; `sqrt` of negative from float error → NaN seeds that poison the whole sequence. **Fix:** Shepperd's 4-case method or `scipy R.from_matrix(R).as_quat()`, clamp `sqrt` arg `≥0`, normalize.

**H3 — `get_body2world_rot` called with args swapped** — `transformer_informed.py:97–99`, `Linear_informed_with_EKF.py:88–90`. Signature is `(m0, a0)` but called `(acc_cols, mag_cols)` → `DOWN` built from magnetometer, `EAST=cross(acc,mag)`. **Fix:** call `get_body2world_rot(mag, acc)` with keyword args.

**H4 — Acc "bias" removal subtracts the first sample before rotation** — `transformer_informed.py:125`, `Linear_informed_with_EKF.py:118`. `X[:,:,acc] -= X[:,:1,acc]` zeroes genuine t0 motion, removes gravity in the *body* frame at one arbitrary instant (so it doesn't cancel constant world-frame gravity), and double-debiases on top of normalization. **Fix:** remove first-sample subtraction; remove gravity in world frame after a correct rotation, or use `synced/linacce`.

**H5 — Predicted quaternions never normalized to unit norm at output** — `pred_ori_2_lstm.py:198–228`, `pred_ori_lstm.py:97–103`, `quaternion_update_anton_new.ipynb` cell 4. Unconstrained 4-vector under L1/L2/phi2 lets the net shrink magnitude to cheat the loss. **Fix:** `q = q / q.norm(dim=-1, keepdim=True)` in `forward`, and/or a `(‖q‖−1)²` penalty.

**H6 — `FourVectorLoss = arccos(|cos_sim|)` has a singular (NaN) gradient at the optimum** — `pred_ori_lstm.py:106–123`. `d/du arccos(u) = −1/√(1−u²) → −∞` as `u→1` (i.e. exactly when correct); `q1=q2` gives loss 0 but grad NaN. Also `.cpu()` every step. **Fix:** use `1 − |⟨q1,q2⟩|` (smooth) or `acos(clamp(|cos|, max=1−1e-6))` in float64; use `.mean()`.

**H7 — Orientation filters seeded with a 3D position as q0** — `pred_pos_lstm.py:261–294`, `one_prediction.py:253`, `many_predictions.py:161`. `q_0 = y_train[0,0,:]` but `output_features=['pose/tango_pos']` → a 3-vector position fed where a 4-element unit quaternion is required. **Fix:** seed q0 from `pose/tango_ori` or `acc2q(acc[0])`, length-4 unit.

**H8 — AHRS filters run over `vstack(train, test)` as one recursive chain (state leakage)** — `pred_pos_lstm.py:260–297`, `one_prediction.py:252–289`, `many_predictions.py:160–197`. End-of-train orientation state carries into first test samples. The author's correct per-split version sits dead under `if 0:`. **Fix:** filter `X_train` and `X_test` separately (ideally per recording), each with its own q0.

**H9 — Filters integrate a reshaped windowed stream that duplicates/reorders boundary samples** — `pred_pos_lstm.py:264–266`. `reshape(-1,3)` of `(Nseq, seq_len, 3)` windows built with `overlap=1` duplicates the shared boundary sample (stream becomes `0,1,2,3,3,4,5,6,6,…`), injecting spurious zero-dt updates. **Fix:** run filters on the original de-duplicated continuous stream, then window the resulting quaternions.

**H10 — `use_truth_input` leaks GT position into model input** — `pred_pos_lstm.py:222–223,342–344,486–498`; `one_prediction.py:208–209,333–335,580–592`. GT position appended as input; at test the first 11 windows are fed true GT outright. **Fix:** never feed the target as input; if autoregressive, feed only the model's own previous prediction from step 0.

**H11 — RTE is not RoNIN RTE (multiple wrong formulas)** —
- `pred_pos_lstm.py:543`: `RTE = ATE · (windows-per-60s / total-windows)` → collapses to `RTE = ATE` for any realistic run; physically meaningless.
- `one_prediction.py:640–648`: sums `Nmins−1` block means, divides by `Nmins` (off-by-one, drops last minute), and averages cumulative absolute error not segment drift; same at 725–731 and `many_predictions.py:977–983`.
- **Correct behavior:** RoNIN RTE = mean over fixed 1-minute (12000-frame @200 Hz) sliding windows of the relative-displacement-error RMSE, with short-sequence scaling (`delta=N−1`, ×`12000/N`).
- **Fix:** single `compute_ate_rte()` mirroring RoNIN `metric.py` (§4/§5).

**H12 — Module-level Ω vs inner Ω differ by 2×** — `utils.py:285–293` vs `423–436`. `Ω_module(w) == 2·Ω_inner(w)`. The 0.5 belongs in the propagator exponent, not baked into Ω. **Fix:** subsumed by C1 (one canonical Ω = full skew).

**H13 — Declared Transformer/RNN are dead code; `forward()` is a pure MLP** — `transformer_informed.py:188–192,213–227`; `Linear_informed_with_EKF.py:181–185,206–220`. `self.transformer(x,x)` and `self.rnn(x)` are commented out; the flattened window goes straight through `fc_in→fc_out`. The 50×23 window is MLP'd into a 3-vector with no sequence model. `nhead=1` even if wired. **Fix:** either wire a real sequence model on `[batch, seq, feat]` or delete the modules and rename.

**H14 — Autoregressive test feedback broadcasts batch-element-0's prediction to all samples** — `pred_ori_2_lstm.py:332–346`. RHS `[0, -1]` → `(No,)` broadcasts sample-0's quaternion into every row. **Fix:** `[:, -1]` → `(B, No)`.

**H15 — `load_much_data` train branch reads `start` only assigned in the test branch** — `utils.py:178–205`. `start` is set inside `if dir in test_dir`; the train branch reuses the test dir's (possibly random) offset; works only because sorted `dirs` puts `test_dir` first; `NameError` if ordering changes. Plus first-assignment slices `[start:start+N]` but vstack paths use `[:N]` (offset dropped for all but the first dir). **Fix:** independent, bounds-checked per-directory `start`; consistent slicing in both branches.

**H16 — Sequences built across concatenated multi-trajectory / train+test data, no recording boundaries** — `utils.py:187–205,372–388`. `create_sequences` tiles across the seam between recordings; ≈`num_dirs−1` corrupted windows per split straddle two unrelated coordinate origins/clocks. **Fix:** window per trajectory and concatenate window tensors; carry per-window provenance so reconstruction restarts at each trajectory's initial pose.

### MEDIUM

- **M1 — ATE is `mean(‖error‖)` not RMSE** — `pred_pos_lstm.py:542`, `one_prediction.py:638`, `many_predictions.py:598,845,975`. Move `.mean()` inside the `sqrt`: `sqrt(mean(sum(err²,axis=1)))`. Systematically under-reports vs RoNIN.
- **M2 — Unguarded `RTE /= Nmins` div-by-zero for sub-60s clips** — `one_prediction.py:853`, `many_predictions.py:983`. Add the `if Nmins>0` guard used at 645–648; fall back to `ATE·60/Nseconds`.
- **M3 — `overlap` semantic inverted/confusing; defaults disagree** — `utils.py:82–105`. `stride = seq_len − overlap`; `overlap=1` ⇒ ~99% overlap, `overlap=0` (`load_split_data` default) ⇒ tiling; `create_sequences` default is `1`. If `overlap≥seq_len`, empty output silently. **Fix:** rename to explicit `stride`, validate `0<stride≤seq_len`, unify defaults.
- **M4 — `calculate_position_difference` mutates its input in place** — `utils.py:401–404`. `y[i] -= y[i][0]` corrupts shared/source arrays, non-idempotent. **Fix:** `return y - y[:, 0:1, :]`.
- **M5 — Single-sample unbatched inference loop** — `transformer_informed.py:288–294`, EKF script 281–287. `model(X_test[i])` (1-D, no batch dim) looped over up to ~1.6M rows; works only for the MLP path, breaks any real sequence model. **Fix:** one batched call `model(torch.from_numpy(X_test).float())`.
- **M6 — In-place mutation of shared `col_locations`; `-10/-6/-3` hard-coded offsets** — `transformer_informed.py:146–148,158–160`. Brittle; assumes exactly 10 appended cols. **Fix:** deep-copy per split; derive offsets from explicit widths.
- **M7 — Train-time autoregression mutates held-out `X_test` in place** — `quaternion_update_anton.ipynb` cell 6 lines 602–607; `quaternion_update_anton_new.ipynb` cell 10 897–900. Overwrites GT inputs permanently. **Fix:** operate on `X_test.clone()` (cell 11 already does this correctly).
- **M8 — `many_predictions` train/test reduction mismatch + overlap=20 cumsum** — `pred_pos_lstm_many_predictions.py:80–86,548,568–569,583–591`. Trained seq-to-seq, evaluated last-step, fed into the overlap-sensitive cumsum. **Fix:** one reduction; offset-based reconstruction.
- **M9 — `DataLoader shuffle=False` on the training loader** — `transformer_informed.py:166,169`. Identical batch order every epoch biases AdamW. **Fix:** `shuffle=True` for training; keep a separate ordered array for trajectory plotting.
- **M10 — `NameError: trial_no`** — `pred_pos_lstm_one_prediction.py:627`. `np.savetxt(f'...{trial_no}...')`; defined var is `n_trial`. Aborts the test/metric section. **Fix:** use `n_trial`.
- **M11 — `position_estimate` calls `load_split_data` with stale kwargs** — `position_estimate_anton.ipynb` cell 2. Passes `Ntrain/Nval` (ignored; defaults 1000/100 used) and unpacks 2 of a 5-tuple → silently trains on 1000 samples then `ValueError`. Also missing torch imports. **Fix:** current API `X_train,y_train,X_test,y_test,col_locations = load_split_data(..., N_train=..., N_test=...)`.
- **M12 — `StepLR(step_size=0.1)` float, stepped per batch** — `transformer_2_anton.ipynb` cell 0, `transformer_initial_anton.ipynb` cell 4. Collapses LR to ~0 within first batches. **Fix:** integer `step_size` in epochs, step once per epoch.
- **M13 — `calced/position` per-window absolute summed as inter-window delta** — `transformer_informed.py:299–300,311`. Each window's integration restarts at 0; summing endpoints loses velocity continuity. **Fix:** integrate v/p continuously across the full sequence for the dead-reckoning baseline.
- **M14 — Same-subject contiguous-tail validation** — `pred_pos_lstm.py:352–354`. `shuffle=False` last-20% from the same train dirs/subjects; val tail shares a boundary sample with train. **Fix:** hold out validation by subject/recording with a `≥seq_len` gap.

### LOW

- **L1 — `random_start` magic-number cap 20000, checks only first feature's length** — `utils.py:180–182`. Short recordings collapse to deterministic 0; long ones never sample the tail. **Fix:** derive bound from `len(stream)−N`; verify equal-length streams.
- **L2 — `split_data` is dead code with a non-shuffled positional cut** — `utils.py:394–399`. If used, leaky on overlapping windows. **Fix:** remove or add a `≥seq_len` guard gap.
- **L3 — `include_clusters`/`theta` mutate the caller's list** — `utils.py:147–148,351`. Repeated calls accumulate duplicate entries, breaking width accounting. **Fix:** local copies.
- **L4 — `vec_norm` returns `(1,1)` not a scalar** — `utils.py:449–450`. Fragile `if vec_norm(w)<eps` guards compare against an array. **Fix:** `float(np.linalg.norm(v.ravel()))`.
- **L5 — `factor=.002`/`a/100` "arbitrary" hacks masking upstream bugs** — `kalman_filter_anton.ipynb` cell 4. Remove once C1/C5 are fixed.
- **L6 — Train-loss print divided by `seq_len`** — `pred_pos_lstm.py:447,460`. 30× smaller than the objective, not comparable to the other scripts. **Fix:** print `running_loss/n_minibatches`.
- **L7 — Display sign-flip uses test GT to pick a global double-cover sign inside the per-axis loop** — `quaternions_loss_func.ipynb` cell 1 lines 44–45. Cosmetic but misleading. **Fix:** per-sample sign once, outside the loop.
- **L8 — LR 3e-6 / 1e-6 far too low, weight init commented out** — `transformer_informed.py:34,243`, EKF script 1e-6/1000 epochs. Model underfits toward the target mean. **Fix:** enable init, LR ~1e-3/1e-4, confirm train loss beats the variance-of-target baseline.

**Pervasive smells (not bugs but mandate the rebuild):** hardcoded absolute author paths (`C:\Users\Simon Andersen\...`, `/Users/antongolles/...`) in every file; massive copy-paste of `Theta`/`Omega`/`get_body2world_rot`/`Net`/`CustomLoss`/preprocessing across notebooks and `.py` files with divergent copies; author uncertainty left in comments (`# maybe a minus sign is needed`, `# they may not be in the right order`); `np.Inf` (removed in modern NumPy); `assert(include_mag is True & (...))` bitwise-`&` on bools; broken-syntax cells; magic numbers `dt=1/200`, `23`, `19+4`, `-10/-6/-3`.

---

## 3. Why the approach fundamentally underperforms (ATE ~30 m vs RoNIN ~5 m)

The ~6× benchmark gap is over-determined. Each of the following alone would inflate ATE; they stack:

1. **Wrong target frame ⇒ structural failure to generalize (the dominant cause).** Raw body-frame-derived features regressed to (effectively) global/absolute position couple the target to arbitrary device attachment **and** walking heading. The same physical "walk forward" produces different (input, target) pairs under every device orientation and compass heading, so the network must memorize all combinations — it cannot, and overfits to training headings (Woodman/RoNIN/IONet/EqNIO all flag this as *the* generalization killer). RoNIN's ~5 m comes precisely from regressing 2D motion in a **gravity-aligned, heading-agnostic frame** with **random-yaw augmentation**, which this project does neither of.

2. **The reconstruction formula drifts linearly even with a perfect model (C7/C8).** With `overlap=20, seq_len=173`, the audit verified a perfect predictor reconstructs `0,0,−19,−38,−57,…` — pure metric-induced linear drift. Over a multi-minute trajectory that alone is tens of meters of ATE before the model contributes anything.

3. **The physics prior is anti-informative (C1–C5, H1–H4).** The orientation used to rotate accel into the world frame is wrong by 400× / 2× and vertically inverted; the integration adds a meaningless `acc²` monotonic-drift term; it integrates dimensionless whitened acceleration with gravity whitened away. Classical strapdown already drifts **cubically** (gyro bias → linear tilt → gravity leakage `g·sin θ` → double-integrated → ~t³; Woodman: 0.05° tilt → 7.7 m in 30 s; stationary IMU → 152 m in 60 s). Feeding the network a *wrong* physics channel is worse than feeding none — it injects systematic, orientation-dependent error correlated across the window.

4. **The learning signal is largely disconnected (H13, C11, C13).** The "transformer" is a flattened-window MLP; the recurrent nets see length-1 sequences; the transformers have no positional encoding. The 200 Hz temporal structure that the velocity/displacement signal *is* never reaches an attention or recurrence mechanism. At LR 3e-6 with init disabled (L8), the model collapses toward the target mean.

5. **The reported number isn't even the RoNIN number (M1, H11).** ATE as `mean(‖·‖)` (not RMSE) and the meaningless RTE mean the headline figures are not comparable to published RoNIN/TLIO ATE/RTE in the first place — the "30 m" is computed by a different, leaky (C6, H8, H10) yardstick.

**Conclusion:** the gap is not a tuning or capacity problem. Fixing the bugs without changing the target formulation would still leave (1) unaddressed; changing only the formulation without fixing the reconstruction/metric would still mis-measure. The rebuild must address frame, target, reconstruction, physics, architecture, and metric together.

---

## 4. Literature-backed design for the clean rebuild

### 4.1 Target formulation (the single most important decision)

**Frame:** Rotate the raw accel+gyro window from the device frame into a **gravity-aligned, heading-agnostic frame (HACF)** before the network — Z aligned with gravity, horizontal yaw deliberately arbitrary (RoNIN; TLIO uses the orientation at the window *start*). Tilt (roll/pitch) is observable from gravity and must be pinned; **yaw about gravity is unobservable from accel+gyro alone** and must be left out of the target [RoNIN arXiv:1905.12853; Woodman UCAM-CL-TR-696; EqNIO arXiv:2408.06321].

**Output target (default):** regress a **2D horizontal velocity** `v = (P_i − P_{i−w}) / (w·dt)` over a 1 s window (`w=200` @200 Hz) in the HACF — the RoNIN ResNet target [RoNIN; Sachini/ronin `ronin_resnet.py`]. Discard the vertical (Z) component; evaluate in 2D. Do **not** regress global/absolute position or device-frame velocity.

**Alternative targets** (implement as configurable heads, in increasing sophistication):
- **TLIO** [arXiv:2007.01867]: 3D **displacement + diagonal log-std covariance** `Σ = diag(exp(2u))` over a 1 s window, gravity-aligned to window start; fuse into a stochastic-cloning EKF. Withhold initial velocity from the net (it learns a pure motion prior).
- **IONet** [arXiv:1802.02209]: **polar** `(Δl, Δψ)` = displacement magnitude + heading change, reconstructed by `x += Δl·cos ψ, y += Δl·sin ψ`. Inherently heading-invariant; good for an LSTM baseline. Treat each window independently ("integrator reset").

**Reconstruction:** integrate by cumulative sum scaled by dt: `pos[1:] = cumsum(pred_v · dt, axis=0) + pos[0]` [RoNIN `recon_traj_with_preds`]. The **same** operator must build both prediction and GT trajectories (fixes C7/C8). For displacement targets, `pos[i] = pred_displacement[i] + true_start_offset[i]`.

### 4.2 Architectures to implement

1. **RoNIN 1D ResNet-18 (primary, build first).** Channel-first input `[B, 6, 200]`; head `Conv1d(6,64,k=7,s=2,p=3)+BN+ReLU+MaxPool1d(3,2,1)`; 4 stages `BasicBlock1D` widths `[64,128,256,512]`, group sizes `[2,2,2,2]`; head `AdaptiveAvgPool1d(1) → Linear(512, 2)`. (Trust the released code: global average pool, not the "512-unit FC" the prose implies) [Sachini/ronin `model_resnet1d.py`].
2. **Transformer encoder (fair A/B baseline).** Per-timestep `Linear(6→d_model)`, **additive sinusoidal positional encoding** `[200, d_model]` (mandatory — fixes C13), `TransformerEncoderLayer` stack: `d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1, GELU, batch_first=True, norm_first=True`; reduce by **mean-pool over time** (or CLS token) → `Linear(d_model, 2)` [Zerveas KDD 2021; CTIN arXiv:2112.02143]. Never `nhead=1`; never flatten-then-MLP.
3. **RoNIN LSTM/TCN (optional).** 3×100 unidirectional LSTM (or TCN: 6 residual blocks `16/32/64/128/72/36`, kernel 3, RF 253) emitting per-frame velocity through a **non-trainable integration layer** ("latent velocity loss" — integrate, *then* supervise) [RoNIN `ronin_lstm_tcn.py`].
4. **TLIO ResNet + diagonal-covariance head + stochastic-cloning EKF (stretch goal).** Two FC heads on the ResNet trunk (displacement + log-std); 15-dim state `(R,v,p,b_g,b_a)` + cloned poses; measurement `h = R_γᵀ(p_j − p_i)` in the **yaw-only** local frame (extrinsic-XYZ Euler decomposition — getting this wrong reinjects unobservable yaw); χ² gate (11.345, 3 dof); inflate net covariance ×10 for overlapping-window correlation [TLIO; RNIN-VIO].
5. **(Lightweight)** IMUNet depthwise-separable MobileResNet blocks (ELU) as a drop-in efficient trunk if edge deployment matters [arXiv:2208.00068].

### 4.3 Loss

- **Position default:** plain **MSE** on the 2D velocity/displacement target. L1/Huber/LogCosh are within ±1% — don't expect gains [RoNIN; Keller et al. arXiv:2501.01327].
- **TLIO uncertainty path:** two-stage — ~10 epochs MSE warm-start, then Gaussian NLL `L = mean[0.5·log det Σ + 0.5·‖d−d̂‖²_{Σ⁻¹}]`, `Σ=diag(exp(2u))`. NLL from scratch does **not** converge; regress **log-std**, never std/variance [TLIO]. Validate covariance calibration (3σ coverage, χ²/Mahalanobis) before trusting the EKF.
- **Orientation (if predicted):** report the geodesic angle `θ = 2·arccos(|⟨q_pred,q_gt⟩|)` in degrees; **train** with a smooth surrogate — chordal `‖R_pred−R_gt‖_F²` (with 6D/SVD-orthogonalized output) or `1 − |⟨q_pred,q_gt⟩|`. If you use a log-map/arccos loss, clamp the argument to `[−1+ε, 1−ε]` in float64 (fixes H6) [Huynh 2009; RIANN arXiv:2104.07391; Hitchhiker's guide arXiv:2404.11735]. Always resolve double-cover **per sample** (fixes C10). Don't regress raw quaternions/Euler as the geometric output — they are discontinuous; prefer 6D-GSO or 9D-SVD [Zhou et al. CVPR 2019].
- **Gyro front-end (optional, Brossard):** integrated-increment loss `L = L_16 + L_32`, `L_j = Σ ρ(log(dR_{i,i+j}·dR̂ᵀ))`, Huber `δ=0.005` — supervises relative increments at 12.5/6.25 Hz so it needs no full-rate GT [arXiv:2002.10718].

### 4.4 Metrics (precise, RoNIN-compatible)

Compute **per recording, then average** (report median/std/CDF); hold out **entire subjects/recordings** for test [RoNIN; survey arXiv:2303.03757].

- **ATE** = `sqrt(mean((est − gt)²))` over all matched 2D timesteps — a **direct positional RMSE, no SE(2)/Sim(3) alignment** [RoNIN `metric.py`]. (Fixes M1.)
- **RTE@1min** = mean over a sliding `Δ = pred_per_min = 12000` frame (@200 Hz) window of the relative-displacement-error RMSE: `err = (est[Δ:] − est[:−Δ]) − (gt[Δ:] − gt[:−Δ]); rte = sqrt(mean(err²))`. For sequences shorter than 1 min, `Δ = N−1` and scale by `12000/N` [RoNIN `metric.py`]. (Fixes H11/M2.)
- Also log raw per-window velocity/displacement MSE. **MSE-lower ≠ ATE-lower** — tune on ATE/drift, not MSE [TLIO].

### 4.5 Normalization

- **Sensor calibration + gravity-frame rotation is the preprocessing; do NOT z-score the IMU channels.** RoNIN/TLIO do not normalize the rotated channels, and Keller et al. found z-score/robust normalization *degraded* neural inertial regression by −308% to −907% — absolute accel/gyro magnitudes carry the motion signal [Keller arXiv:2501.01327; RoNIN code].
- If any scaler is used, **fit on train only, `transform` (never `fit_transform`) on val/test** (fixes C6) [Brownlee data-leakage].

### 4.6 Augmentation (every batch, by impact)

1. **Random yaw rotation about gravity** applied to **both** the HACF IMU window and the target velocity (≈+7% avg; mandatory for heading invariance without an equivariant net — fixes the §3.1 root cause) [RoNIN; Keller; EqNIO].
2. **Additive Gaussian noise** ≈0.1 m/s² accel, ≈0.001 rad/s gyro (≈+6%).
3. **Bias offsets** accel `U(±0.2 m/s²)`, gyro `U(±0.05 rad/s)` (≈+2%) [TLIO; Keller].
4. **Gravity-tilt** `[0,5]°` about a random horizontal axis — bridges the gap to the noisier runtime orientation [TLIO].
5. **No magnetometer** in the position regressor (indoor magnetic disturbance) [RoNIN; TLIO]. For equivariant designs, treat gyro as a **pseudovector** `ω' = det(R)·R·ω` [EqNIO].

---

## 5. Proposed module layout + test strategy

A flat `playground_notebooks/` of divergent copy-pasted scripts is the root organizational cause of the divergent-`Theta` class of bugs. Replace with one installable package, one definition per concept, hard convention assertions, and a test suite that proves correctness on synthetic data (the real RoNIN HDF5 is not in the repo).

### 5.1 Package layout (`ninav/`)

```
ninav/
  __init__.py
  config.py            # dataclasses: WindowCfg(seq_len=200, stride, fs=200), TrainCfg, paths from env/CLI — NO hardcoded absolute paths
  geometry/
    quaternion.py      # ONE convention (scalar-last [x,y,z,w]), asserted. quat_mul (Hamilton), quat_conj,
                       #   quat_normalize, quat_to_rotmat, rotmat_to_quat (Shepperd, clamp sqrt≥0 — fixes H2),
                       #   quat_exp/quat_log. (Replaces utils.py 488-511)
    propagate.py       # ONE closed-form gyro propagator: theta_update(w, dt) = cos(|w|dt/2)·I +
                       #   (sin(|w|dt/2)/|w|)·Omega_full(w), |w|→0 guard → I (fixes C1/H1/H12). NO factor=.002.
    frames.py          # body2world_rot(mag, acc): DOWN=-a/|a| (fixes C2), EAST=cross(DOWN,mag),
                       #   NORTH=cross(EAST,DOWN), assert det=+1. gravity_align(window, orientation) → HACF.
  data/
    ronin_hdf5.py      # load one recording (gyro/acc/mag/tango_pos/tango_ori) from HDF5; per-recording, no vstack-across-recordings.
    windowing.py       # create_windows(stream, seq_len, stride) PER recording (fixes H16); explicit `stride`
                       #   not inverted `overlap` (fixes M3); no in-place mutation (fixes M4/L3).
    targets.py         # velocity_target, displacement_target, polar_target — all relative, per-window, identical train/test (fixes C9).
    splits.py          # split BY subject/recording with a ≥seq_len guard gap (fixes M14/L2/H15).
    normalize.py       # fit-on-train transform-on-test scaler (fixes C6); default: identity (no IMU z-score).
    augment.py         # random_yaw (input+target together), gaussian_noise, bias_offset, gravity_tilt (§4.6).
  models/
    resnet1d.py        # RoNIN ResNet-18 1D, channel-first [B,6,L] (fixes layout pitfall).
    transformer.py     # input embed + sinusoidal PE + encoder + mean-pool head; nhead=8, batch_first=True (fixes C13).
    lstm_tcn.py        # RoNIN LSTM/TCN + non-trainable integration layer (latent velocity loss).
    tlio_head.py       # displacement + diagonal log-std covariance head.
  losses/
    regression.py      # MSE; Gaussian NLL (log-std).
    rotation.py        # geodesic metric (degrees, float64 clamped); chordal & 1-|cos| surrogates; per-sample double-cover (fixes C10/H6).
  filters/
    ahrs.py            # Madgwick / Mahony / quaternion-EKF — run PER split, PER recording, q0 = unit quaternion (fixes H7/H8/H9).
    sc_ekf.py          # TLIO stochastic-cloning EKF; yaw-only measurement frame; χ² gate; ×10 covariance inflation (fixes C14).
  reconstruct.py       # ONE reconstruct_trajectory() used for pred AND gt (fixes C7/C8/C12/M8/M13).
  metrics.py           # compute_ate_rte() mirroring RoNIN metric.py exactly (fixes M1/M2/H11).
  train.py             # training loop: shuffle=True, weight init, LR ~1e-3/1e-4, two-stage for TLIO (fixes M9/L8); batched inference (fixes M5).
  cli.py               # entrypoints; paths via args/env.
tests/
  ...                  # see 5.2
```

### 5.2 Test strategy (synthetic-data-first, since real HDF5 is absent)

**Geometry / propagation (analytic ground truth — catches C1, C2, C3, H1, H2, H12):**
- `test_quaternion_convention`: round-trip `quat_to_rotmat ∘ rotmat_to_quat == I`; cross-check against `scipy.spatial.transform.Rotation` with an **explicit index permutation**, asserting the mismatch is caught (this is the test the original code never had).
- `test_theta_orthonormal`: for random `w`, assert `Theta@Thetaᵀ ≈ I` and `det ≈ 1` to 1e-6 (the original failed at ~2·I / ~1.98·I).
- `test_theta_constant_rate`: integrate a constant `w` for `T` seconds via `theta_update`; assert recovered angle `== |w|·T` (catches the 400× and 2× errors).
- `test_theta_zero_gyro`: `w=[0,0,0]` returns `I`, **no NaN** (catches H1).
- `test_rotmat_to_quat_180deg`: rotation near 180° (trace→−1) returns a finite unit quaternion (catches H2).
- `test_body2world_static`: flat phone `a=[0,0,+9.81]` ⇒ `DOWN` column ≈ `[0,0,−1]` in world, `det(R)=+1` (catches C2).

**Windowing / targets / splits (catches H16, C9, M3, M4, M14):**
- `test_windows_per_recording`: two synthetic recordings concatenated never produce a window spanning the seam.
- `test_stride_semantics`: `stride=seq_len` ⇒ disjoint tiling; assert `overlap`-style inversion is gone.
- `test_target_relative_consistency`: train and test produce identical per-window rebasing.
- `test_no_input_mutation`: target builder does not mutate its input array.
- `test_subject_split_gap`: no window shared across train/val/test; ≥`seq_len` gap.

**Normalization / leakage (catches C6):**
- `test_scaler_transform_not_fit`: passing a fitted scaler calls `transform`, and test mean is **not** ~0 (proves train stats are reused, not re-fit).

**Reconstruction + metrics — the load-bearing tests (catches C7, C8, C12, M1, M2, H11):**
- `test_perfect_model_zero_ate`: synthetic trajectory → compute velocity targets → reconstruct → assert ATE ≈ 0 for **any** stride/overlap (the original drifted linearly; this is the single most diagnostic test).
- `test_recon_operator_identical`: pred and GT use the same `reconstruct_trajectory`.
- `test_ate_is_rmse`: distances `{5,0,10}` → ATE = 6.45 (RMSE), not 5.0 (mean).
- `test_rte_matches_ronin`: compare `compute_ate_rte` against a transcription of RoNIN `metric.py` on a fixed synthetic trajectory, including the short-sequence `Δ=N−1, ×12000/N` branch and the `Nmins=0` guard.

**Physics integration (catches C4, C5, H4, M13):**
- `test_double_integration`: constant accel `a` over `T` ⇒ `p = ½·a·T²` (analytic), **no `acc²` term** (catches C4).
- `test_integration_on_raw_units`: integrating whitened (mean-0) accel yields ≈0 displacement, demonstrating C5; the pipeline must integrate raw m/s².
- `test_gravity_removal`: static window with gravity present integrates to ≈0 only after world-frame gravity subtraction (catches H4).

**Models (catches C11, C13, H13):**
- `test_resnet_layout`: input `[B,6,200]` → output `[B,2]`; assert a `[B,200,6]` input is rejected/asserted, not silently consumed.
- `test_transformer_positional`: **shuffling the time axis changes the output** (proves PE is wired — directly catches C13).
- `test_transformer_nhead`: `d_model % nhead == 0`, `nhead ≥ 4`.
- `test_seq_dim_real`: windowed tensor fed to LSTM/transformer has `seq_len > 1` (catches C11).

**Losses (catches C10, H5, H6):**
- `test_double_cover_per_sample`: a batch where half the samples need `+q` and half need `−q`; assert the loss is small for all (whole-batch single-sign loss fails this).
- `test_arccos_grad_finite`: `q_pred == q_gt` → loss 0 **and finite gradient** (catches H6).
- `test_unit_norm_output`: model `forward` returns unit quaternions (catches H5).

**Filters (catches C14, H7, H8):**
- `test_ekf_nonzero_innovation`: with a real measurement differing from prediction, the state update is non-trivial (catches C14).
- `test_q0_is_quaternion`: filter init rejects a 3-vector seed (catches H7).
- `test_filter_per_split`: train and test filter outputs are independent (no shared recursive state — catches H8).

**End-to-end smoke (CPU, tiny):** generate a short synthetic gravity-aligned trajectory with known velocity, run ResNet for a few steps, assert train loss decreases below the variance-of-target baseline (catches L8) and reconstructed ATE is finite and decreasing.

**Optional real-data conformance:** if a single RoNIN HDF5 recording is later added under a gitignored `data/`, add a slow-marked test asserting `load_split_data`-equivalent shapes and that ATE/RTE land in the published RoNIN ballpark (~5 m) for the released ResNet weights — guarded by a skip when the file is absent.

---

**Relevant absolute paths (audited originals, to be replaced — not edited in place):**
`/Users/tonton/agent-assistant/repos/inertial_navigation_transformer/utils.py`,
`/Users/tonton/agent-assistant/repos/inertial_navigation_transformer/playground_notebooks/transformer_informed.py`,
`/Users/tonton/agent-assistant/repos/inertial_navigation_transformer/playground_notebooks/Linear_informed_with_EKF.py`,
`/Users/tonton/agent-assistant/repos/inertial_navigation_transformer/playground_notebooks/pred_pos_lstm.py`,
`.../pred_pos_lstm_one_prediction.py`, `.../pred_pos_lstm_many_predictions.py`,
`.../pred_ori_lstm.py`, `.../pred_ori_2_lstm.py`,
`.../kalman_filter_anton.ipynb`, `.../position_estimate_anton.ipynb`,
`.../transformer_initial_anton.ipynb`, `.../transformer_2_anton.ipynb`,
`.../quaternions_LSTM_GRU_RNN.ipynb`, `.../quaternions_loss_func.ipynb`,
`.../quaternion_update_anton.ipynb`, `.../quaternion_update_anton_new.ipynb`,
`.../bodyFrame_to_worldFrame.ipynb`, `.../Acce_Ori_to_Pos.ipynb`.

The clean rebuild lives in a new `ninav/` package (§5.1) at the repo root; the originals are retained for the bug report's provenance only.