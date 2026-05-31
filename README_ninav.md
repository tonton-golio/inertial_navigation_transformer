# ninav — neural inertial navigation, rebuilt

`ninav` is a clean, tested reimplementation of the neural inertial-navigation
project in this repository. The original course code (`utils.py` +
`playground_notebooks/`) tried to compress a high-rate IMU window straight into a
global-frame position, and did so through a stack of independent physics, frame,
metric, and architecture bugs. `ninav` discards that pipeline and rebuilds the
problem the way the modern literature (RoNIN, TLIO, IONet) actually solves it:
**regress a heading-agnostic, gravity-aligned motion quantity per window, then
integrate**.

The full forensic catalogue of what was wrong with the original code — every bug,
its call sites, and why the approach scored ATE ~30 m against RoNIN's ~5 m — is in
[`REVIEW.md`](REVIEW.md). This file is just the operating manual for the rebuild.

## Why the original approach failed (and what changed)

The single dominant flaw: the old code regressed **absolute / global-frame
position** from raw body-frame features, then stitched windows together with an
ad-hoc `cumsum` that drifted *even with a perfect model*. Regressing global
position couples the target to both the device's mounting orientation and the
walking heading, so the network has to memorize every (orientation × heading)
combination — it can't, and it overfits.

`ninav` does the opposite, following RoNIN/TLIO:

- **Heading-agnostic, gravity-aligned frame.** Each IMU window is rotated so Z is
  aligned with gravity (tilt is observable; yaw about gravity is *not* observable
  from accel+gyro alone and is deliberately left out of the target).
- **Regress a local motion quantity, not position.** The default target is the 2D
  horizontal **velocity** over the window. Optional heads regress 3D
  **displacement** (with a TLIO-style diagonal-covariance variant) or **polar**
  `(Δl, Δψ)` displacement.
- **One reconstruction operator for prediction *and* ground truth.** A single
  `reconstruct_trajectory()` integrates velocity (or places displacements at their
  true window-start offsets), applied identically to the estimate and the GT, so a
  perfect prediction reconstructs to ~0 error for any stride.
- **RoNIN-correct metrics.** ATE is a positional RMSE (not `mean(‖·‖)`); RTE is the
  RoNIN 1-minute sliding relative-displacement RMSE with the short-sequence
  scaling — not the original's `ATE × fraction-of-trajectory`.
- **Real sequence models.** A 1D ResNet-18 and a Transformer encoder *with*
  positional encoding (the original "transformer" was a flatten-then-MLP with no
  positional encoding and `nhead=1`).
- **No data leakage.** Feature scaling is fit on the train split only; windows are
  built per recording (never across recording seams); splits are by recording.

## Package layout

```
ninav/
  config.py            WindowCfg / TrainCfg / AugmentCfg / Paths (env-driven, no hardcoded paths)
  geometry/
    quaternion.py      ONE convention (scalar-last [x,y,z,w]); Hamilton mul, exp/log, Shepperd rotmat<->quat
    propagate.py       ONE closed-form gyro propagator (correct exp-map, |w|->0 guard); integrate_gyro
    frames.py          body->world rotation (DOWN = -a/|a|), gravity_align -> HACF, yaw_rotation
  data/
    synthetic.py       physically self-consistent forward model + Recording dataclass
    ronin_hdf5.py      load real RoNIN data.hdf5 (converts scalar-first tango_ori -> scalar-last at the boundary)
    windowing.py       per-recording windows; explicit stride (not the old inverted "overlap")
    targets.py         velocity / displacement / polar targets (relative, identical train & test)
    splits.py          split BY recording with a guard gap
    normalize.py       FeatureScaler: fit-on-train, transform-on-test (default: identity / no z-score)
    augment.py         random yaw (input+target together), gaussian noise, bias offset, gravity tilt
    dataset.py         build_dataset / prepare_recording / WindowDataset
  models/
    resnet1d.py        RoNIN 1D ResNet-18, channel-first [B, 6, L] -> [B, out_dim]
    transformer.py     input embed + sinusoidal positional encoding + encoder + mean-pool head
    lstm_tcn.py        LSTM and TCN regressors
    tlio_head.py       ResNet trunk + (mean, log_std) diagonal-covariance head
    __init__.py        build_model(name, ...) factory + MODEL_REGISTRY
  losses/
    regression.py      MSE; Gaussian NLL (regresses log-std)
    rotation.py        geodesic metric + smooth surrogates; per-sample double-cover
  filters/
    ahrs.py            Madgwick / Mahony / quaternion-EKF (per split, per recording)
    sc_ekf.py          TLIO-style stochastic-cloning EKF (yaw-only measurement frame)
  reconstruct.py       ONE reconstruct_trajectory() for est AND gt
  metrics.py           compute_ate_rte() mirroring RoNIN metric.py
  train.py             training-loop helpers
  cli.py               `python -m ninav.cli {train,eval}` entrypoints
tests/                 synthetic-data test suite (see "Running tests")
```

Design rationale and literature citations for each module are in
[`REVIEW.md`](REVIEW.md) and `docs/_design_brief.md`.

## Setup (uv)

The project targets Python ≥ 3.10 (developed on 3.13). A virtual environment with
all dependencies already lives at `.venv/`. To recreate it with [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[test]"
```

This installs the runtime deps (numpy, scipy, scikit-learn, torch, h5py,
matplotlib) plus pytest. Exact pinned versions are in `requirements.txt`. Pure CPU
— no GPU required.

## Running tests

```bash
.venv/bin/python -m pytest -q
```

The suite is synthetic-data-first: the synthetic generator in
`ninav/data/synthetic.py` is the *forward model* of the navigation problem, so the
pipeline (gravity removal → frame rotation → integration → velocity targets →
reconstruction → metrics) can be inverted back to known ground truth and asserted
to ~0 error. The load-bearing test is "perfect model ⇒ ATE ≈ 0 for any stride",
which the original code failed by construction.

## Train / evaluate on synthetic data

No external data is required: with no `--data`, the CLI generates a fresh
synthetic dataset, so the whole thing is runnable out of the box.

```bash
# Train the RoNIN 1D ResNet on synthetic data and evaluate the held-out split.
.venv/bin/python -m ninav.cli train \
  --model resnet1d --target velocity \
  --synthetic-recordings 8 --epochs 100 \
  --out runs/resnet_synth
```

This writes `checkpoint.pt`, `metrics.json` (ATE/RTE plus full train/val history),
and `trajectory.png` (estimate vs ground truth) under `--out`. Evaluate a saved
checkpoint with:

```bash
.venv/bin/python -m ninav.cli eval \
  --ckpt runs/resnet_synth/checkpoint.pt \
  --out runs/resnet_synth_eval
```

A quick smoke run (a few epochs, fewer recordings) finishes in seconds on CPU:

```bash
.venv/bin/python -m ninav.cli train --model resnet1d --epochs 3 \
  --synthetic-recordings 4 --out runs/smoke
```

## Pointing at real RoNIN data

Download RoNIN from <https://ronin.cs.sfu.ca/> (reference code:
<https://github.com/Sachini/ronin>). Each recording is a directory containing a
`data.hdf5`:

```
<ronin_root>/
  a000_1/ data.hdf5
  a000_2/ data.hdf5
  ...
```

Point `ninav` at the root with `--data` (or the `NINAV_DATA` environment variable;
`--data` wins if both are set):

```bash
.venv/bin/python -m ninav.cli train --data /path/to/ronin_root --model resnet1d
# or
export NINAV_DATA=/path/to/ronin_root
.venv/bin/python -m ninav.cli train --model resnet1d
```

The loader reads `synced/gyro`, `synced/acce`, `synced/magnet` (optional),
`pose/tango_pos`, and `pose/tango_ori`. RoNIN stores `tango_ori` as a
**scalar-first** `[w,x,y,z]` quaternion; the loader converts it to the repo's
**scalar-last** `[x,y,z,w]` convention exactly once at the boundary (this was bug
C3 in the original code). Output directory follows the same `--out` / `NINAV_OUT`
precedence.

## Implemented models

Selected with `--model`; constructed by name via `ninav.models.build_model`. All
take channel-first input `[B, 6, L]` (3 world-frame accel + 3 gyro) and return
`[B, out_dim]`.

| `--model`     | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `resnet1d`    | RoNIN 1D ResNet-18 (primary baseline).                                       |
| `transformer` | Transformer encoder with sinusoidal positional encoding + mean-pool head.    |
| `lstm`        | LSTM regressor.                                                              |
| `tcn`         | Temporal convolutional network regressor.                                    |
| `tlio`        | ResNet trunk + diagonal log-std covariance head; `forward` returns `(mean, log_std)` and trains with Gaussian NLL after an MSE warm-up. |

Targets (`--target`): `velocity` (default, 2D), `displacement` (2D), `polar`
(2D `(Δl, Δψ)`; trajectory metrics are skipped for polar since it has no spine
reconstruction operator). Train-time augmentation (random yaw about gravity,
Gaussian noise, bias offsets, gravity tilt) is on by default — disable with
`--no-augment`.

## Metrics

`ninav.metrics.compute_ate_rte`, mirroring RoNIN's `metric.py`:

- **ATE** — absolute trajectory error, the RMSE of the per-point 2D positional
  error: `sqrt(mean_t ‖est_t − gt_t‖²)`. No SE(2)/Sim(3) alignment.
- **RTE** — relative trajectory error over a fixed 1-minute lag `d`:
  `sqrt(mean_t ‖(est_{t+d} − est_t) − (gt_{t+d} − gt_t)‖²)`. For trajectories
  shorter than a minute, `d = T−1` scaled by `pred_per_min / T`.

Metrics are computed per recording and aggregated (mean / median / std) across
recordings by `aggregate_metrics`, and written to `metrics.json`.

## See also

- [`REVIEW.md`](REVIEW.md) — the full bug report on the original code (the rebuild's motivation).
- `docs/_design_brief.md` — consolidated audit findings + literature-backed design rationale.
- Original (buggy) code retained for provenance: `utils.py`, `playground_notebooks/`.
