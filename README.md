# Neural Inertial Navigation

Estimating a device's **position and orientation from its IMU alone** — the
accelerometer, gyroscope and magnetometer in a phone — with neural networks
instead of hand-tuned filters. Trained and evaluated on the
[RoNIN](https://ronin.cs.sfu.ca/) dataset (200 Hz, pedestrian motion).

<figure>
  <img src="https://github.com/tonton-golio/inertial_navigation_transformer/blob/main/assets/IMU_interpretation_flowchart.png?raw=true" alt="IMU interpretation flowchart" width="640">
</figure>

> **2026 update.** This started as a university course project (original code in
> [`utils.py`](utils.py) and [`playground_notebooks/`](playground_notebooks/)). A
> later review found the pipeline that "compresses" an IMU window into a position
> was broken at nearly every stage — wrong quaternion math, invalid double
> integration, a reconstruction that drifts *even with a perfect model*, data
> leakage, and sequence models that never saw the time axis. The repo now ships a
> clean, tested rebuild in [`ninav/`](ninav/) and a full bug report in
> [`REVIEW.md`](REVIEW.md).

## Why the original approach failed

The original code regressed **global-frame position** from raw, body-frame
features, then stitched windows with an ad-hoc `cumsum`. That couples the target
to both the device mounting and the walking heading, so the network must memorize
every orientation × heading combination — it can't, and it overfits (ATE ~30 m vs
RoNIN's ~5 m). `REVIEW.md` catalogues all 74 findings; three are confirmed
numerically (the quaternion update is 400× too large, the `cumsum(acc²)`
integration term is dimensionally invalid, and the overlap-`cumsum`
reconstruction drifts a perfect model to hundreds of metres).

## What `ninav/` does instead

Follows the modern literature (RoNIN, TLIO, IONet): **regress a heading-agnostic,
gravity-aligned 2D velocity per window, then integrate.**

- **One correct geometry stack** — single scalar-last quaternion convention, an
  exp-map gyro propagator, gravity-aligned frames (`DOWN = −a/‖a‖`).
- **Correct physics & metrics** — pure double integration on gravity-compensated
  acceleration; one shared reconstruction operator for prediction *and* ground
  truth; RoNIN-faithful ATE/RTE.
- **Real sequence models** — RoNIN 1D ResNet-18, a Transformer encoder *with*
  positional encoding, LSTM, TCN, and a TLIO mean+log-std head.
- **No leakage** — per-recording windowing, by-recording splits, fit-on-train
  scaling, random-yaw augmentation.
- **91 passing tests**, synthetic-data-first (the load-bearing one: *a perfect
  model reconstructs to ATE ≈ 0 at any stride* — which the original failed).

## Quickstart

```bash
# environment (uv); CPU-only, no GPU needed
uv venv .venv && uv pip install --python .venv/bin/python -e ".[test]"

# run the test suite
.venv/bin/python -m pytest -q

# train + evaluate on synthetic data (no download required)
.venv/bin/python -m ninav.cli train --model resnet1d --synthetic-recordings 8 \
  --epochs 100 --out runs/resnet_synth
```

Point it at real RoNIN data (download from <https://ronin.cs.sfu.ca/>; one
`data.hdf5` per recording directory):

```bash
.venv/bin/python -m ninav.cli train --data /path/to/ronin_root --model resnet1d
```

`--model` ∈ `{resnet1d, transformer, lstm, tcn, tlio}`, `--target` ∈
`{velocity, displacement, polar}`. Outputs a checkpoint, `metrics.json`
(ATE/RTE + history), and a trajectory plot.

## Layout

```
ninav/        clean rebuild — geometry · data · models · losses · filters · train · cli
tests/        91 synthetic-data tests
REVIEW.md     full bug report on the original code (the rebuild's motivation)
docs/         consolidated audit findings + literature-backed design rationale
utils.py      original course code (kept for provenance)
playground_notebooks/   original experiments (kept for provenance)
```

## Original project

By S. G. Andersen, M. Bach, A. G. Golles, A. A. Sousa-Poza & K. Tavanxhi.
See the writeup
[`AntonSimonAdrianMichaelChris_NetworkBasedInertialNavigation.pdf`](AntonSimonAdrianMichaelChris_NetworkBasedInertialNavigation.pdf)
and [`Presentation.pdf`](Presentation.pdf). Key references: RoNIN
([arXiv:1905.12853](https://arxiv.org/abs/1905.12853)), TLIO
([arXiv:2007.01867](https://arxiv.org/abs/2007.01867)), IONet
([arXiv:1802.02209](https://arxiv.org/abs/1802.02209)).
