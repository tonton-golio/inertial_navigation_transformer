"""Command-line interface for ninav: train and evaluate neural inertial models.

This module is the thin orchestration layer over the (already-proven) numeric
spine. It owns NO numeric logic of its own -- every transform, target, loss,
metric and reconstruction call routes into ``ninav.*`` spine modules.

Subcommands
-----------
``train``
    Build a dataset (real RoNIN folder via ``--data``, else synthetic), split it
    by recording, train one of the registered models, then evaluate on the held
    out split. Saves ``checkpoint.pt``, ``metrics.json`` and ``trajectory.png``
    under ``--out``.
``eval``
    Load a checkpoint produced by ``train`` and evaluate it on a dataset
    (real or synthetic), writing ``metrics.json`` and ``trajectory.png``.

No filesystem paths are hardcoded: ``--data`` / ``--out`` come from the CLI, and
fall back to the env-driven :class:`ninav.config.Paths` (``NINAV_DATA`` /
``NINAV_OUT``). The whole thing runs on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import AugmentCfg, Paths, TrainCfg, WindowCfg
from .data.dataset import WindowDataset, build_dataset, prepare_recording
from .data.normalize import FeatureScaler
from .data.splits import split_recordings
from .data.synthetic import generate_dataset
from .losses.regression import gaussian_nll_loss, mse_loss
from .metrics import aggregate_metrics, compute_ate_rte
from .models import build_model
from .reconstruct import reconstruct_trajectory

# Target kind -> trajectory reconstruction mode + output dimensionality.
# Polar targets have no spine reconstruction operator, so trajectory metrics are
# skipped for them (training/val loss is still reported).
_TARGET_RECON = {"velocity": "velocity", "displacement": "displacement"}
_TARGET_DIMS = {"velocity": 2, "displacement": 2, "polar": 2}


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def _load_recordings(data_dir: str | None, *, n_synthetic: int, fs: float,
                     seed: int):
    """Return ``(recordings, source_label)``.

    A real RoNIN folder if ``data_dir`` is given and exists, otherwise a fresh
    synthetic dataset (so the CLI is runnable with zero external data).
    """
    if data_dir:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"--data directory not found: {data_dir}")
        # Local import: h5py is only needed for real-data ingestion.
        from .data.ronin_hdf5 import load_ronin_folder
        recs = load_ronin_folder(data_dir)
        if not recs:
            raise ValueError(
                f"no RoNIN recordings (subdir/data.hdf5) found under {data_dir}")
        return recs, f"ronin:{data_dir}"
    recs = generate_dataset(n_recordings=n_synthetic, fs=fs, seed=seed)
    return recs, f"synthetic:n={n_synthetic}"


def _split_recordings(recordings, val_fraction: float, test_fraction: float,
                      seed: int):
    """Split a recording list into ``(train, val, test)`` lists by recording id."""
    by_id = {r.rec_id: r for r in recordings}
    split = split_recordings(list(by_id), val_fraction=val_fraction,
                             test_fraction=test_fraction, seed=seed)
    pick = lambda ids: [by_id[i] for i in ids]  # noqa: E731
    return pick(split["train"]), pick(split["val"]), pick(split["test"])


# --------------------------------------------------------------------------- #
# Model forward helpers (TLIO returns a tuple)
# --------------------------------------------------------------------------- #
def _forward_mean(model, x):
    """Return ``(mean, log_std_or_None)`` for any model in the registry."""
    out = model(x)
    if isinstance(out, tuple):          # TLIONet -> (mean, log_std)
        return out[0], out[1]
    return out, None


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _train_loop(model, train_loader, val_loader, train_cfg: TrainCfg,
                *, is_probabilistic: bool, mse_warmup: int | None = None):
    # Honour the configured warm-up so the persisted TrainCfg.nll_warmup matches
    # the actual MSE->NLL switch epoch (no provenance mismatch).
    if mse_warmup is None:
        mse_warmup = train_cfg.nll_warmup
    """A compact RoNIN/TLIO-style training loop. Returns a history dict."""
    device = torch.device(train_cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(train_cfg.epochs):
        model.train()
        running, n = 0.0, 0
        # TLIO warm-up: pure MSE on the mean for the first few epochs, then NLL.
        use_nll = is_probabilistic and epoch >= mse_warmup
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            mean, log_std = _forward_mean(model, x)
            if use_nll and log_std is not None:
                loss = gaussian_nll_loss(mean, y, log_std)
            else:
                loss = mse_loss(mean, y)
            loss.backward()
            if train_cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_cfg.grad_clip)
            opt.step()
            bs = x.shape[0]
            running += float(loss.item()) * bs
            n += bs
        train_loss = running / max(n, 1)

        val_loss = _eval_loss(model, val_loader, device) if val_loader else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[epoch {epoch + 1}/{train_cfg.epochs}] "
              f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
              f"{' (nll)' if use_nll else ''}", flush=True)
    return history


@torch.no_grad()
def _eval_loss(model, loader, device) -> float:
    """Mean MSE on the mean prediction over ``loader`` (comparable across models)."""
    model.eval()
    running, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mean, _ = _forward_mean(model, x)
        running += float(mse_loss(mean, y).item()) * x.shape[0]
        n += x.shape[0]
    return running / max(n, 1)


# --------------------------------------------------------------------------- #
# Evaluation: predict windows -> reconstruct trajectory -> ATE/RTE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _predict_windows(model, inputs: np.ndarray, device, batch_size: int) -> np.ndarray:
    """Run the model over ``inputs`` ``(N, L, 6)`` and return means ``(N, dims)``."""
    model.eval()
    preds = []
    for i in range(0, inputs.shape[0], batch_size):
        chunk = inputs[i:i + batch_size]
        # (N, L, 6) -> (N, 6, L) channel-first, matching the model contract.
        x = torch.from_numpy(
            np.ascontiguousarray(chunk.transpose(0, 2, 1), dtype=np.float32)
        ).to(device)
        mean, _ = _forward_mean(model, x)
        preds.append(mean.cpu().numpy())
    if not preds:
        return np.zeros((0, model.out_dim), dtype=np.float64)
    return np.concatenate(preds, axis=0).astype(np.float64)


def _evaluate(model, recordings, window_cfg: WindowCfg, target_kind: str,
              orientation: str, dims: int, device, batch_size: int):
    """Per-recording reconstruction + ATE/RTE. Returns ``(agg, per_rec, traj)``.

    ``traj`` is ``(rec_id, est_xy, gt_xy)`` for the first reconstructable
    recording, used for the trajectory plot. Reconstruction uses the SAME
    operator for est and gt (a perfect model -> ~0 error).
    """
    recon_mode = _TARGET_RECON.get(target_kind)
    per_rec = []
    first_traj = None
    for rec in recordings:
        prep = prepare_recording(rec, window_cfg, target_kind, orientation, dims)
        if prep["input"].shape[0] == 0:
            continue
        pred = _predict_windows(model, prep["input"], device, batch_size)
        if recon_mode is None:
            # Polar (no spine reconstruction): skip trajectory metrics.
            continue
        kwargs = {}
        if recon_mode == "velocity":
            kwargs = {"dt": window_cfg.window_dt, "p0": prep["p_start"][0]}
        elif recon_mode == "displacement":
            kwargs = {"start_offsets": prep["p_start"]}
        est = reconstruct_trajectory(pred, mode=recon_mode, **kwargs)
        gt = reconstruct_trajectory(prep["target"], mode=recon_mode, **kwargs)
        ate, rte = compute_ate_rte(est, gt, pred_per_min=window_cfg.pred_per_min)
        per_rec.append({"rec_id": rec.rec_id, "ate": ate, "rte": rte})
        if first_traj is None:
            first_traj = (rec.rec_id, est, gt)

    agg = (aggregate_metrics([(r["ate"], r["rte"]) for r in per_rec])
           if per_rec else {"n": 0})
    return agg, per_rec, first_traj


# --------------------------------------------------------------------------- #
# Output artifacts
# --------------------------------------------------------------------------- #
def _save_trajectory_plot(traj, out_path: str, title: str) -> bool:
    """Plot estimated vs ground-truth XY (matplotlib Agg). Returns True if drawn."""
    if traj is None:
        return False
    import matplotlib
    matplotlib.use("Agg")  # headless backend; no display required
    import matplotlib.pyplot as plt

    rec_id, est, gt = traj
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt[:, 0], gt[:, 1], "-", color="0.4", label="ground truth")
    ax.plot(est[:, 0], est[:, 1], "-", color="tab:red", label="estimate")
    ax.scatter([gt[0, 0]], [gt[0, 1]], c="k", marker="o", zorder=5, label="start")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title}\n{rec_id}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _write_metrics(out_dir: str, payload: dict) -> str:
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=float)
    return path


# --------------------------------------------------------------------------- #
# Subcommand: train
# --------------------------------------------------------------------------- #
def _cmd_train(args: argparse.Namespace) -> int:
    paths = Paths()
    data_dir = args.data if args.data is not None else (paths.data_root or None)
    out_dir = args.out if args.out is not None else paths.out_dir
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    window_cfg = WindowCfg(fs=args.fs, window_len=args.window_len, stride=args.stride)
    train_cfg = TrainCfg(
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, val_fraction=args.val_fraction,
        seed=args.seed, device=args.device, target=args.target, model=args.model,
        grad_clip=args.grad_clip,
    )
    dims = _TARGET_DIMS[args.target]

    recordings, source = _load_recordings(
        data_dir, n_synthetic=args.synthetic_recordings, fs=args.fs, seed=args.seed)
    train_recs, val_recs, _test_recs = _split_recordings(
        recordings, train_cfg.val_fraction, args.test_fraction, args.seed)
    print(f"data source: {source} | recordings: "
          f"train={len(train_recs)} val={len(val_recs)}", flush=True)

    # Build training windows from the TRAIN split only.
    _per_train, train_stacked = build_dataset(
        train_recs, window_cfg, args.target, args.orientation, dims)
    n_train_windows = train_stacked["input"].shape[0]

    # Fit the feature scaler on TRAIN windows only (default identity -> no-op).
    scaler = FeatureScaler(mode=args.scaler)
    flat = train_stacked["input"].reshape(-1, train_stacked["input"].shape[-1])
    scaler.fit(flat)

    def _scale(arr):  # (N, L, 6) -> scaled
        shp = arr.shape
        return scaler.transform(arr.reshape(-1, shp[-1])).reshape(shp).astype(np.float32)

    train_inputs = _scale(train_stacked["input"])
    augment_cfg = None if args.no_augment else AugmentCfg()
    train_ds = WindowDataset(train_inputs, train_stacked["target"],
                             augment_cfg=augment_cfg, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True, drop_last=False)

    val_loader = None
    if val_recs:
        _per_val, val_stacked = build_dataset(
            val_recs, window_cfg, args.target, args.orientation, dims)
        val_inputs = _scale(val_stacked["input"])
        val_ds = WindowDataset(val_inputs, val_stacked["target"],
                               augment_cfg=None, seed=args.seed)
        val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size,
                                shuffle=False, drop_last=False)

    model = build_model(args.model, in_channels=6, out_dim=dims)
    is_probabilistic = args.model == "tlio"
    print(f"model={args.model} out_dim={dims} train_windows={n_train_windows} "
          f"params={sum(p.numel() for p in model.parameters())}", flush=True)

    history = _train_loop(model, train_loader, val_loader, train_cfg,
                          is_probabilistic=is_probabilistic)

    # Checkpoint (everything needed to rebuild + scale at eval time).
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save({
        "model_name": args.model,
        "in_channels": 6,
        "out_dim": dims,
        "state_dict": model.state_dict(),
        "window_cfg": asdict(window_cfg),
        "train_cfg": asdict(train_cfg),
        "target": args.target,
        "orientation": args.orientation,
        "scaler": {"mode": scaler.mode,
                   "mean": None if scaler.mean_ is None else scaler.mean_.tolist(),
                   "std": None if scaler.std_ is None else scaler.std_.tolist()},
    }, ckpt_path)

    # Evaluate on val (held-out recordings); fall back to train if no val split.
    eval_recs = val_recs if val_recs else train_recs
    eval_label = "val" if val_recs else "train"
    device = torch.device(train_cfg.device)
    # Re-scale eval recordings' windows: prepare_recording is unscaled, but eval
    # predicts per recording, so scaling is applied inside _evaluate via a wrapper.
    agg, per_rec, traj = _evaluate_scaled(
        model, eval_recs, window_cfg, args.target, args.orientation, dims,
        device, train_cfg.batch_size, scaler)

    plot_path = os.path.join(out_dir, "trajectory.png")
    plotted = _save_trajectory_plot(
        traj, plot_path, title=f"{args.model} ({eval_label})")

    metrics = {
        "source": source,
        "model": args.model,
        "target": args.target,
        "orientation": args.orientation,
        "eval_split": eval_label,
        "window_cfg": asdict(window_cfg),
        "train_cfg": asdict(train_cfg),
        "n_train_windows": int(n_train_windows),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "history": history,
        "metrics": agg,
        "per_recording": per_rec,
    }
    metrics_path = _write_metrics(out_dir, metrics)

    print(f"\nsaved checkpoint: {ckpt_path}")
    print(f"saved metrics:    {metrics_path}")
    print(f"saved plot:       {plot_path if plotted else '(skipped: no reconstructable recording)'}")
    if agg.get("n"):
        print(f"ATE mean={agg['ate_mean']:.4f} m | RTE mean={agg['rte_mean']:.4f} m "
              f"over {agg['n']} recording(s) [{eval_label}]")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: eval
# --------------------------------------------------------------------------- #
def _cmd_eval(args: argparse.Namespace) -> int:
    paths = Paths()
    data_dir = args.data if args.data is not None else (paths.data_root or None)
    out_dir = args.out if args.out is not None else paths.out_dir
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_name = ckpt["model_name"]
    dims = ckpt["out_dim"]
    target = ckpt["target"]
    orientation = ckpt.get("orientation", "gt")
    window_cfg = WindowCfg(**ckpt["window_cfg"])

    model = build_model(model_name, in_channels=ckpt["in_channels"], out_dim=dims)
    model.load_state_dict(ckpt["state_dict"])

    scaler = FeatureScaler(mode=ckpt["scaler"]["mode"])
    scaler._fitted = True
    scaler.mean_ = (None if ckpt["scaler"]["mean"] is None
                    else np.asarray(ckpt["scaler"]["mean"], dtype=np.float64))
    scaler.std_ = (None if ckpt["scaler"]["std"] is None
                   else np.asarray(ckpt["scaler"]["std"], dtype=np.float64))

    device = torch.device(args.device)

    recordings, source = _load_recordings(
        data_dir, n_synthetic=args.synthetic_recordings, fs=window_cfg.fs,
        seed=args.seed)

    agg, per_rec, traj = _evaluate_scaled(
        model, recordings, window_cfg, target, orientation, dims, device,
        args.batch_size, scaler)

    plot_path = os.path.join(out_dir, "trajectory.png")
    plotted = _save_trajectory_plot(traj, plot_path, title=f"{model_name} (eval)")

    metrics = {
        "source": source,
        "checkpoint": os.path.abspath(args.ckpt),
        "model": model_name,
        "target": target,
        "orientation": orientation,
        "window_cfg": asdict(window_cfg),
        "metrics": agg,
        "per_recording": per_rec,
    }
    metrics_path = _write_metrics(out_dir, metrics)

    print(f"data source: {source} | recordings: {len(recordings)}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved plot:    {plot_path if plotted else '(skipped: no reconstructable recording)'}")
    if agg.get("n"):
        print(f"ATE mean={agg['ate_mean']:.4f} m | RTE mean={agg['rte_mean']:.4f} m "
              f"over {agg['n']} recording(s)")
    return 0


def _evaluate_scaled(model, recordings, window_cfg, target_kind, orientation,
                     dims, device, batch_size, scaler: FeatureScaler):
    """Wrap :func:`_evaluate`, applying the fitted scaler to each recording's
    windows before prediction (no leakage: scaler was fit on train only)."""
    recon_mode = _TARGET_RECON.get(target_kind)
    per_rec = []
    first_traj = None
    for rec in recordings:
        prep = prepare_recording(rec, window_cfg, target_kind, orientation, dims)
        if prep["input"].shape[0] == 0:
            continue
        shp = prep["input"].shape
        scaled = scaler.transform(
            prep["input"].reshape(-1, shp[-1])).reshape(shp).astype(np.float32)
        pred = _predict_windows(model, scaled, device, batch_size)
        if recon_mode is None:
            continue
        if recon_mode == "velocity":
            kwargs = {"dt": window_cfg.window_dt, "p0": prep["p_start"][0]}
        else:
            kwargs = {"start_offsets": prep["p_start"]}
        est = reconstruct_trajectory(pred, mode=recon_mode, **kwargs)
        gt = reconstruct_trajectory(prep["target"], mode=recon_mode, **kwargs)
        ate, rte = compute_ate_rte(est, gt, pred_per_min=window_cfg.pred_per_min)
        per_rec.append({"rec_id": rec.rec_id, "ate": ate, "rte": rte})
        if first_traj is None:
            first_traj = (rec.rec_id, est, gt)
    agg = (aggregate_metrics([(r["ate"], r["rte"]) for r in per_rec])
           if per_rec else {"n": 0})
    return agg, per_rec, first_traj


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=None,
                   help="RoNIN dataset folder (subdir/data.hdf5 per recording). "
                        "If absent, falls back to $NINAV_DATA, else synthetic.")
    p.add_argument("--out", default=None,
                   help="Output directory for artifacts (default $NINAV_OUT or 'runs').")
    p.add_argument("--device", default="cpu", help="torch device (default cpu).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--window-len", type=int, default=200, help="samples per window.")
    p.add_argument("--stride", type=int, default=10, help="samples between windows.")
    p.add_argument("--fs", type=float, default=200.0, help="sample rate (Hz).")
    p.add_argument("--target", default="velocity",
                   choices=["velocity", "displacement", "polar"])
    p.add_argument("--orientation", default="gt", choices=["gt", "gyro"],
                   help="per-sample orientation source for the world-frame rotation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-recordings", type=int, default=8,
                   help="number of synthetic recordings when --data is absent.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ninav.cli",
        description="Train / evaluate neural inertial navigation models (ninav).")
    sub = parser.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="train a model and evaluate the held-out split.")
    _add_common(pt)
    pt.add_argument("--model", default="resnet1d",
                    choices=["resnet1d", "transformer", "lstm", "tcn", "tlio"])
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight-decay", type=float, default=1e-4)
    pt.add_argument("--val-fraction", type=float, default=0.2)
    pt.add_argument("--test-fraction", type=float, default=0.0)
    pt.add_argument("--grad-clip", type=float, default=1.0)
    pt.add_argument("--scaler", default="identity", choices=["identity", "standard"])
    pt.add_argument("--no-augment", action="store_true",
                    help="disable train-time augmentation (random yaw, noise, bias).")
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("eval", help="evaluate a saved checkpoint on a dataset.")
    _add_common(pe)
    pe.add_argument("--ckpt", required=True, help="path to a checkpoint.pt from train.")
    pe.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
