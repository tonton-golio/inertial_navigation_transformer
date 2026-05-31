"""Training + evaluation loop that ties the ninav spine together.

This module is the only "glue" layer over the proven numeric spine. It does
NOT redefine any geometry, target, reconstruction or metric logic -- it only
orchestrates the spine pieces:

    split_recordings  (by whole recording)            -> ninav.data.splits
    build_dataset     (world-frame windows + targets) -> ninav.data.dataset
    WindowDataset     (channel-first (6, L) items)    -> ninav.data.dataset
    build_model       (model contract by name)        -> here (thin factory)
    mse / gaussian_nll                                -> ninav.losses.regression
    integrate_velocity (est AND gt, identically)      -> ninav.reconstruct
    compute_ate_rte / aggregate_metrics               -> ninav.metrics

Design intent of :func:`evaluate`: predictions are WORLD-FRAME velocity, so they
are integrated directly. The ground-truth trajectory is reconstructed with the
SAME ``integrate_velocity`` call from the SAME per-window velocity *targets* and
the SAME start anchor, so a perfect model (pred == target) reconstructs to zero
error for any stride (the spine's telescoping property).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import AugmentCfg, TrainCfg, WindowCfg
from .data.dataset import WindowDataset, build_dataset
from .data.splits import split_recordings
from .losses.regression import gaussian_nll_loss, mse_loss
from .metrics import aggregate_metrics, compute_ate_rte
from .models.lstm_tcn import LSTMRegressor, TCNRegressor
from .models.resnet1d import ResNet1D
from .models.tlio_head import TLIONet
from .models.transformer import TransformerRegressor
from .reconstruct import integrate_velocity

__all__ = ["build_model", "train", "evaluate", "is_probabilistic"]


# --------------------------------------------------------------------------- #
# Model factory (no spine `build_model` exists; thin local mapping by name).   #
# --------------------------------------------------------------------------- #
_MODELS = {
    "resnet1d": ResNet1D,
    "transformer": TransformerRegressor,
    "lstm": LSTMRegressor,
    "tcn": TCNRegressor,
    "tlio": TLIONet,
}

# Models whose forward returns (mean, log_std) and train with Gaussian NLL.
_PROBABILISTIC = {"tlio"}


def is_probabilistic(model_name: str) -> bool:
    """True if ``model_name`` produces a ``(mean, log_std)`` tuple."""
    return model_name in _PROBABILISTIC


def build_model(model_name: str, in_channels: int = 6, out_dim: int = 2,
                **arch_kwargs) -> nn.Module:
    """Construct a position model by name, honouring the model contract.

    Every model takes ``(in_channels, out_dim, **arch_kwargs)`` and a
    ``forward(x)`` over ``x`` of shape ``(B, in_channels, L)``. The ``tlio``
    model returns ``(mean, log_std)``; all others return ``(B, out_dim)``.
    """
    try:
        cls = _MODELS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model {model_name!r}; choose from {sorted(_MODELS)}"
        ) from exc
    return cls(in_channels=in_channels, out_dim=out_dim, **arch_kwargs)


# --------------------------------------------------------------------------- #
# Forward + loss helper (uniformly handles the deterministic / TLIO split).    #
# --------------------------------------------------------------------------- #
def _forward_pred(model: nn.Module, x: torch.Tensor, probabilistic: bool):
    """Run the model and split out ``(mean, log_std)`` for the loss/eval.

    Returns ``(mean, log_std_or_None)``. For deterministic models the second
    element is ``None``.
    """
    out = model(x)
    if probabilistic:
        mean, log_std = out
        return mean, log_std
    return out, None


def _loss(mean: torch.Tensor, target: torch.Tensor,
          log_std: torch.Tensor | None, use_nll: bool = True) -> torch.Tensor:
    """MSE for deterministic models; for the probabilistic (TLIO) head, MSE on
    the mean during the warm-up (``use_nll=False``) then Gaussian NLL. NLL from
    scratch does not converge (TLIO), hence the warm-up."""
    if log_std is None or not use_nll:
        return mse_loss(mean, target)
    return gaussian_nll_loss(mean, target, log_std)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train(recordings: Sequence, win_cfg: WindowCfg, train_cfg: TrainCfg,
          aug_cfg: AugmentCfg | None = None) -> dict:
    """Train a position model on synthetic / real recordings.

    Splits by WHOLE recording (held-out val), builds world-frame windows +
    velocity (or displacement / polar) targets via the spine, trains with
    augmentation on the train split only, validates each epoch, and returns a
    history dict plus the trained model and the held-out eval bundles.

    Returns
    -------
    dict with keys:
        ``model``            -- the trained ``nn.Module`` (in eval mode);
        ``history``          -- ``{"train_loss": [...], "val_loss": [...]}``;
        ``train_per_recording`` / ``val_per_recording`` -- per-recording dicts
            from :func:`ninav.data.dataset.build_dataset` (for evaluation);
        ``split``            -- the recording-id split dict.
    """
    device = torch.device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)

    # --- 1. split by whole recording -------------------------------------- #
    ids = [r.rec_id for r in recordings]
    by_id = {r.rec_id: r for r in recordings}
    split = split_recordings(ids, val_fraction=train_cfg.val_fraction,
                             test_fraction=0.0, seed=train_cfg.seed)
    train_recs = [by_id[i] for i in split["train"]]
    val_recs = [by_id[i] for i in split["val"]]
    if not train_recs:
        raise ValueError("no training recordings after split")

    # --- 2. build world-frame windows + targets (GT orientation) ---------- #
    out_dim = _target_dims(train_cfg.target)
    probabilistic = is_probabilistic(train_cfg.model)

    train_per, train_stacked = build_dataset(
        train_recs, win_cfg, target_kind=train_cfg.target,
        orientation="gt", dims=out_dim)
    val_per = []
    val_stacked = None
    if val_recs:
        try:
            val_per, val_stacked = build_dataset(
                val_recs, win_cfg, target_kind=train_cfg.target,
                orientation="gt", dims=out_dim)
        except ValueError:
            # Val recordings too short for the window/stride -> skip val.
            val_per, val_stacked = [], None

    # --- 3. datasets / loaders (augment train, NOT val) ------------------- #
    train_ds = WindowDataset(train_stacked["input"], train_stacked["target"],
                             augment_cfg=aug_cfg, seed=train_cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True, drop_last=False)
    val_loader = None
    if val_stacked is not None:
        val_ds = WindowDataset(val_stacked["input"], val_stacked["target"],
                               augment_cfg=None, seed=train_cfg.seed)
        val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size,
                                shuffle=False, drop_last=False)

    # --- 4. model + optimizer --------------------------------------------- #
    model = build_model(train_cfg.model, in_channels=6, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)

    history = {"train_loss": [], "val_loss": []}

    # --- 5. epochs -------------------------------------------------------- #
    for _epoch in range(train_cfg.epochs):
        model.train()
        # TLIO-style two-stage objective: MSE warm-up, then Gaussian NLL.
        use_nll = (not probabilistic) or (_epoch >= train_cfg.nll_warmup)
        running, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            mean, log_std = _forward_pred(model, x, probabilistic)
            loss = _loss(mean, y, log_std, use_nll=use_nll)
            loss.backward()
            if train_cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            running += float(loss.detach())
            n_batches += 1
        history["train_loss"].append(running / max(n_batches, 1))

        # validation
        if val_loader is not None:
            model.eval()
            v_running, v_batches = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    mean, log_std = _forward_pred(model, x, probabilistic)
                    v_running += float(_loss(mean, y, log_std, use_nll=use_nll))
                    v_batches += 1
            history["val_loss"].append(v_running / max(v_batches, 1))
        else:
            history["val_loss"].append(float("nan"))

    model.eval()
    return {
        "model": model,
        "history": history,
        "train_per_recording": train_per,
        "val_per_recording": val_per,
        "split": split,
    }


def _target_dims(target_kind: str) -> int:
    """Output dimensionality implied by the target kind (2D horizontal)."""
    # velocity / displacement are 2D vectors; polar is (magnitude, angle).
    return 2


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, per_recording_list, win_cfg: WindowCfg,
             device: str = "cpu") -> dict:
    """Reconstruct + score every recording, then aggregate.

    For each recording (windows already in start order) the model predicts a
    world-frame velocity stream which is integrated with
    ``integrate_velocity(pred, dt=win_cfg.window_dt, p0=p_start[0])``. The
    ground-truth trajectory is reconstructed by the SAME call from the true
    per-window velocity *targets* and the SAME start anchor, so identical
    predictions give zero error. ``compute_ate_rte`` then scores est vs gt with
    ``pred_per_min=win_cfg.pred_per_min``.

    Returns
    -------
    dict with:
        ``aggregate``       -- :func:`ninav.metrics.aggregate_metrics` output;
        ``per_recording``   -- list of dicts ``{rec_id, ate, rte, est, gt,
                                pred_vel, gt_vel}`` for plotting.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    out_model_dim = getattr(model, "out_dim", 2)
    probabilistic = is_probabilistic(_model_name(model))

    per_results = []
    pairs = []
    dt = win_cfg.window_dt
    for rec in per_recording_list:
        inputs = rec["input"]                       # (N, L, 6)
        if inputs.shape[0] == 0:
            continue
        target_vel = np.asarray(rec["target"], dtype=np.float64)  # (N, dims)
        p_start = np.asarray(rec["p_start"], dtype=np.float64)    # (N, dims)
        dims = target_vel.shape[1]

        # channel-first batch: (N, 6, L)
        x = torch.from_numpy(
            np.ascontiguousarray(inputs.transpose(0, 2, 1), dtype=np.float32)
        ).to(dev)
        out = model(x)
        mean = out[0] if probabilistic else out
        pred_vel = mean.detach().cpu().numpy().astype(np.float64)  # (N, dims)
        pred_vel = pred_vel[:, :dims]

        # Single shared reconstruction operator, identical for est and gt.
        p0 = p_start[0]
        est = integrate_velocity(pred_vel, dt=dt, p0=p0)
        gt = integrate_velocity(target_vel, dt=dt, p0=p0)

        ate, rte = compute_ate_rte(est, gt, pred_per_min=win_cfg.pred_per_min)
        pairs.append((ate, rte))
        per_results.append({
            "rec_id": rec["rec_id"],
            "ate": ate,
            "rte": rte,
            "est": est,
            "gt": gt,
            "pred_vel": pred_vel,
            "gt_vel": target_vel,
        })

    if not pairs:
        raise ValueError("no recordings produced windows to evaluate")

    return {
        "aggregate": aggregate_metrics(pairs),
        "per_recording": per_results,
    }


def _model_name(model: nn.Module) -> str:
    """Reverse-lookup the registry name for a model instance (for eval split)."""
    for name, cls in _MODELS.items():
        if isinstance(model, cls):
            return name
    return ""


if __name__ == "__main__":
    from .data.synthetic import generate_dataset

    recs = generate_dataset(n_recordings=4, duration_s=12, fs=200)
    wcfg = WindowCfg(window_len=100, stride=20)
    tcfg = TrainCfg(epochs=3, model="resnet1d")
    result = train(recs, wcfg, tcfg, aug_cfg=AugmentCfg())
    print("train_loss:", result["history"]["train_loss"])
    print("val_loss:  ", result["history"]["val_loss"])
    ev = evaluate(result["model"], result["train_per_recording"], wcfg)
    print("aggregate:", ev["aggregate"])
