"""Configuration dataclasses. No hardcoded absolute paths (the old code was full
of ``C:\\Users\\Simon Andersen\\...`` and ``/Users/antongolles/...``)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class WindowCfg:
    """IMU windowing configuration."""
    fs: float = 200.0          # sample rate (Hz)
    window_len: int = 200      # samples per window (1 s @ 200 Hz)
    stride: int = 10           # samples between window starts (>0, <= window_len)

    def __post_init__(self):
        if self.window_len < 1:
            raise ValueError("window_len must be >= 1")
        if not (0 < self.stride <= self.window_len):
            raise ValueError(f"stride must satisfy 0 < stride <= window_len, got {self.stride}")

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def window_dt(self) -> float:
        """Time between consecutive window *starts* (for velocity integration)."""
        return self.stride / self.fs

    @property
    def pred_per_min(self) -> int:
        """Number of window predictions per minute (for RTE)."""
        return int(round(60.0 * self.fs / self.stride))


@dataclass
class TrainCfg:
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "cpu"
    target: str = "velocity"        # "velocity" | "displacement" | "polar"
    model: str = "resnet1d"         # "resnet1d" | "transformer" | "lstm" | "tcn" | "tlio"
    grad_clip: float | None = 1.0
    nll_warmup: int = 10            # MSE warm-up epochs before Gaussian NLL (TLIO head)


@dataclass
class AugmentCfg:
    random_yaw: bool = True                 # rotate input+target together about gravity
    gaussian_noise_acc: float = 0.1         # m/s^2 std
    gaussian_noise_gyro: float = 0.001      # rad/s std
    bias_acc: float = 0.2                   # uniform +/- m/s^2
    bias_gyro: float = 0.05                 # uniform +/- rad/s
    gravity_tilt_deg: float = 5.0           # max random tilt (deg)


@dataclass
class Paths:
    """Filesystem paths resolved from env/args, never hardcoded."""
    data_root: str = field(default_factory=lambda: os.environ.get("NINAV_DATA", ""))
    out_dir: str = field(default_factory=lambda: os.environ.get("NINAV_OUT", "runs"))


def to_dict(cfg) -> dict:
    return asdict(cfg)
