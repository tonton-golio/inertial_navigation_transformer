"""Feature scaling that fits on TRAIN ONLY and transforms val/test.

Fixes bug C6: the old ``normalize_features`` called ``fit_transform``
unconditionally, so passing a fitted scaler still re-fit on the test set
(leakage / invalid eval). Here ``fit`` and ``transform`` are separate, and the
default is **identity** -- per Keller et al. (2025) z-scoring raw IMU channels
*degrades* neural inertial regression because absolute accel/gyro magnitudes
carry the motion signal; calibration + gravity-frame rotation is the real
preprocessing, not whitening.
"""
from __future__ import annotations

import numpy as np


class FeatureScaler:
    """Per-channel scaler. ``mode='identity'`` (default) or ``'standard'``."""

    def __init__(self, mode: str = "identity"):
        if mode not in ("identity", "standard"):
            raise ValueError(f"mode must be 'identity' or 'standard', got {mode!r}")
        self.mode = mode
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """Fit on TRAIN data only. ``X`` is ``(..., C)``; stats over all but last."""
        X = np.asarray(X, dtype=np.float64)
        axes = tuple(range(X.ndim - 1))
        if self.mode == "standard":
            self.mean_ = X.mean(axis=axes)
            std = X.std(axis=axes)
            self.std_ = np.where(std < 1e-8, 1.0, std)
        else:
            self.mean_ = np.zeros(X.shape[-1])
            self.std_ = np.ones(X.shape[-1])
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the FITTED stats. Never re-fits (no leakage)."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        if self.mode == "identity":
            return X.copy()
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit + transform -- ONLY ever call this on training data."""
        return self.fit(X).transform(X)
