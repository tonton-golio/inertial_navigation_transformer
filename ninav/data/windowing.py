"""Per-recording windowing with an explicit (non-inverted) stride.

Fixes:
    * windows were built across concatenated recordings, so ~``num_recordings``
      windows straddled the seam between two unrelated trajectories (bug H16);
    * the ``overlap`` argument was inverted (``stride = seq_len - overlap``) and
      silently produced empty output for ``overlap >= seq_len`` (bug M3).

Here windows are always built from a SINGLE recording's stream, and the stride
is explicit. The window provides ``window_len`` samples of context; the motion
target (see :mod:`ninav.data.targets`) is defined over the ``stride`` interval
so that reconstruction telescopes exactly (a perfect model -> zero ATE).
"""
from __future__ import annotations

import numpy as np


def valid_starts(n_samples: int, window_len: int, stride: int) -> np.ndarray:
    """Window-start indices such that both the window AND a ``stride``-ahead
    target sample exist within a single recording of length ``n_samples``."""
    if window_len < 1:
        raise ValueError("window_len must be >= 1")
    if not (0 < stride <= window_len):
        raise ValueError(f"stride must satisfy 0 < stride <= window_len, got {stride}")
    # need s + window_len <= n   and   s + stride <= n-1 (target uses pos[s+stride])
    max_start = min(n_samples - window_len, n_samples - 1 - stride)
    if max_start < 0:
        return np.empty((0,), dtype=np.int64)
    return np.arange(0, max_start + 1, stride, dtype=np.int64)


def create_windows(stream: np.ndarray, window_len: int, stride: int):
    """Slice one recording's ``(T, C)`` stream into windows.

    Returns ``(windows, starts)`` where ``windows`` is ``(N, window_len, C)`` and
    ``starts`` is ``(N,)`` of the sample index each window begins at. Never spans
    more than one recording (call once per recording). Does not mutate ``stream``.
    """
    stream = np.asarray(stream)
    if stream.ndim != 2:
        raise ValueError(f"stream must be (T, C), got {stream.shape}")
    starts = valid_starts(stream.shape[0], window_len, stride)
    if starts.size == 0:
        return np.empty((0, window_len, stream.shape[1]), dtype=stream.dtype), starts
    # Vectorized gather via a sliding index matrix (no Python loop, no mutation).
    idx = starts[:, None] + np.arange(window_len)[None, :]
    windows = stream[idx]
    return windows, starts
