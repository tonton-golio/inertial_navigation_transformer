"""Dataset splitting BY recording/subject (no window shared across splits).

Fixes the old in-distribution split where validation was a contiguous tail of
the SAME recordings as training, sharing a boundary window (bug M14), and the
unused positional ``split_data`` that would leak across overlapping windows (L2).

Splitting whole recordings guarantees no train/val/test window overlap and
mirrors the RoNIN protocol (held-out subjects).
"""
from __future__ import annotations

import numpy as np


def split_recordings(recording_ids, val_fraction: float = 0.2,
                     test_fraction: float = 0.0, seed: int = 42) -> dict:
    """Partition recording ids into train/val/test by WHOLE recording.

    Parameters
    ----------
    recording_ids : sequence of hashable ids (one per recording).
    val_fraction, test_fraction : fractions of recordings (not samples).
    seed : RNG seed for the shuffle.

    Returns ``{"train": [...], "val": [...], "test": [...]}`` of recording ids.
    """
    ids = list(recording_ids)
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("require val_fraction + test_fraction < 1, each in [0,1)")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_fraction * n))
    n_val = int(round(val_fraction * n))
    # Guarantee at least one training recording.
    n_val = min(n_val, max(0, n - n_test - 1))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return {
        "train": [ids[i] for i in train_idx],
        "val": [ids[i] for i in val_idx],
        "test": [ids[i] for i in test_idx],
    }
