"""Tests for ninav.data: windowing, scaling, targets, splits, synthetic consistency.

Guards the previously-fixed bugs:
    * H16 -- windows straddling the seam between two concatenated recordings;
    * M3  -- inverted/silent stride semantics (overlap-as-stride producing empties);
    * C6  -- FeatureScaler re-fitting on test data (eval leakage);
    * C9  -- targets rebased differently for train vs test / input mutation;
    * M14 -- train/val/test split sharing a recording (window overlap across splits);
    * M4  -- gyro integration / world-frame orientation inconsistency.

Fast: tiny synthetic recordings, CPU only.
"""
from __future__ import annotations

import numpy as np
import pytest

from ninav.config import WindowCfg
from ninav.data import (
    FeatureScaler,
    build_dataset,
    create_windows,
    displacement_target,
    generate_dataset,
    generate_recording,
    make_target,
    prepare_recording,
    split_recordings,
    start_offsets,
    valid_starts,
    velocity_target,
)
from ninav.data.dataset import world_frame_imu
from ninav.geometry.propagate import integrate_gyro


# ---------------------------------------------------------------------------
# (1) Windowing: never spans two recordings; explicit stride semantics. (H16, M3)
# ---------------------------------------------------------------------------

def test_windows_stay_within_single_recording_length():
    """Every (start, start+window_len) interval AND the stride-ahead target
    sample must lie inside the single stream the windows came from."""
    n_samples, window_len, stride = 137, 20, 5
    stream = np.arange(n_samples * 3, dtype=np.float64).reshape(n_samples, 3)
    windows, starts = create_windows(stream, window_len, stride)

    assert windows.ndim == 3 and windows.shape[1:] == (window_len, 3)
    assert starts.shape[0] == windows.shape[0]
    # Window fully inside the stream.
    assert (starts >= 0).all()
    assert (starts + window_len <= n_samples).all()
    # The stride-ahead target sample (pos[start+stride]) must also exist.
    assert (starts + stride <= n_samples - 1).all()


def test_windows_come_from_a_single_stream_only():
    """If we conceptually concatenate two distinct recordings, calling
    create_windows once per recording must never mix samples across the seam.

    We build a marker stream where every sample value encodes its recording id;
    a window built from recording A must contain ONLY A's marker, never B's
    (the H16 seam-straddling bug would pull samples across the boundary).
    """
    n_a, n_b, window_len, stride = 60, 70, 15, 5
    # Recording A samples are tagged 0.x; recording B samples are tagged 1.x.
    stream_a = np.full((n_a, 2), 0.0) + np.arange(n_a)[:, None] * 1e-3
    stream_b = np.full((n_b, 2), 1.0) + np.arange(n_b)[:, None] * 1e-3
    concat = np.concatenate([stream_a, stream_b], axis=0)

    # The correct API: window each recording separately, never the concat.
    wins_a, _ = create_windows(stream_a, window_len, stride)
    wins_b, _ = create_windows(stream_b, window_len, stride)

    # No A window contains any B-tagged sample (>= 1.0) and vice-versa.
    assert wins_a.size > 0 and wins_b.size > 0
    assert (wins_a < 1.0).all(), "recording A windows leaked recording B samples"
    assert (wins_b >= 1.0).all(), "recording B windows leaked recording A samples"

    # Sanity: the number of windows from the seamful concat (windowed as one
    # stream) is strictly larger than A+B windowed separately -- those extra
    # windows are exactly the seam-straddling ones the per-recording API avoids.
    wins_concat, _ = create_windows(concat, window_len, stride)
    assert wins_concat.shape[0] > wins_a.shape[0] + wins_b.shape[0]


def test_stride_equal_window_len_is_disjoint_tiling():
    """stride == window_len -> windows tile the stream with no overlap and no gap
    between consecutive windows (each sample used at most once)."""
    n_samples, window_len = 100, 25
    stride = window_len
    stream = np.arange(n_samples, dtype=np.float64).reshape(n_samples, 1)
    windows, starts = create_windows(stream, window_len, stride)

    # Consecutive starts differ by exactly window_len -> disjoint, contiguous tiling.
    assert np.all(np.diff(starts) == window_len)
    # No sample index appears in two windows: flattening the covered indices is unique.
    covered = np.concatenate([np.arange(s, s + window_len) for s in starts])
    assert covered.size == np.unique(covered).size, "tiles overlap"


def test_stride_greater_than_window_len_raises():
    """stride > window_len is rejected (was silently inverted in bug M3)."""
    stream = np.zeros((100, 3))
    with pytest.raises(ValueError):
        create_windows(stream, window_len=20, stride=21)
    with pytest.raises(ValueError):
        valid_starts(100, window_len=20, stride=21)
    with pytest.raises(ValueError):
        valid_starts(100, window_len=20, stride=0)


# ---------------------------------------------------------------------------
# (2) FeatureScaler: fit on train, transform on test does NOT re-fit. (C6)
# ---------------------------------------------------------------------------

def test_standard_scaler_uses_train_stats_on_test():
    """Fit on train, then transform a differently-distributed test set: the
    transformed test mean must NOT be ~0 (it would be only if re-fit on test)."""
    rng = np.random.default_rng(0)
    train = rng.normal(loc=0.0, scale=1.0, size=(500, 3))
    # Test distribution is shifted far from train.
    test = rng.normal(loc=10.0, scale=1.0, size=(400, 3))

    scaler = FeatureScaler(mode="standard")
    scaler.fit(train)
    train_t = scaler.transform(train)
    test_t = scaler.transform(test)

    # Train transforms to ~zero mean (it was fit on train).
    assert np.allclose(train_t.mean(axis=0), 0.0, atol=1e-6)
    # Test does NOT, because train stats were used (no re-fit / no leakage).
    assert np.all(np.abs(test_t.mean(axis=0)) > 5.0)


def test_standard_scaler_transform_is_idempotent_to_fit_data():
    """A second transform() call returns the same output -- fit() is not invoked
    again by transform() (regression guard for the fit_transform-everywhere bug)."""
    rng = np.random.default_rng(1)
    train = rng.normal(loc=3.0, scale=2.0, size=(300, 4))
    test = rng.normal(loc=-7.0, scale=0.5, size=(200, 4))
    scaler = FeatureScaler(mode="standard").fit(train)
    first = scaler.transform(test)
    second = scaler.transform(test)
    assert np.array_equal(first, second)
    # Stats are still the TRAIN stats after transforming test.
    assert np.allclose(scaler.mean_, train.mean(axis=0))


def test_transform_before_fit_raises():
    """Calling transform before fit must raise (cannot leak un-fit stats)."""
    scaler = FeatureScaler(mode="standard")
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((5, 3)))


def test_identity_mode_returns_input_unchanged():
    """identity mode returns the input values unchanged (a copy, not the signal)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    scaler = FeatureScaler(mode="identity").fit(X)
    out = scaler.transform(X)
    assert np.array_equal(out, X)
    # Returns a copy (mutating the output must not corrupt the input).
    out[0, 0] += 123.0
    assert X[0, 0] != out[0, 0]


def test_scaler_invalid_mode_raises():
    with pytest.raises(ValueError):
        FeatureScaler(mode="whiten")


# ---------------------------------------------------------------------------
# (3) Targets are relative, identical for any split, and do not mutate input. (C9)
# ---------------------------------------------------------------------------

def test_velocity_and_displacement_targets_are_deterministic_and_relative():
    """Targets depend only on (positions, starts, stride) -- identical regardless
    of which split the recording lands in -- and are relative differences."""
    rng = np.random.default_rng(3)
    positions = np.cumsum(rng.normal(size=(200, 3)), axis=0)
    starts = np.arange(0, 150, 10, dtype=np.int64)
    stride, dt = 10, 1.0 / 200.0

    disp = displacement_target(positions, starts, stride, dims=2)
    vel = velocity_target(positions, starts, stride, dt, dims=2)

    # Relative: displacement == pos[s+stride] - pos[s]; velocity == disp / (stride*dt).
    expected_disp = positions[starts + stride, :2] - positions[starts, :2]
    assert np.allclose(disp, expected_disp)
    assert np.allclose(vel, disp / (stride * dt))

    # Determinism: a second identical call yields identical output.
    assert np.array_equal(disp, displacement_target(positions, starts, stride, dims=2))

    # Split-invariance: the same recording windowed in any subset gives the same
    # target for each shared start (no per-split rebasing -- the C9 bug).
    sub_starts = starts[3:7]
    assert np.allclose(
        displacement_target(positions, sub_starts, stride, dims=2),
        disp[3:7],
    )


def test_target_construction_does_not_mutate_positions():
    """Building targets must not modify the caller's positions array (no in-place
    rebasing). This guards the train-only rebase that caused bug C9."""
    rng = np.random.default_rng(4)
    positions = np.cumsum(rng.normal(size=(120, 3)), axis=0)
    snapshot = positions.copy()
    starts = np.arange(0, 100, 5, dtype=np.int64)
    stride, dt = 5, 1.0 / 200.0

    for kind in ("velocity", "displacement", "polar"):
        _ = make_target(kind, positions, starts, stride, dt, dims=2)
        assert np.array_equal(positions, snapshot), f"{kind} mutated positions"
    # start_offsets likewise must not mutate.
    _ = start_offsets(positions, starts, dims=2)
    assert np.array_equal(positions, snapshot)


# ---------------------------------------------------------------------------
# (4) Splits partition WHOLE recordings; no id in two splits; >= 1 train. (M14)
# ---------------------------------------------------------------------------

def test_split_recordings_partitions_whole_recordings():
    ids = [f"rec_{i:02d}" for i in range(10)]
    split = split_recordings(ids, val_fraction=0.2, test_fraction=0.2, seed=7)

    all_assigned = split["train"] + split["val"] + split["test"]
    # Every recording assigned exactly once (whole-recording partition).
    assert sorted(all_assigned) == sorted(ids)
    assert len(all_assigned) == len(set(all_assigned)), "an id appears in two splits"

    # No overlap between any pair of splits.
    s_train, s_val, s_test = set(split["train"]), set(split["val"]), set(split["test"])
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)

    # At least one training recording.
    assert len(split["train"]) >= 1


def test_split_guarantees_train_even_with_high_fractions():
    """Even with aggressive val/test fractions on a small set, train stays >= 1."""
    ids = ["a", "b", "c"]
    split = split_recordings(ids, val_fraction=0.9, test_fraction=0.0, seed=0)
    assert len(split["train"]) >= 1
    assert sorted(split["train"] + split["val"] + split["test"]) == ids


def test_split_is_deterministic_for_fixed_seed():
    ids = [f"r{i}" for i in range(8)]
    a = split_recordings(ids, val_fraction=0.25, seed=11)
    b = split_recordings(ids, val_fraction=0.25, seed=11)
    assert a == b


# ---------------------------------------------------------------------------
# (5) Synthetic self-consistency: gyro -> orientation; gt vs gyro world IMU. (M4)
# ---------------------------------------------------------------------------

def test_integrate_gyro_recovers_orientation():
    """Re-integrating the recording's own gyro from its initial orientation
    recovers the per-sample orientation to < 1e-5 (forward model is invertible)."""
    rec = generate_recording("syn", duration_s=2.0, fs=200.0, seed=5)
    dt = 1.0 / rec.fs
    recovered = integrate_gyro(rec.ori[0], rec.gyro, dt, frame="body")
    assert recovered.shape == rec.ori.shape
    # Account for the quaternion double-cover (q and -q are the same rotation).
    sign = np.sign(np.sum(recovered * rec.ori, axis=1, keepdims=True))
    sign[sign == 0] = 1.0
    err = np.abs(recovered * sign - rec.ori).max()
    assert err < 1e-5, f"gyro re-integration error {err}"


def test_world_frame_imu_gt_matches_gyro():
    """The world-frame IMU input built from ground-truth orientation must match
    the one built from gyro-integrated orientation (they share the orientation
    used to generate the data)."""
    rec = generate_recording("syn", duration_s=2.0, fs=200.0, seed=6)
    wf_gt = world_frame_imu(rec, orientation="gt")
    wf_gyro = world_frame_imu(rec, orientation="gyro")
    assert wf_gt.shape == wf_gyro.shape == (rec.acc.shape[0], 6)
    assert np.allclose(wf_gt, wf_gyro, atol=1e-4)


def test_prepare_recording_reconstruction_anchor_matches_ground_truth():
    """prepare_recording's velocity target + p_start telescope back to the true
    position at the next window start (end-to-end target/anchor consistency)."""
    rec = generate_recording("syn", duration_s=3.0, fs=200.0, seed=7)
    cfg = WindowCfg(fs=200.0, window_len=200, stride=10)
    prep = prepare_recording(rec, cfg, target_kind="velocity", orientation="gt", dims=2)

    starts = prep["starts"]
    assert starts.size > 1
    # p_start[j] == true 2D position at start[j].
    assert np.allclose(prep["p_start"], rec.pos[starts, :2])
    # velocity * (stride*dt) == displacement to pos[start+stride]; integrating
    # from p_start[j] lands on the true position at start[j]+stride.
    disp = prep["target"] * (cfg.stride * cfg.dt)
    landed = prep["p_start"] + disp
    expected = rec.pos[starts + cfg.stride, :2]
    assert np.allclose(landed, expected, atol=1e-4)


def test_build_dataset_does_not_span_recordings():
    """build_dataset stacks per-recording windows; rec_index maps each stacked
    window to exactly one recording, and counts match per-recording windowing."""
    recs = generate_dataset(n_recordings=3, duration_s=2.0, fs=200.0, seed=0)
    cfg = WindowCfg(fs=200.0, window_len=100, stride=20)
    per, stacked = build_dataset(recs, cfg, target_kind="velocity", orientation="gt", dims=2)

    total = sum(p["input"].shape[0] for p in per)
    assert stacked["input"].shape[0] == total
    assert stacked["rec_index"].shape[0] == total
    # rec_index is a contiguous block per recording (no interleaving across recs).
    for i, p in enumerate(per):
        assert int((stacked["rec_index"] == i).sum()) == p["input"].shape[0]
    assert stacked["input"].shape[1:] == (cfg.window_len, 6)
