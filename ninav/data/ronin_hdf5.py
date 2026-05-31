"""Load real RoNIN HDF5 recordings into the repo's :class:`Recording` dataclass.

The numeric correctness spine (:mod:`ninav.geometry`, :mod:`ninav.data.synthetic`)
is exercised on synthetic data, but this module lets the same pipeline ingest the
real RoNIN dataset once it is downloaded.

Download
--------
RoNIN ("Robust Neural Inertial Navigation") is published by the Sensing,
Interaction & Perception Lab (SFU). The dataset and reference code are at:

    https://ronin.cs.sfu.ca/            (dataset landing page)
    https://github.com/Sachini/ronin    (official reference implementation)

After downloading, each *recording* is a directory containing a ``data.hdf5``
file (and metadata). A typical split layout looks like::

    <ronin_root>/
        a000_1/ data.hdf5
        a000_2/ data.hdf5
        ...

HDF5 layout (the fields this loader reads)
------------------------------------------
    synced/time        (T,)    timestamps (seconds)
    synced/gyro        (T, 3)  body-frame angular velocity (rad/s)
    synced/acce        (T, 3)  body-frame specific force / accelerometer (m/s^2)
    synced/magnet      (T, 3)  body-frame magnetometer (uT)   [optional]
    pose/tango_pos     (T, 3)  world-frame position from Tango ground truth (m)
    pose/tango_ori     (T, 4)  body->world orientation quaternion

CRITICAL CONVENTION FIX (bug C3)
--------------------------------
RoNIN stores ``pose/tango_ori`` as a **SCALAR-FIRST** quaternion ``[w, x, y, z]``
(the Android / Tango convention). Everything in :mod:`ninav.geometry` is
**SCALAR-LAST** ``[x, y, z, w]`` (matching
``scipy.spatial.transform.Rotation.from_quat``). The old ``utils.py`` mixed the
two conventions silently (``rotation_matrix_2_quaternion`` was scalar-first while
``Omega`` was scalar-last -- see bug C3 in REVIEW.md), which corrupted every
rotation that touched orientation.

This loader performs the conversion **exactly once, at the boundary**, in
:func:`_scalar_first_to_scalar_last`, so that the returned :class:`Recording`
already carries scalar-last orientation and the rest of the pipeline never has to
reason about the RoNIN convention again.
"""
from __future__ import annotations

import os

import numpy as np

from .synthetic import Recording

#: Dataset paths inside each RoNIN ``data.hdf5`` file.
_GYRO_KEY = "synced/gyro"
_ACCE_KEY = "synced/acce"
_MAGNET_KEY = "synced/magnet"
_TIME_KEY = "synced/time"
_POS_KEY = "pose/tango_pos"
_ORI_KEY = "pose/tango_ori"


def _scalar_first_to_scalar_last(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert RoNIN scalar-first ``[w, x, y, z]`` -> repo scalar-last ``[x, y, z, w]``.

    This is the single, documented fix for the old convention-mismatch bug C3.
    ``q_wxyz`` is ``(T, 4)``; the returned array is ``(T, 4)`` and ready to hand
    to :mod:`ninav.geometry` (e.g. ``quat_to_rotmat``).
    """
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    if q_wxyz.ndim != 2 or q_wxyz.shape[1] != 4:
        raise ValueError(
            f"orientation must be (T, 4), got shape {q_wxyz.shape}"
        )
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    return np.stack([x, y, z, w], axis=1)


def load_ronin_recording(hdf5_path: str, rec_id: str | None = None) -> Recording:
    """Load one RoNIN ``data.hdf5`` file into a :class:`Recording`.

    Parameters
    ----------
    hdf5_path:
        Path to a RoNIN ``data.hdf5`` file.
    rec_id:
        Identifier for the recording. Defaults to the name of the parent
        directory (RoNIN's per-recording folder, e.g. ``a000_1``); falls back to
        the file stem.

    Returns
    -------
    Recording
        With scalar-last orientation (``ori``), world velocity (``vel``) computed
        as ``np.gradient(pos, dt)``, and a zero magnetometer if ``synced/magnet``
        is absent.
    """
    import h5py  # local import: h5py is only needed for real-data ingestion

    if rec_id is None:
        parent = os.path.basename(os.path.dirname(os.path.abspath(hdf5_path)))
        rec_id = parent or os.path.splitext(os.path.basename(hdf5_path))[0]

    with h5py.File(hdf5_path, "r") as f:
        time = np.asarray(f[_TIME_KEY], dtype=np.float64).reshape(-1)
        gyro = np.asarray(f[_GYRO_KEY], dtype=np.float64)
        acc = np.asarray(f[_ACCE_KEY], dtype=np.float64)
        pos = np.asarray(f[_POS_KEY], dtype=np.float64)
        ori_raw = np.asarray(f[_ORI_KEY], dtype=np.float64)
        # Magnetometer is optional in some RoNIN dumps -> zeros if missing.
        if _MAGNET_KEY in f:
            mag = np.asarray(f[_MAGNET_KEY], dtype=np.float64)
        else:
            mag = np.zeros_like(acc)

    n = time.shape[0]
    if not (gyro.shape == acc.shape == pos.shape == (n, 3)):
        raise ValueError(
            f"{hdf5_path}: expected (T, 3) gyro/acce/pos with T={n}, got "
            f"gyro={gyro.shape}, acce={acc.shape}, pos={pos.shape}"
        )
    if mag.shape != (n, 3):
        raise ValueError(
            f"{hdf5_path}: expected (T, 3) magnet with T={n}, got {mag.shape}"
        )

    # RoNIN tango_ori is scalar-first [w,x,y,z]; convert to scalar-last [x,y,z,w]
    # to match ninav.geometry (fixes bug C3).
    ori = _scalar_first_to_scalar_last(ori_raw)

    # Sampling rate from timestamps; RoNIN is nominally 200 Hz.
    if n >= 2:
        dt = float(np.median(np.diff(time)))
        if dt <= 0:
            raise ValueError(f"{hdf5_path}: non-positive sample interval dt={dt}")
    else:
        dt = 1.0 / 200.0
    fs = 1.0 / dt

    # World velocity by central differences of position (matches synthetic gen).
    vel = np.gradient(pos, dt, axis=0)

    return Recording(rec_id=rec_id, fs=fs, time=time, gyro=gyro, acc=acc,
                     mag=mag, pos=pos, ori=ori, vel=vel)


def load_ronin_folder(folder: str) -> list[Recording]:
    """Load every RoNIN recording under ``folder``.

    Iterates the immediate subdirectories of ``folder``; each subdirectory that
    contains a ``data.hdf5`` file becomes one :class:`Recording` (its directory
    name is the ``rec_id``). Subdirectories are visited in sorted order so the
    returned list is deterministic.
    """
    recordings: list[Recording] = []
    for name in sorted(os.listdir(folder)):
        subdir = os.path.join(folder, name)
        if not os.path.isdir(subdir):
            continue
        hdf5_path = os.path.join(subdir, "data.hdf5")
        if os.path.isfile(hdf5_path):
            recordings.append(load_ronin_recording(hdf5_path, rec_id=name))
    return recordings
