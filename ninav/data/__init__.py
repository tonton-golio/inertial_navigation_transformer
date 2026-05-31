"""Data: synthetic generator, windowing, targets, splits, normalization, dataset."""
from .synthetic import Recording, generate_dataset, generate_recording
from .windowing import create_windows, valid_starts
from .targets import (
    displacement_target,
    make_target,
    polar_target,
    start_offsets,
    velocity_target,
)
from .splits import split_recordings
from .normalize import FeatureScaler
from .augment import apply_yaw, augment_window
from .dataset import (
    WindowDataset,
    build_dataset,
    prepare_recording,
    world_frame_imu,
)

__all__ = [
    "Recording", "generate_dataset", "generate_recording",
    "create_windows", "valid_starts",
    "displacement_target", "make_target", "polar_target", "start_offsets",
    "velocity_target",
    "split_recordings", "FeatureScaler", "apply_yaw", "augment_window",
    "WindowDataset", "build_dataset", "prepare_recording", "world_frame_imu",
]
