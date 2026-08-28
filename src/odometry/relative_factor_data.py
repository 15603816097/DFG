from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .body_frame_converter import (
    project_to_se3,
    rotation_angle,
)


@dataclass
class RelativeFactorData:
    coordinate_transforms: np.ndarray
    between_measurements: np.ndarray
    translations: np.ndarray
    rotations: np.ndarray
    valid: np.ndarray
    quality: np.ndarray

    def __len__(self):
        return int(self.between_measurements.shape[0])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            coordinate_transforms=self.coordinate_transforms,
            between_measurements=self.between_measurements,
            translations=self.translations,
            rotations=self.rotations,
            valid=self.valid,
            quality=self.quality,
        )

    @classmethod
    def load(cls, path):
        data = np.load(
            path,
            allow_pickle=False,
        )

        return cls(
            coordinate_transforms=data["coordinate_transforms"],
            between_measurements=data["between_measurements"],
            translations=data["translations"],
            rotations=data["rotations"],
            valid=data["valid"].astype(bool),
            quality=data["quality"],
        )


def normalize_quality(values, valid=None):
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if valid is None:
        valid = np.ones(
            len(values),
            dtype=bool,
        )
    else:
        valid = np.asarray(
            valid,
            dtype=bool,
        ).reshape(-1)

    result = np.zeros(
        len(values),
        dtype=np.float64,
    )

    mask = (
        valid
        &
        np.isfinite(values)
    )

    if not np.any(mask):
        return result

    selected = values[mask]

    low = float(
        np.percentile(
            selected,
            5.0,
        )
    )

    high = float(
        np.percentile(
            selected,
            95.0,
        )
    )

    if high <= low + 1e-12:
        result[mask] = 1.0
        return result

    result[mask] = np.clip(
        (
            values[mask]
            -
            low
        )
        /
        (
            high
            -
            low
        ),
        0.0,
        1.0,
    )

    return result


def make_factor_data(
    coordinate_transforms,
    between_measurements,
    valid,
    quality,
):
    coordinate_transforms = np.asarray(
        coordinate_transforms,
        dtype=np.float64,
    )

    between_measurements = np.asarray(
        between_measurements,
        dtype=np.float64,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    ).reshape(-1)

    quality = np.asarray(
        quality,
        dtype=np.float64,
    ).reshape(-1)

    n = len(valid)

    if coordinate_transforms.shape != (n, 4, 4):
        raise ValueError(
            "coordinate_transforms must have shape (N,4,4)"
        )

    if between_measurements.shape != (n, 4, 4):
        raise ValueError(
            "between_measurements must have shape (N,4,4)"
        )

    if quality.shape != (n,):
        raise ValueError(
            "quality must have shape (N,)"
        )

    translations = between_measurements[
        :,
        :3,
        3,
    ].copy()

    rotations = between_measurements[
        :,
        :3,
        :3,
    ].copy()

    return RelativeFactorData(
        coordinate_transforms=coordinate_transforms,
        between_measurements=between_measurements,
        translations=translations,
        rotations=rotations,
        valid=valid,
        quality=quality,
    )


def motion_disagreement(
    lidar_between,
    camera_between,
):
    lidar_between = project_to_se3(
        lidar_between
    )

    camera_between = project_to_se3(
        camera_between
    )

    t_error = float(
        np.linalg.norm(
            lidar_between[:3, 3]
            -
            camera_between[:3, 3]
        )
    )

    R_error = (
        lidar_between[:3, :3].T
        @
        camera_between[:3, :3]
    )

    r_error_rad = rotation_angle(
        R_error
    )

    return (
        t_error,
        float(
            np.degrees(
                r_error_rad
            )
        ),
    )
