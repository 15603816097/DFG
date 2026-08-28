from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rigid_transform import (
    estimate_rigid_transform_svd,
    transform_points,
)


@dataclass
class LidarOdometryResult:
    transform: np.ndarray
    fitness: float
    rmse: float
    correspondences: int
    converged: bool


class LidarOdometry:
    """
    Lightweight point-to-point ICP for consecutive KITTI Velodyne scans.

    Returned transform:
        T_target_source

    so that:
        p_target ~= T_target_source @ p_source

    In the experiment scripts:
        source = frame t
        target = frame t+1

    Therefore the transform maps points from LiDAR frame t into
    LiDAR frame t+1.
    """

    def __init__(
        self,
        max_iterations=20,
        max_correspondence_distance=1.5,
        convergence_translation=1e-4,
        convergence_rotation=1e-4,
        min_correspondences=80,
        max_points=6000,
        min_range=2.0,
        max_range=50.0,
        z_min=-3.0,
        z_max=2.0,
        random_seed=20260826,
    ):
        self.max_iterations = int(
            max_iterations
        )

        self.max_correspondence_distance = float(
            max_correspondence_distance
        )

        self.convergence_translation = float(
            convergence_translation
        )

        self.convergence_rotation = float(
            convergence_rotation
        )

        self.min_correspondences = int(
            min_correspondences
        )

        self.max_points = int(
            max_points
        )

        self.min_range = float(
            min_range
        )

        self.max_range = float(
            max_range
        )

        self.z_min = float(
            z_min
        )

        self.z_max = float(
            z_max
        )

        self.random_seed = int(
            random_seed
        )

    def _prepare_points(
        self,
        points,
        frame_id=0,
    ):
        points = np.asarray(
            points
        )

        if (
            points.ndim != 2
            or points.shape[1] < 3
        ):
            raise ValueError(
                "LiDAR points must be N x 3 or N x 4"
            )

        xyz = np.asarray(
            points[
                :,
                :3
            ],
            dtype=np.float64,
        )

        finite = np.all(
            np.isfinite(
                xyz
            ),
            axis=1,
        )

        xyz = xyz[
            finite
        ]

        if len(
            xyz
        ) == 0:
            return xyz

        distance = np.linalg.norm(
            xyz,
            axis=1,
        )

        keep = (
            (distance >= self.min_range)
            &
            (distance <= self.max_range)
            &
            (xyz[:, 2] >= self.z_min)
            &
            (xyz[:, 2] <= self.z_max)
        )

        xyz = xyz[
            keep
        ]

        if (
            self.max_points > 0
            and
            len(
                xyz
            ) > self.max_points
        ):
            rng = np.random.default_rng(
                self.random_seed
                +
                int(
                    frame_id
                )
                *
                97
            )

            indices = rng.choice(
                len(
                    xyz
                ),
                size=
                    self.max_points,
                replace=False,
            )

            xyz = xyz[
                indices
            ]

        return xyz

    @staticmethod
    def _nearest_neighbor(
        source,
        target,
    ):
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(
                target
            )

            distance, index = tree.query(
                source,
                k=1,
                workers=-1,
            )

            return (
                np.asarray(
                    distance,
                    dtype=np.float64,
                ),
                np.asarray(
                    index,
                    dtype=np.int64,
                ),
            )

        except Exception:
            # Dependency-free fallback.
            distances = np.empty(
                len(
                    source
                ),
                dtype=np.float64,
            )

            indices = np.empty(
                len(
                    source
                ),
                dtype=np.int64,
            )

            chunk = 256

            for start in range(
                0,
                len(
                    source
                ),
                chunk,
            ):
                end = min(
                    start + chunk,
                    len(
                        source
                    ),
                )

                diff = (
                    source[
                        start:end,
                        None,
                        :
                    ]
                    -
                    target[
                        None,
                        :,
                        :
                    ]
                )

                d2 = np.sum(
                    diff
                    *
                    diff,
                    axis=2,
                )

                local_index = np.argmin(
                    d2,
                    axis=1,
                )

                indices[
                    start:end
                ] = local_index

                distances[
                    start:end
                ] = np.sqrt(
                    d2[
                        np.arange(
                            end
                            -
                            start
                        ),
                        local_index,
                    ]
                )

            return (
                distances,
                indices,
            )

    @staticmethod
    def _rotation_angle(
        R,
    ):
        value = (
            np.trace(
                R
            )
            -
            1.0
        ) / 2.0

        value = np.clip(
            value,
            -1.0,
            1.0,
        )

        return float(
            np.arccos(
                value
            )
        )

    def estimate(
        self,
        source_points,
        target_points,
        initial_transform=None,
        source_frame_id=0,
        target_frame_id=1,
    ):
        source = self._prepare_points(
            source_points,
            source_frame_id,
        )

        target = self._prepare_points(
            target_points,
            target_frame_id,
        )

        if (
            len(
                source
            ) < self.min_correspondences
            or
            len(
                target
            ) < self.min_correspondences
        ):
            return LidarOdometryResult(
                transform=
                    np.eye(
                        4,
                        dtype=np.float64,
                    ),
                fitness=0.0,
                rmse=float(
                    "inf"
                ),
                correspondences=0,
                converged=False,
            )

        if initial_transform is None:
            T = np.eye(
                4,
                dtype=np.float64,
            )
        else:
            T = np.asarray(
                initial_transform,
                dtype=np.float64,
            ).copy()

        converged = False
        last_rmse = float(
            "inf"
        )
        last_count = 0

        for _ in range(
            self.max_iterations
        ):
            transformed = (
                transform_points(
                    source,
                    T,
                )
            )

            distance, index = (
                self._nearest_neighbor(
                    transformed,
                    target,
                )
            )

            keep = (
                distance
                <=
                self.max_correspondence_distance
            )

            count = int(
                np.sum(
                    keep
                )
            )

            if (
                count
                <
                self.min_correspondences
            ):
                break

            source_match = (
                transformed[
                    keep
                ]
            )

            target_match = (
                target[
                    index[
                        keep
                    ]
                ]
            )

            delta = (
                estimate_rigid_transform_svd(
                    source_match,
                    target_match,
                )
            )

            T = (
                delta
                @
                T
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        distance[
                            keep
                        ]
                        ** 2
                    )
                )
            )

            translation_step = float(
                np.linalg.norm(
                    delta[
                        :3,
                        3
                    ]
                )
            )

            rotation_step = (
                self._rotation_angle(
                    delta[
                        :3,
                        :3
                    ]
                )
            )

            last_rmse = rmse
            last_count = count

            if (
                translation_step
                <
                self.convergence_translation
                and
                rotation_step
                <
                self.convergence_rotation
            ):
                converged = True
                break

        transformed = (
            transform_points(
                source,
                T,
            )
        )

        distance, _ = (
            self._nearest_neighbor(
                transformed,
                target,
            )
        )

        inlier = (
            distance
            <=
            self.max_correspondence_distance
        )

        fitness = float(
            np.mean(
                inlier
            )
        )

        if np.any(
            inlier
        ):
            final_rmse = float(
                np.sqrt(
                    np.mean(
                        distance[
                            inlier
                        ]
                        ** 2
                    )
                )
            )
        else:
            final_rmse = float(
                "inf"
            )

        return LidarOdometryResult(
            transform=T,
            fitness=
                fitness,
            rmse=
                final_rmse
                if np.isfinite(
                    final_rmse
                )
                else last_rmse,
            correspondences=
                int(
                    np.sum(
                        inlier
                    )
                )
                if np.any(
                    inlier
                )
                else last_count,
            converged=
                converged
                or
                (
                    fitness
                    >
                    0.15
                    and
                    np.isfinite(
                        final_rmse
                    )
                ),
        )
