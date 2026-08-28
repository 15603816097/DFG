from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# ============================================================
# Project root
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================
# Existing project modules
# ============================================================

from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)

from src.factor_graph.sensorwise_reliability_graph import (
    ReliabilitySource,
    _timestamp_seconds,
    _pose3_from_matrix,
    _pose3_to_matrix,
    _imu_between_pose,
    _initial_trajectory,
    _prepare_reliability,
    _gps_sigma,
    _imu_sigma,
    _camera_sigmas,
)

from src.factor_graph.sensor_factor_config import (
    FourSensorFactorConfig,
)


# ============================================================
# Paths
# ============================================================

BASE = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

PRED = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

BENCH = os.path.join(
    ROOT,
    "results",
    "lidar_physical_oracle_benchmark",
)

PHYSICAL_FACTOR = os.path.join(
    BENCH,
    "physical_lidar_factor_data.npz",
)

MAPPING = os.path.join(
    ROOT,
    "results",
    "lidar_covariance_calibration_v1",
    "mapping",
    "lidar_dynamic_mapping.npz",
)

OUT = os.path.join(
    ROOT,
    "results",
    "lidar_covariance_calibration_v1",
    "predictive_dynamic_fg",
)


# ============================================================
# Load physical measurements
# ============================================================

def load_physical_measurements():

    print(
        "Loading degraded GPS / IMU / Camera measurements..."
    )

    base = load_degraded_four_sensor_measurements(
        BASE
    )

    measurements = SimpleNamespace(
        **dict(vars(base))
    )

    if not Path(PHYSICAL_FACTOR).exists():
        raise FileNotFoundError(
            PHYSICAL_FACTOR
        )

    print(
        "Loading physical LiDAR factors..."
    )

    data = np.load(
        PHYSICAL_FACTOR,
        allow_pickle=False,
    )

    measurements.lidar_between = np.asarray(
        data["body_between"],
        dtype=np.float64,
    )

    measurements.lidar_quality = np.asarray(
        data["quality"],
        dtype=np.float64,
    )

    measurements.lidar_valid = np.asarray(
        data["converged"],
        dtype=bool,
    )

    return measurements


# ============================================================
# Load calibrated dynamic covariance
# ============================================================

def load_dynamic_lidar_covariance(
    expected_pairs,
):

    if not Path(MAPPING).exists():

        raise FileNotFoundError(
            "Dynamic LiDAR mapping not found:\n"
            f"{MAPPING}\n\n"
            "Run:\n"
            "python experiments/lidar_predictive/"
            "calibrate_lidar_error_covariance_mapping.py"
        )

    data = np.load(
        MAPPING,
        allow_pickle=False,
    )

    required = [
        "translation_sigma_pair",
        "rotation_sigma_pair",
    ]

    for key in required:

        if key not in data.files:

            raise KeyError(
                f"Missing '{key}' in {MAPPING}\n"
                f"Available arrays: {data.files}"
            )

    sigma_t = np.asarray(
        data["translation_sigma_pair"],
        dtype=np.float64,
    ).reshape(-1)

    sigma_r = np.asarray(
        data["rotation_sigma_pair"],
        dtype=np.float64,
    ).reshape(-1)

    if len(sigma_t) != expected_pairs:

        raise ValueError(
            "LiDAR translation sigma length mismatch: "
            f"{len(sigma_t)} != {expected_pairs}"
        )

    if len(sigma_r) != expected_pairs:

        raise ValueError(
            "LiDAR rotation sigma length mismatch: "
            f"{len(sigma_r)} != {expected_pairs}"
        )

    if not np.all(
        np.isfinite(
            sigma_t
        )
    ):

        raise ValueError(
            "translation_sigma_pair contains NaN/Inf"
        )

    if not np.all(
        np.isfinite(
            sigma_r
        )
    ):

        raise ValueError(
            "rotation_sigma_pair contains NaN/Inf"
        )

    if np.any(
        sigma_t <= 0.0
    ):

        raise ValueError(
            "translation_sigma_pair contains non-positive values"
        )

    if np.any(
        sigma_r <= 0.0
    ):

        raise ValueError(
            "rotation_sigma_pair contains non-positive values"
        )

    return (
        sigma_t,
        sigma_r,
    )


# ============================================================
# Sensor modes
#
# GPS    predictive
# IMU    predictive
# LiDAR  DIRECT dynamic covariance
# Camera predictive
# ============================================================

def build_sources():

    # LiDAR is marked fixed here ONLY so that
    # _prepare_reliability() does not try to load
    # a LiDAR predictive reliability file.
    #
    # Its covariance is NOT fixed in the graph below.
    # sigma_t[i] / sigma_r[i] are injected directly.
    return {

        "gps":
            ReliabilitySource(
                "predictive"
            ),

        "imu":
            ReliabilitySource(
                "predictive"
            ),

        "lidar":
            ReliabilitySource(
                "fixed"
            ),

        "camera":
            ReliabilitySource(
                "predictive"
            ),
    }


# ============================================================
# Run graph
# ============================================================

def run_dynamic_graph(
    measurements,
    sigma_t_pair,
    sigma_r_pair,
    output_dir,
):

    try:
        import gtsam

    except Exception as exc:

        raise ImportError(
            "gtsam is required"
        ) from exc

    config = FourSensorFactorConfig()

    sources = build_sources()

    n = len(
        measurements.gps_local
    )

    if n < 2:

        raise ValueError(
            "Need at least 2 frames"
        )

    if len(sigma_t_pair) != n - 1:

        raise ValueError(
            f"sigma_t_pair length={len(sigma_t_pair)}, "
            f"expected {n - 1}"
        )

    if len(sigma_r_pair) != n - 1:

        raise ValueError(
            f"sigma_r_pair length={len(sigma_r_pair)}, "
            f"expected {n - 1}"
        )

    # ========================================================
    # GPS / IMU / Camera reliability
    # ========================================================

    reliability = _prepare_reliability(
        n,
        sources,
        predictive_dir=PRED,
        oracle_path=None,
    )

    # ========================================================
    # Graph + initial values
    # ========================================================

    graph = gtsam.NonlinearFactorGraph()

    initial = gtsam.Values()

    poses0 = _initial_trajectory(
        gtsam,
        measurements,
    )

    # ========================================================
    # Prior
    # ========================================================

    prior_noise = (
        gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    config.prior_rotation_sigma,
                    config.prior_rotation_sigma,
                    config.prior_rotation_sigma,

                    config.prior_translation_sigma,
                    config.prior_translation_sigma,
                    config.prior_translation_sigma,
                ],
                dtype=np.float64,
            )
        )
    )

    graph.add(
        gtsam.PriorFactorPose3(
            0,
            gtsam.Pose3(),
            prior_noise,
        )
    )

    # ========================================================
    # Initial trajectory
    # ========================================================

    for i in range(n):

        initial.insert(
            i,
            poses0[i],
        )

    # ========================================================
    # Time
    # ========================================================

    times = _timestamp_seconds(
        measurements.timestamps
    )

    # ========================================================
    # Statistics
    # ========================================================

    counts = {
        "gps": 0,
        "imu": 0,
        "lidar": 0,
        "camera": 0,
    }

    sigma_stats = {
        "gps": [],
        "imu": [],
        "lidar_translation": [],
        "lidar_rotation": [],
        "camera": [],
    }

    # ========================================================
    # Add factors
    # ========================================================

    for i in range(n):

        # ----------------------------------------------------
        # GPS predictive
        # ----------------------------------------------------

        gps_r = reliability[
            "gps"
        ][i]

        gps_sigma = _gps_sigma(
            config,
            "predictive",
            gps_r,
        )

        sigma_stats[
            "gps"
        ].append(
            gps_sigma
        )

        gps_noise = (
            gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    [
                        1e6,
                        1e6,
                        1e6,

                        gps_sigma,
                        gps_sigma,
                        gps_sigma,
                    ],
                    dtype=np.float64,
                )
            )
        )

        gps_pose = gtsam.Pose3(
            gtsam.Rot3(),

            gtsam.Point3(
                float(
                    measurements.gps_local[
                        i,
                        0
                    ]
                ),

                float(
                    measurements.gps_local[
                        i,
                        1
                    ]
                ),

                float(
                    measurements.gps_local[
                        i,
                        2
                    ]
                ),
            ),
        )

        graph.add(
            gtsam.PriorFactorPose3(
                i,
                gps_pose,
                gps_noise,
            )
        )

        counts[
            "gps"
        ] += 1

        # Last frame has no between factors.
        if i >= n - 1:
            continue

        # ----------------------------------------------------
        # dt
        # ----------------------------------------------------

        if len(times) == n:

            dt = float(
                times[
                    i + 1
                ]
                -
                times[
                    i
                ]
            )

        else:

            dt = 0.1

        if (
            not np.isfinite(
                dt
            )
            or
            dt <= 0.0
            or
            dt > 1.0
        ):

            dt = 0.1

        # ----------------------------------------------------
        # IMU predictive
        # ----------------------------------------------------

        imu_r = reliability[
            "imu"
        ][
            i + 1
        ]

        imu_sigma = _imu_sigma(
            config,
            "predictive",
            imu_r,
        )

        sigma_stats[
            "imu"
        ].append(
            imu_sigma
        )

        imu_noise = (
            gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    [
                        imu_sigma,
                        imu_sigma,
                        imu_sigma,

                        config.imu_translation_sigma,
                        config.imu_translation_sigma,
                        config.imu_translation_sigma,
                    ],
                    dtype=np.float64,
                )
            )
        )

        graph.add(
            gtsam.BetweenFactorPose3(
                i,
                i + 1,

                _imu_between_pose(
                    gtsam,
                    measurements.imu_gyro[i],
                    dt,
                ),

                imu_noise,
            )
        )

        counts[
            "imu"
        ] += 1

        # ----------------------------------------------------
        # LiDAR
        #
        # DIRECT calibrated dynamic covariance.
        #
        # IMPORTANT:
        #
        # Pair i corresponds exactly to:
        #
        # pose i -> pose i+1
        #
        # therefore use sigma_pair[i].
        # ----------------------------------------------------

        if (
            bool(
                measurements.lidar_valid[
                    i
                ]
            )
            and
            float(
                measurements.lidar_quality[
                    i
                ]
            )
            >=
            config.lidar_quality_min
        ):

            lidar_trans_sigma = float(
                sigma_t_pair[
                    i
                ]
            )

            lidar_rot_sigma = float(
                sigma_r_pair[
                    i
                ]
            )

            sigma_stats[
                "lidar_translation"
            ].append(
                lidar_trans_sigma
            )

            sigma_stats[
                "lidar_rotation"
            ].append(
                lidar_rot_sigma
            )

            lidar_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.asarray(
                        [
                            lidar_rot_sigma,
                            lidar_rot_sigma,
                            lidar_rot_sigma,

                            lidar_trans_sigma,
                            lidar_trans_sigma,
                            lidar_trans_sigma,
                        ],
                        dtype=np.float64,
                    )
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,

                    _pose3_from_matrix(
                        gtsam,
                        measurements.lidar_between[
                            i
                        ],
                    ),

                    lidar_noise,
                )
            )

            counts[
                "lidar"
            ] += 1

        # ----------------------------------------------------
        # Camera predictive
        # ----------------------------------------------------

        if (
            bool(
                measurements.camera_valid[
                    i
                ]
            )
            and
            float(
                measurements.camera_quality[
                    i
                ]
            )
            >=
            config.camera_quality_min
        ):

            camera_r = reliability[
                "camera"
            ][
                i + 1
            ]

            (
                camera_rot_sigma,
                camera_trans_sigma,
            ) = _camera_sigmas(
                config,
                "predictive",
                camera_r,
            )

            sigma_stats[
                "camera"
            ].append(
                camera_trans_sigma
            )

            camera_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.asarray(
                        [
                            camera_rot_sigma,
                            camera_rot_sigma,
                            camera_rot_sigma,

                            camera_trans_sigma,
                            camera_trans_sigma,
                            camera_trans_sigma,
                        ],
                        dtype=np.float64,
                    )
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,

                    _pose3_from_matrix(
                        gtsam,
                        measurements.camera_between[
                            i
                        ],
                    ),

                    camera_noise,
                )
            )

            counts[
                "camera"
            ] += 1

        if i % 500 == 0:

            print(
                "Add factors:",
                i,
            )

    # ========================================================
    # Graph diagnostics
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "GRAPH DIAGNOSTICS"
    )

    print(
        "=" * 120
    )

    print(
        "Graph factors:",
        graph.size(),
    )

    print(
        "Factor counts:",
        counts,
    )

    print(
        "Modes:"
    )

    print(
        {
            "gps": "predictive",
            "imu": "predictive",
            "lidar": "calibrated_dynamic_direct_sigma",
            "camera": "predictive",
        }
    )

    # --------------------------------------------------------
    # Sigma diagnostics
    # --------------------------------------------------------

    for key in [
        "gps",
        "imu",
        "lidar_translation",
        "lidar_rotation",
        "camera",
    ]:

        values = np.asarray(
            sigma_stats[
                key
            ],
            dtype=np.float64,
        )

        if len(values) == 0:
            continue

        print(
            f"{key:20s} sigma min/max/mean: "
            f"{values.min():.6f} / "
            f"{values.max():.6f} / "
            f"{values.mean():.6f}"
        )

    # ========================================================
    # Verify dynamic LiDAR actually entered graph
    # ========================================================

    lidar_used = np.asarray(
        sigma_stats[
            "lidar_translation"
        ],
        dtype=np.float64,
    )

    if len(lidar_used) == 0:

        raise RuntimeError(
            "No LiDAR factors were added."
        )

    if np.ptp(
        lidar_used
    ) <= 1e-12:

        raise RuntimeError(
            "LiDAR covariance is constant. "
            "Dynamic mapping was not applied correctly."
        )

    print()
    print(
        "SUCCESS:"
    )

    print(
        "Dynamic LiDAR covariance entered GTSAM directly."
    )

    # ========================================================
    # Optimize
    # ========================================================

    params = (
        gtsam.LevenbergMarquardtParams()
    )

    params.setMaxIterations(
        100
    )

    params.setRelativeErrorTol(
        1e-7
    )

    print()
    print(
        "Optimizing..."
    )

    result = (
        gtsam.LevenbergMarquardtOptimizer(
            graph,
            initial,
            params,
        ).optimize()
    )

    # ========================================================
    # Extract trajectory
    # ========================================================

    trajectory = np.zeros(
        (
            n,
            3,
        ),
        dtype=np.float64,
    )

    poses = np.zeros(
        (
            n,
            4,
            4,
        ),
        dtype=np.float64,
    )

    for i in range(n):

        pose = result.atPose3(
            i
        )

        T = _pose3_to_matrix(
            pose
        )

        poses[
            i
        ] = T

        trajectory[
            i
        ] = T[
            :3,
            3
        ]

    # ========================================================
    # Save
    # ========================================================

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        output_dir
        /
        "trajectory.txt",

        trajectory,

        fmt="%.8f",
    )

    np.save(
        output_dir
        /
        "poses.npy",

        poses,
    )

    np.savez_compressed(
        output_dir
        /
        "used_lidar_dynamic_covariance.npz",

        translation_sigma_pair=
            sigma_t_pair,

        rotation_sigma_pair=
            sigma_r_pair,

        used_translation_sigma=
            np.asarray(
                sigma_stats[
                    "lidar_translation"
                ],
                dtype=np.float64,
            ),

        used_rotation_sigma=
            np.asarray(
                sigma_stats[
                    "lidar_rotation"
                ],
                dtype=np.float64,
            ),
    )

    with (
        output_dir
        /
        "modes.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "gps=predictive\n"
        )

        file.write(
            "imu=predictive\n"
        )

        file.write(
            "lidar=calibrated_dynamic_direct_sigma\n"
        )

        file.write(
            "camera=predictive\n"
        )

    return (
        trajectory,
        poses,
        counts,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    trajectory,
):

    gt_path = os.path.join(
        ROOT,
        "results",
        "ground_truth",
        "trajectory.txt",
    )

    if not Path(
        gt_path
    ).exists():

        raise FileNotFoundError(
            gt_path
        )

    gt = np.loadtxt(
        gt_path,
        dtype=np.float64,
    )

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    n = min(
        len(gt),
        len(trajectory),
    )

    gt = gt[
        :n
    ]

    trajectory = trajectory[
        :n
    ]

    error = (
        trajectory
        -
        gt
    )

    error3d = np.linalg.norm(
        error,
        axis=1,
    )

    error2d = np.linalg.norm(
        error[
            :,
            :2
        ],
        axis=1,
    )

    ate3d = float(
        np.sqrt(
            np.mean(
                error3d ** 2
            )
        )
    )

    ate2d = float(
        np.sqrt(
            np.mean(
                error2d ** 2
            )
        )
    )

    mean3d = float(
        np.mean(
            error3d
        )
    )

    max3d = float(
        np.max(
            error3d
        )
    )

    zrmse = float(
        np.sqrt(
            np.mean(
                error[
                    :,
                    2
                ] ** 2
            )
        )
    )

    return {
        "ATE3D": ate3d,
        "ATE2D": ate2d,
        "Mean3D": mean3d,
        "Max3D": max3d,
        "ZRMSE": zrmse,
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "CALIBRATED PREDICTIVE DYNAMIC LIDAR FACTOR GRAPH"
    )

    print(
        "=" * 120
    )

    # ========================================================
    # Load physical measurements
    # ========================================================

    measurements = (
        load_physical_measurements()
    )

    n = len(
        measurements.gps_local
    )

    print()
    print(
        "Frames:",
        n,
    )

    print(
        "LiDAR pairs:",
        n - 1,
    )

    print(
        "LiDAR valid:",
        int(
            measurements.lidar_valid.sum()
        ),
        "/",
        len(
            measurements.lidar_valid
        ),
    )

    # ========================================================
    # Load dynamic covariance
    # ========================================================

    (
        sigma_t_pair,
        sigma_r_pair,
    ) = load_dynamic_lidar_covariance(
        n - 1
    )

    print()
    print(
        "=" * 120
    )

    print(
        "INPUT DYNAMIC COVARIANCE"
    )

    print(
        "=" * 120
    )

    print(
        "Translation sigma min/max/mean:"
    )

    print(
        f"  {sigma_t_pair.min():.6f} / "
        f"{sigma_t_pair.max():.6f} / "
        f"{sigma_t_pair.mean():.6f}"
    )

    print(
        "Rotation sigma min/max/mean:"
    )

    print(
        f"  {sigma_r_pair.min():.6f} / "
        f"{sigma_r_pair.max():.6f} / "
        f"{sigma_r_pair.mean():.6f}"
    )

    print(
        "Dynamic factors:",
        int(
            np.sum(
                sigma_t_pair
                >
                (
                    sigma_t_pair.min()
                    +
                    1e-12
                )
            )
        ),
        "/",
        len(
            sigma_t_pair
        ),
    )

    # ========================================================
    # Run graph
    # ========================================================

    trajectory, poses, counts = (
        run_dynamic_graph(
            measurements,
            sigma_t_pair,
            sigma_r_pair,
            OUT,
        )
    )

    # ========================================================
    # Evaluate
    # ========================================================

    metrics = evaluate(
        trajectory
    )

    # ========================================================
    # Save metrics
    # ========================================================

    metrics_path = os.path.join(
        OUT,
        "metrics.txt",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        for key, value in metrics.items():

            file.write(
                f"{key}: {value:.10f}\n"
            )

    # ========================================================
    # Comparison
    # ========================================================

    best_fixed = 1.977235
    gt_oracle = 1.890239

    predictive = metrics[
        "ATE3D"
    ]

    improvement_vs_fixed = (
        (
            best_fixed
            -
            predictive
        )
        /
        best_fixed
        *
        100.0
    )

    gap_vs_oracle = (
        (
            predictive
            -
            gt_oracle
        )
        /
        gt_oracle
        *
        100.0
    )

    print()
    print()
    print(
        "=" * 120
    )

    print(
        "FINAL FAIR LIDAR COMPARISON"
    )

    print(
        "=" * 120
    )

    print(
        f"Best Fixed LiDAR:"
        f"             {best_fixed:.6f} m"
    )

    print(
        f"Predictive Dynamic LiDAR:"
        f"      {predictive:.6f} m"
    )

    print(
        f"GT-Oracle Dynamic LiDAR:"
        f"        {gt_oracle:.6f} m"
    )

    print()

    print(
        f"Improvement vs Best Fixed:"
        f"     {improvement_vs_fixed:+.2f}%"
    )

    print(
        f"Gap vs GT Oracle:"
        f"               {gap_vs_oracle:+.2f}%"
    )

    print()

    print(
        "ATE2D :",
        f"{metrics['ATE2D']:.6f}",
    )

    print(
        "Mean3D:",
        f"{metrics['Mean3D']:.6f}",
    )

    print(
        "Max3D :",
        f"{metrics['Max3D']:.6f}",
    )

    print(
        "ZRMSE :",
        f"{metrics['ZRMSE']:.6f}",
    )

    print()

    if predictive < best_fixed:

        print(
            "SUCCESS:"
        )

        print(
            "Predictive dynamic covariance beats "
            "the optimized fixed LiDAR covariance."
        )

    else:

        print(
            "RESULT:"
        )

        print(
            "Predictive dynamic covariance does NOT yet "
            "beat the optimized fixed LiDAR covariance."
        )

        print(
            "Next step should be covariance-mapping calibration, "
            "NOT retraining the predictor."
        )

    print()
    print(
        "Saved:"
    )

    print(
        OUT
    )


if __name__ == "__main__":

    main()
