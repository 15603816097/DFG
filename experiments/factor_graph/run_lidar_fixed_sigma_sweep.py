from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace
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
# Project modules
# ============================================================

from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)

from src.factor_graph.sensorwise_reliability_graph import (
    ReliabilitySource,
    run_sensorwise_reliability_graph,
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

FACTOR = os.path.join(
    BENCH,
    "physical_lidar_factor_data.npz",
)

GT_PATH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)

OUT = os.path.join(
    ROOT,
    "results",
    "lidar_covariance_calibration_v1",
)

SWEEP_DIR = os.path.join(
    OUT,
    "fixed_sweep",
)

CSV_PATH = os.path.join(
    OUT,
    "fixed_sigma_sweep.csv",
)

BEST_SIGMA_PATH = os.path.join(
    OUT,
    "best_fixed_sigma.txt",
)

BEST_RESULT_PATH = os.path.join(
    OUT,
    "best_fixed_result.txt",
)


# ============================================================
# Sweep candidates
# ============================================================

SIGMA_CANDIDATES = [
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
]


# ============================================================
# Existing LiDAR covariance ratio
#
# original:
# translation sigma = 0.35
# rotation sigma    = 0.03
# ============================================================

ROT_TRANS_RATIO = 0.03 / 0.35


# ============================================================
# Physical LiDAR measurements
# ============================================================

def load_physical_measurements():

    print(
        "Loading degraded four-sensor measurements..."
    )

    base = load_degraded_four_sensor_measurements(
        BASE
    )

    measurements = SimpleNamespace(
        **dict(vars(base))
    )

    if not Path(FACTOR).exists():

        raise FileNotFoundError(
            f"Physical LiDAR factor file not found:\n{FACTOR}"
        )

    data = np.load(
        FACTOR,
        allow_pickle=False,
    )

    print()
    print(
        "Physical LiDAR factor arrays:"
    )

    for key in data.files:

        print(
            f"  {key:32s}",
            data[key].shape,
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
# Reliability configuration
#
# GPS    predictive
# IMU    predictive
# LiDAR  fixed
# Camera predictive
# ============================================================

def build_sources():

    return {
        "gps": ReliabilitySource(
            "predictive"
        ),

        "imu": ReliabilitySource(
            "predictive"
        ),

        "lidar": ReliabilitySource(
            "fixed"
        ),

        "camera": ReliabilitySource(
            "predictive"
        ),
    }


# ============================================================
# Evaluate trajectory
# ============================================================

def evaluate_trajectory(
    trajectory: np.ndarray,
    ground_truth: np.ndarray,
):

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float64,
    )

    n = min(
        len(trajectory),
        len(ground_truth),
    )

    trajectory = trajectory[:n]
    ground_truth = ground_truth[:n]

    error = (
        trajectory
        -
        ground_truth
    )

    error_3d = np.linalg.norm(
        error,
        axis=1,
    )

    error_2d = np.linalg.norm(
        error[:, :2],
        axis=1,
    )

    ate3d = float(
        np.sqrt(
            np.mean(
                error_3d ** 2
            )
        )
    )

    ate2d = float(
        np.sqrt(
            np.mean(
                error_2d ** 2
            )
        )
    )

    mean3d = float(
        np.mean(
            error_3d
        )
    )

    max3d = float(
        np.max(
            error_3d
        )
    )

    zrmse = float(
        np.sqrt(
            np.mean(
                error[:, 2] ** 2
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
# Build frozen config correctly
# ============================================================

def build_config(
    translation_sigma: float,
):

    rotation_sigma = (
        float(translation_sigma)
        *
        ROT_TRANS_RATIO
    )

    # FourSensorFactorConfig is a frozen dataclass.
    #
    # Therefore DO NOT do:
    #
    # config.lidar_translation_sigma = ...
    #
    # Instead use dataclasses.replace().
    base_config = FourSensorFactorConfig()

    config = replace(
        base_config,

        lidar_translation_sigma=
            float(
                translation_sigma
            ),

        lidar_rotation_sigma=
            float(
                rotation_sigma
            ),
    )

    return (
        config,
        rotation_sigma,
    )


# ============================================================
# Save current sweep result continuously
#
# This means if the program is interrupted at sigma 0.30,
# completed results are still preserved.
# ============================================================

def save_csv(
    results,
):

    fieldnames = [
        "sigma_translation",
        "sigma_rotation",
        "ATE3D",
        "ATE2D",
        "Mean3D",
        "Max3D",
        "ZRMSE",
        "gps_factors",
        "imu_factors",
        "lidar_factors",
        "camera_factors",
    ]

    Path(OUT).mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "LIDAR FIXED COVARIANCE SWEEP"
    )

    print(
        "=" * 120
    )

    print()
    print(
        "Purpose:"
    )

    print(
        "Find the best FIXED LiDAR covariance while keeping"
    )

    print(
        "GPS / IMU / Camera predictive settings unchanged."
    )

    # ========================================================
    # Basic checks
    # ========================================================

    if not Path(FACTOR).exists():

        raise FileNotFoundError(
            FACTOR
        )

    if not Path(GT_PATH).exists():

        raise FileNotFoundError(
            GT_PATH
        )

    if not Path(PRED).exists():

        raise FileNotFoundError(
            PRED
        )

    Path(OUT).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(SWEEP_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load data
    # ========================================================

    measurements = (
        load_physical_measurements()
    )

    ground_truth = np.loadtxt(
        GT_PATH,
        dtype=np.float64,
    )

    print()
    print(
        "Frames:",
        len(
            measurements.gps_local
        ),
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

    print(
        "Ground truth:",
        ground_truth.shape,
    )

    # ========================================================
    # Sweep
    # ========================================================

    results = []

    for index, translation_sigma in enumerate(
        SIGMA_CANDIDATES,
        start=1,
    ):

        config, rotation_sigma = (
            build_config(
                translation_sigma
            )
        )

        print()
        print(
            "#" * 120
        )

        print(
            f"[{index}/{len(SIGMA_CANDIDATES)}] "
            f"LiDAR fixed covariance"
        )

        print(
            "#" * 120
        )

        print(
            f"translation sigma = "
            f"{translation_sigma:.6f}"
        )

        print(
            f"rotation sigma    = "
            f"{rotation_sigma:.6f}"
        )

        # ====================================================
        # Verify the frozen config actually contains our values
        # ====================================================

        print()
        print(
            "Config check:"
        )

        print(
            "  config.lidar_translation_sigma =",
            config.lidar_translation_sigma,
        )

        print(
            "  config.lidar_rotation_sigma    =",
            config.lidar_rotation_sigma,
        )

        # ====================================================
        # Output path
        # ====================================================

        sigma_tag = (
            f"{translation_sigma:.2f}"
            .replace(
                ".",
                "p",
            )
        )

        case_dir = os.path.join(
            SWEEP_DIR,
            f"sigma_{sigma_tag}",
        )

        # ====================================================
        # Run factor graph
        # ====================================================

        trajectory, poses, counts = (
            run_sensorwise_reliability_graph(

                measurements,

                case_dir,

                build_sources(),

                predictive_dir=
                    PRED,

                oracle_path=
                    None,

                config=
                    config,
            )
        )

        # ====================================================
        # Evaluation
        # ====================================================

        metrics = evaluate_trajectory(
            trajectory,
            ground_truth,
        )

        result = {
            "sigma_translation":
                float(
                    translation_sigma
                ),

            "sigma_rotation":
                float(
                    rotation_sigma
                ),

            "ATE3D":
                metrics["ATE3D"],

            "ATE2D":
                metrics["ATE2D"],

            "Mean3D":
                metrics["Mean3D"],

            "Max3D":
                metrics["Max3D"],

            "ZRMSE":
                metrics["ZRMSE"],

            "gps_factors":
                int(
                    counts["gps"]
                ),

            "imu_factors":
                int(
                    counts["imu"]
                ),

            "lidar_factors":
                int(
                    counts["lidar"]
                ),

            "camera_factors":
                int(
                    counts["camera"]
                ),
        }

        results.append(
            result
        )

        # Save after every run.
        save_csv(
            results
        )

        print()
        print(
            "-" * 120
        )

        print(
            "CURRENT RESULT"
        )

        print(
            "-" * 120
        )

        print(
            f"ATE3D : "
            f"{result['ATE3D']:.6f}"
        )

        print(
            f"ATE2D : "
            f"{result['ATE2D']:.6f}"
        )

        print(
            f"Mean3D: "
            f"{result['Mean3D']:.6f}"
        )

        print(
            f"Max3D : "
            f"{result['Max3D']:.6f}"
        )

        print(
            f"ZRMSE : "
            f"{result['ZRMSE']:.6f}"
        )

    # ========================================================
    # Best fixed sigma
    # ========================================================

    best = min(
        results,
        key=lambda x:
            x["ATE3D"],
    )

    # ========================================================
    # Save best sigma
    #
    # IMPORTANT:
    # calibration script expects ONLY a floating point number.
    # ========================================================

    with open(
        BEST_SIGMA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"{best['sigma_translation']:.10f}\n"
        )

    # ========================================================
    # Save human-readable result
    # ========================================================

    with open(
        BEST_RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "BEST FIXED LIDAR COVARIANCE\n"
        )

        file.write(
            "=" * 80
            +
            "\n"
        )

        for key, value in best.items():

            file.write(
                f"{key}: {value}\n"
            )

    # ========================================================
    # Final result table
    # ========================================================

    print()
    print()
    print(
        "=" * 120
    )

    print(
        "FIXED LIDAR COVARIANCE SWEEP RESULTS"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Sigma T':>10s} "
        f"{'Sigma R':>10s} "
        f"{'ATE3D':>12s} "
        f"{'ATE2D':>12s} "
        f"{'Mean3D':>12s} "
        f"{'Max3D':>12s} "
        f"{'ZRMSE':>12s}"
    )

    print(
        "-" * 120
    )

    for result in results:

        marker = ""

        if np.isclose(
            result[
                "sigma_translation"
            ],
            best[
                "sigma_translation"
            ],
        ):

            marker = "  <-- BEST"

        print(
            f"{result['sigma_translation']:10.3f} "
            f"{result['sigma_rotation']:10.5f} "
            f"{result['ATE3D']:12.6f} "
            f"{result['ATE2D']:12.6f} "
            f"{result['Mean3D']:12.6f} "
            f"{result['Max3D']:12.6f} "
            f"{result['ZRMSE']:12.6f}"
            f"{marker}"
        )

    print()
    print(
        "=" * 120
    )

    print(
        "BEST FIXED LIDAR COVARIANCE"
    )

    print(
        "=" * 120
    )

    print(
        f"Translation sigma : "
        f"{best['sigma_translation']:.6f}"
    )

    print(
        f"Rotation sigma    : "
        f"{best['sigma_rotation']:.6f}"
    )

    print(
        f"ATE3D             : "
        f"{best['ATE3D']:.6f}"
    )

    print(
        f"ATE2D             : "
        f"{best['ATE2D']:.6f}"
    )

    print(
        f"Mean3D            : "
        f"{best['Mean3D']:.6f}"
    )

    print(
        f"Max3D             : "
        f"{best['Max3D']:.6f}"
    )

    print(
        f"ZRMSE             : "
        f"{best['ZRMSE']:.6f}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        CSV_PATH
    )

    print(
        BEST_SIGMA_PATH
    )

    print(
        BEST_RESULT_PATH
    )

    print()
    print(
        "=" * 120
    )

    print(
        "NEXT STEP"
    )

    print(
        "=" * 120
    )

    print(
        "python "
        "experiments/lidar_predictive/"
        "calibrate_lidar_error_covariance_mapping.py"
    )


if __name__ == "__main__":
    main()
