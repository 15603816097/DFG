from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

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

from experiments.factor_graph.run_lidar_calibrated_dynamic_fg import (
    load_physical_measurements,
    run_dynamic_graph,
    evaluate,
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

PREDICTIVE_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

CALIB_DIR = os.path.join(
    ROOT,
    "results",
    "lidar_covariance_calibration_v1",
)

MAPPING_PATH = os.path.join(
    CALIB_DIR,
    "mapping",
    "lidar_dynamic_mapping.npz",
)

OUT_ROOT = os.path.join(
    ROOT,
    "results",
    "final_multisensor_predictive_dfg",
)

CSV_PATH = os.path.join(
    OUT_ROOT,
    "final_multisensor_comparison.csv",
)

SUMMARY_PATH = os.path.join(
    OUT_ROOT,
    "final_multisensor_summary.txt",
)


# ============================================================
# Final LiDAR mapping
#
# Validated best:
#   sigma_t0 = 0.15
#   sigma_r0 = 0.012857...
#   lambda   = 1.0
#
# sigma = sigma0 * (1 + lambda * risk^2)
# ============================================================

LIDAR_SIGMA_T0 = 0.150000
LIDAR_SIGMA_R0 = 0.012857142857142857
LIDAR_LAMBDA = 1.0


# ============================================================
# Experiment definitions
#
# These experiments use the SAME physical LiDAR measurement set.
#
# E0: all fixed
# E1: GPS predictive only
# E2: IMU predictive only
# E3: Camera predictive only
# E4: GPS + IMU predictive
# E5: GPS + Camera predictive
# E6: IMU + Camera predictive
# E7: GPS + IMU + Camera predictive, LiDAR fixed
# E8: FINAL METHOD:
#     GPS + IMU + Camera predictive
#     + LiDAR predictive dynamic covariance lambda=1
# ============================================================

SENSORWISE_EXPERIMENTS = [
    (
        "E0_all_fixed",
        {
            "gps": "fixed",
            "imu": "fixed",
            "lidar": "fixed",
            "camera": "fixed",
        },
    ),
    (
        "E1_gps_predictive",
        {
            "gps": "predictive",
            "imu": "fixed",
            "lidar": "fixed",
            "camera": "fixed",
        },
    ),
    (
        "E2_imu_predictive",
        {
            "gps": "fixed",
            "imu": "predictive",
            "lidar": "fixed",
            "camera": "fixed",
        },
    ),
    (
        "E3_camera_predictive",
        {
            "gps": "fixed",
            "imu": "fixed",
            "lidar": "fixed",
            "camera": "predictive",
        },
    ),
    (
        "E4_gps_imu_predictive",
        {
            "gps": "predictive",
            "imu": "predictive",
            "lidar": "fixed",
            "camera": "fixed",
        },
    ),
    (
        "E5_gps_camera_predictive",
        {
            "gps": "predictive",
            "imu": "fixed",
            "lidar": "fixed",
            "camera": "predictive",
        },
    ),
    (
        "E6_imu_camera_predictive",
        {
            "gps": "fixed",
            "imu": "predictive",
            "lidar": "fixed",
            "camera": "predictive",
        },
    ),
    (
        "E7_gps_imu_camera_predictive_lidar_fixed",
        {
            "gps": "predictive",
            "imu": "predictive",
            "lidar": "fixed",
            "camera": "predictive",
        },
    ),
]


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def build_sources(mode_map):
    return {
        sensor: ReliabilitySource(mode)
        for sensor, mode in mode_map.items()
    }


def build_fixed_config():
    """
    Use the validated best fixed LiDAR covariance.

    Other sensor fixed/predictive covariance settings remain
    exactly those already defined by FourSensorFactorConfig.
    """
    return replace(
        FourSensorFactorConfig(),
        lidar_translation_sigma=
            float(LIDAR_SIGMA_T0),
        lidar_rotation_sigma=
            float(LIDAR_SIGMA_R0),
    )


def load_lidar_risk(
    expected_pairs: int,
) -> np.ndarray:

    path = Path(
        MAPPING_PATH
    )

    if not path.exists():
        raise FileNotFoundError(
            "LiDAR dynamic mapping not found:\n"
            f"{path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    if "risk_score_pair" not in data.files:
        raise KeyError(
            "risk_score_pair not found in mapping file.\n"
            f"Available arrays: {data.files}"
        )

    risk = np.asarray(
        data["risk_score_pair"],
        dtype=np.float64,
    ).reshape(-1)

    if len(risk) != expected_pairs:
        raise ValueError(
            f"risk_score_pair length={len(risk)}, "
            f"expected={expected_pairs}"
        )

    if not np.all(
        np.isfinite(risk)
    ):
        raise ValueError(
            "risk_score_pair contains NaN/Inf"
        )

    return np.clip(
        risk,
        0.0,
        1.0,
    )


def build_lidar_dynamic_covariance(
    risk: np.ndarray,
):
    scale = (
        1.0
        +
        LIDAR_LAMBDA
        *
        np.square(risk)
    )

    sigma_t = (
        LIDAR_SIGMA_T0
        *
        scale
    )

    sigma_r = (
        LIDAR_SIGMA_R0
        *
        scale
    )

    return (
        sigma_t,
        sigma_r,
        scale,
    )


def load_cached_metrics(
    case_dir: Path,
):
    trajectory_path = (
        case_dir
        /
        "trajectory.txt"
    )

    if not trajectory_path.exists():
        return None

    trajectory = np.loadtxt(
        trajectory_path,
        dtype=np.float64,
    )

    return evaluate(
        trajectory
    )


def save_case_metrics(
    case_dir: Path,
    metrics: dict,
):
    ensure_dir(
        case_dir
    )

    with (
        case_dir
        /
        "metrics.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        for key, value in metrics.items():
            file.write(
                f"{key}: {float(value):.10f}\n"
            )


def run_sensorwise_case(
    measurements,
    name: str,
    mode_map: dict,
):
    case_dir = (
        Path(OUT_ROOT)
        /
        name
    )

    cached = load_cached_metrics(
        case_dir
    )

    if cached is not None:
        print()
        print(
            f"[CACHE] {name}"
        )
        print(
            "Existing trajectory.txt found, "
            "skip optimization."
        )

        return cached

    print()
    print(
        "=" * 120
    )

    print(
        name
    )

    print(
        "=" * 120
    )

    print(
        "Modes:",
        mode_map,
    )

    trajectory, poses, counts = (
        run_sensorwise_reliability_graph(
            measurements,
            str(case_dir),
            build_sources(
                mode_map
            ),
            predictive_dir=
                PREDICTIVE_DIR,
            oracle_path=None,
            config=
                build_fixed_config(),
        )
    )

    metrics = evaluate(
        trajectory
    )

    save_case_metrics(
        case_dir,
        metrics,
    )

    return metrics


def run_final_dynamic_case(
    measurements,
    sigma_t,
    sigma_r,
):
    name = (
        "E8_final_full_predictive_dynamic"
    )

    case_dir = (
        Path(OUT_ROOT)
        /
        name
    )

    cached = load_cached_metrics(
        case_dir
    )

    if cached is not None:
        print()
        print(
            f"[CACHE] {name}"
        )
        print(
            "Existing trajectory.txt found, "
            "skip optimization."
        )

        return cached

    print()
    print(
        "=" * 120
    )

    print(
        "E8 FINAL METHOD"
    )

    print(
        "=" * 120
    )

    print(
        "GPS    : predictive"
    )

    print(
        "IMU    : predictive"
    )

    print(
        "Camera : predictive"
    )

    print(
        "LiDAR  : predictive dynamic direct sigma"
    )

    print(
        f"LiDAR lambda: "
        f"{LIDAR_LAMBDA:.6f}"
    )

    print(
        "LiDAR translation sigma min/max/mean:"
    )

    print(
        f"  {sigma_t.min():.6f} / "
        f"{sigma_t.max():.6f} / "
        f"{sigma_t.mean():.6f}"
    )

    print(
        "LiDAR rotation sigma min/max/mean:"
    )

    print(
        f"  {sigma_r.min():.6f} / "
        f"{sigma_r.max():.6f} / "
        f"{sigma_r.mean():.6f}"
    )

    trajectory, poses, counts = (
        run_dynamic_graph(
            measurements,
            sigma_t,
            sigma_r,
            str(case_dir),
        )
    )

    metrics = evaluate(
        trajectory
    )

    save_case_metrics(
        case_dir,
        metrics,
    )

    return metrics


def make_row(
    name: str,
    description: str,
    metrics: dict,
):
    return {
        "experiment": name,
        "description": description,
        "ATE3D": float(
            metrics["ATE3D"]
        ),
        "ATE2D": float(
            metrics["ATE2D"]
        ),
        "Mean3D": float(
            metrics["Mean3D"]
        ),
        "Max3D": float(
            metrics["Max3D"]
        ),
        "ZRMSE": float(
            metrics["ZRMSE"]
        ),
    }


def save_csv(
    rows,
):
    fields = [
        "experiment",
        "description",
        "ATE3D",
        "ATE2D",
        "Mean3D",
        "Max3D",
        "ZRMSE",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def improvement_percent(
    baseline,
    value,
):
    return (
        (
            baseline
            -
            value
        )
        /
        baseline
        *
        100.0
    )


# ============================================================
# Main
# ============================================================

def main():

    ensure_dir(
        OUT_ROOT
    )

    print(
        "=" * 120
    )

    print(
        "FINAL FOUR-SENSOR PREDICTIVE RELIABILITY "
        "DYNAMIC FACTOR GRAPH EXPERIMENT"
    )

    print(
        "=" * 120
    )

    print()
    print(
        "Purpose:"
    )

    print(
        "1. Run fixed/predictive sensor ablations."
    )

    print(
        "2. Run the final four-sensor method."
    )

    print(
        "3. Compare all experiments using the same "
        "physical LiDAR measurement set and evaluator."
    )

    # ========================================================
    # Load measurements once
    # ========================================================

    measurements = (
        load_physical_measurements()
    )

    n = len(
        measurements.gps_local
    )

    pairs = n - 1

    print()
    print(
        f"Frames      : {n}"
    )

    print(
        f"LiDAR pairs : {pairs}"
    )

    # ========================================================
    # Build final dynamic LiDAR covariance
    # ========================================================

    risk = load_lidar_risk(
        pairs
    )

    (
        sigma_t,
        sigma_r,
        scale,
    ) = build_lidar_dynamic_covariance(
        risk
    )

    print()
    print(
        "=" * 120
    )

    print(
        "FINAL LIDAR DYNAMIC COVARIANCE"
    )

    print(
        "=" * 120
    )

    print(
        f"Risk min/max/mean: "
        f"{risk.min():.6f} / "
        f"{risk.max():.6f} / "
        f"{risk.mean():.6f}"
    )

    print(
        f"Scale min/max/mean: "
        f"{scale.min():.6f} / "
        f"{scale.max():.6f} / "
        f"{scale.mean():.6f}"
    )

    print(
        f"Translation sigma min/max/mean: "
        f"{sigma_t.min():.6f} / "
        f"{sigma_t.max():.6f} / "
        f"{sigma_t.mean():.6f}"
    )

    print(
        f"Rotation sigma min/max/mean: "
        f"{sigma_r.min():.6f} / "
        f"{sigma_r.max():.6f} / "
        f"{sigma_r.mean():.6f}"
    )

    # ========================================================
    # Run sensorwise ablations
    # ========================================================

    rows = []

    descriptions = {
        "E0_all_fixed":
            "GPS fixed + IMU fixed + LiDAR fixed + Camera fixed",

        "E1_gps_predictive":
            "GPS predictive only; IMU/LiDAR/Camera fixed",

        "E2_imu_predictive":
            "IMU predictive only; GPS/LiDAR/Camera fixed",

        "E3_camera_predictive":
            "Camera predictive only; GPS/IMU/LiDAR fixed",

        "E4_gps_imu_predictive":
            "GPS+IMU predictive; LiDAR+Camera fixed",

        "E5_gps_camera_predictive":
            "GPS+Camera predictive; IMU+LiDAR fixed",

        "E6_imu_camera_predictive":
            "IMU+Camera predictive; GPS+LiDAR fixed",

        "E7_gps_imu_camera_predictive_lidar_fixed":
            "GPS+IMU+Camera predictive; LiDAR fixed",

        "E8_final_full_predictive_dynamic":
            "GPS+IMU+Camera predictive + LiDAR predictive dynamic covariance",
    }

    for (
        name,
        mode_map,
    ) in SENSORWISE_EXPERIMENTS:

        metrics = run_sensorwise_case(
            measurements,
            name,
            mode_map,
        )

        rows.append(
            make_row(
                name,
                descriptions[name],
                metrics,
            )
        )

        save_csv(
            rows
        )

    # ========================================================
    # Final method
    # ========================================================

    final_metrics = (
        run_final_dynamic_case(
            measurements,
            sigma_t,
            sigma_r,
        )
    )

    rows.append(
        make_row(
            "E8_final_full_predictive_dynamic",
            descriptions[
                "E8_final_full_predictive_dynamic"
            ],
            final_metrics,
        )
    )

    save_csv(
        rows
    )

    # ========================================================
    # Final comparison
    # ========================================================

    best = min(
        rows,
        key=lambda row:
            row["ATE3D"],
    )

    all_fixed = next(
        row
        for row in rows
        if row["experiment"]
        ==
        "E0_all_fixed"
    )

    lidar_fixed_predictive_gic = next(
        row
        for row in rows
        if row["experiment"]
        ==
        "E7_gps_imu_camera_predictive_lidar_fixed"
    )

    final_row = next(
        row
        for row in rows
        if row["experiment"]
        ==
        "E8_final_full_predictive_dynamic"
    )

    improvement_vs_all_fixed = (
        improvement_percent(
            all_fixed["ATE3D"],
            final_row["ATE3D"],
        )
    )

    improvement_vs_lidar_fixed = (
        improvement_percent(
            lidar_fixed_predictive_gic[
                "ATE3D"
            ],
            final_row["ATE3D"],
        )
    )

    print()
    print()
    print(
        "=" * 120
    )

    print(
        "FINAL MULTISENSOR COMPARISON"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Experiment':48s} "
        f"{'ATE3D':>10s} "
        f"{'ATE2D':>10s} "
        f"{'Mean3D':>10s} "
        f"{'Max3D':>10s} "
        f"{'ZRMSE':>10s}"
    )

    print(
        "-" * 120
    )

    for row in rows:

        marker = ""

        if row[
            "experiment"
        ] == best[
            "experiment"
        ]:
            marker = "  <-- BEST"

        print(
            f"{row['experiment']:48s} "
            f"{row['ATE3D']:10.6f} "
            f"{row['ATE2D']:10.6f} "
            f"{row['Mean3D']:10.6f} "
            f"{row['Max3D']:10.6f} "
            f"{row['ZRMSE']:10.6f}"
            f"{marker}"
        )

    print()
    print(
        "=" * 120
    )

    print(
        "FINAL METHOD SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        f"All-fixed ATE3D:"
        f"                     "
        f"{all_fixed['ATE3D']:.6f} m"
    )

    print(
        f"G+I+C predictive, LiDAR fixed:"
        f"       "
        f"{lidar_fixed_predictive_gic['ATE3D']:.6f} m"
    )

    print(
        f"FINAL full predictive dynamic:"
        f"        "
        f"{final_row['ATE3D']:.6f} m"
    )

    print()
    print(
        f"Improvement vs all-fixed:"
        f"              "
        f"{improvement_vs_all_fixed:+.2f}%"
    )

    print(
        f"Improvement from dynamic LiDAR "
        f"over E7:"
        f"  "
        f"{improvement_vs_lidar_fixed:+.2f}%"
    )

    # ========================================================
    # Save summary
    # ========================================================

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "FINAL FOUR-SENSOR PREDICTIVE RELIABILITY "
            "DYNAMIC FACTOR GRAPH\n"
        )

        file.write(
            "=" * 100
            +
            "\n\n"
        )

        file.write(
            f"Frames: {n}\n"
        )

        file.write(
            f"LiDAR pairs: {pairs}\n"
        )

        file.write(
            f"LiDAR lambda: "
            f"{LIDAR_LAMBDA:.6f}\n"
        )

        file.write(
            f"LiDAR sigma_t min/max/mean: "
            f"{sigma_t.min():.10f} / "
            f"{sigma_t.max():.10f} / "
            f"{sigma_t.mean():.10f}\n"
        )

        file.write(
            f"LiDAR sigma_r min/max/mean: "
            f"{sigma_r.min():.10f} / "
            f"{sigma_r.max():.10f} / "
            f"{sigma_r.mean():.10f}\n\n"
        )

        file.write(
            "EXPERIMENTS\n"
        )

        file.write(
            "-" * 100
            +
            "\n"
        )

        for row in rows:
            file.write(
                f"{row['experiment']}\n"
            )

            file.write(
                f"  {row['description']}\n"
            )

            file.write(
                f"  ATE3D={row['ATE3D']:.10f}\n"
            )

            file.write(
                f"  ATE2D={row['ATE2D']:.10f}\n"
            )

            file.write(
                f"  Mean3D={row['Mean3D']:.10f}\n"
            )

            file.write(
                f"  Max3D={row['Max3D']:.10f}\n"
            )

            file.write(
                f"  ZRMSE={row['ZRMSE']:.10f}\n\n"
            )

        file.write(
            "FINAL METHOD\n"
        )

        file.write(
            "-" * 100
            +
            "\n"
        )

        file.write(
            f"Best experiment: "
            f"{best['experiment']}\n"
        )

        file.write(
            f"Final ATE3D: "
            f"{final_row['ATE3D']:.10f}\n"
        )

        file.write(
            f"Improvement vs all-fixed: "
            f"{improvement_vs_all_fixed:.6f}%\n"
        )

        file.write(
            f"Improvement vs E7 "
            f"(G+I+C predictive, LiDAR fixed): "
            f"{improvement_vs_lidar_fixed:.6f}%\n"
        )

    print()
    print(
        "Saved:"
    )

    print(
        CSV_PATH
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()
