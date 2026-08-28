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

BEST_FIXED_SIGMA_PATH = os.path.join(
    CALIB_DIR,
    "best_fixed_sigma.txt",
)

PRED = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

OUT_ROOT = os.path.join(
    CALIB_DIR,
    "mapping_strength_sweep",
)

CSV_PATH = os.path.join(
    OUT_ROOT,
    "mapping_strength_sweep.csv",
)

BEST_LAMBDA_PATH = os.path.join(
    OUT_ROOT,
    "best_lambda.txt",
)

BEST_RESULT_PATH = os.path.join(
    OUT_ROOT,
    "best_mapping_result.txt",
)


# ============================================================
# Reference results
# ============================================================

BEST_FIXED_ATE_REFERENCE = 1.977235

GT_ORACLE_ATE_REFERENCE = 1.890239


# ============================================================
# Sweep candidates
#
# lambda = 0:
#   exact optimized fixed baseline
#
# lambda > 0:
#   genuinely dynamic covariance
# ============================================================

LAMBDA_CANDIDATES = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
]


# ============================================================
# Rotation / translation covariance ratio
#
# original:
# translation = 0.35
# rotation    = 0.03
# ============================================================

ROT_TRANS_RATIO = 0.03 / 0.35


# ============================================================
# Utility
# ============================================================

def lambda_tag(
    value: float,
) -> str:

    return (
        f"{value:.2f}"
        .replace(".", "p")
    )


# ============================================================
# Load best fixed translation sigma
# ============================================================

def load_best_fixed_sigma() -> float:

    path = Path(
        BEST_FIXED_SIGMA_PATH
    )

    if not path.exists():

        raise FileNotFoundError(
            "Best fixed sigma file not found:\n"
            f"{path}"
        )

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    sigma = float(
        text
    )

    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
    ):

        raise ValueError(
            f"Invalid fixed sigma: {sigma}"
        )

    return sigma


# ============================================================
# Load risk score
# ============================================================

def load_risk_score(
    expected_pairs: int,
) -> np.ndarray:

    path = Path(
        MAPPING_PATH
    )

    if not path.exists():

        raise FileNotFoundError(
            "Mapping file not found:\n"
            f"{path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    if "risk_score_pair" not in data.files:

        raise KeyError(
            "risk_score_pair not found.\n"
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
            "risk_score_pair contains NaN/Inf."
        )

    return np.clip(
        risk,
        0.0,
        1.0,
    )


# ============================================================
# Dynamic mapping
#
# sigma_t = sigma_t0 * (1 + lambda * risk^2)
# sigma_r = sigma_r0 * (1 + lambda * risk^2)
# ============================================================

def build_dynamic_covariance(
    risk_score: np.ndarray,
    sigma_t0: float,
    lambda_value: float,
):

    sigma_r0 = (
        float(sigma_t0)
        *
        ROT_TRANS_RATIO
    )

    scale = (
        1.0
        +
        float(lambda_value)
        *
        np.square(
            risk_score
        )
    )

    sigma_t = (
        float(sigma_t0)
        *
        scale
    )

    sigma_r = (
        float(sigma_r0)
        *
        scale
    )

    return (
        sigma_t,
        sigma_r,
        scale,
    )


# ============================================================
# Sources for exact fixed baseline
# ============================================================

def build_fixed_sources():

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
# Run lambda = 0 exact fixed baseline
# ============================================================

def run_lambda_zero_fixed(
    measurements,
    sigma_t0: float,
    output_dir: str,
):

    sigma_r0 = (
        float(sigma_t0)
        *
        ROT_TRANS_RATIO
    )

    # Frozen dataclass -> use replace()
    config = replace(
        FourSensorFactorConfig(),
        lidar_translation_sigma=
            float(
                sigma_t0
            ),
        lidar_rotation_sigma=
            float(
                sigma_r0
            ),
    )

    print()
    print(
        "LAMBDA = 0 SPECIAL CASE"
    )

    print(
        "Using existing FIXED LiDAR graph "
        "for exact baseline reproduction."
    )

    print(
        f"Fixed translation sigma = "
        f"{sigma_t0:.6f}"
    )

    print(
        f"Fixed rotation sigma    = "
        f"{sigma_r0:.6f}"
    )

    trajectory, poses, counts = (
        run_sensorwise_reliability_graph(
            measurements,
            output_dir,
            build_fixed_sources(),
            predictive_dir=PRED,
            oracle_path=None,
            config=config,
        )
    )

    return (
        trajectory,
        poses,
        counts,
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    rows,
):

    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "lambda",
        "sigma_t_min",
        "sigma_t_max",
        "sigma_t_mean",
        "sigma_r_min",
        "sigma_r_max",
        "sigma_r_mean",
        "ATE3D",
        "ATE2D",
        "Mean3D",
        "Max3D",
        "ZRMSE",
        "improvement_vs_best_fixed_percent",
        "gap_vs_gt_oracle_percent",
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


# ============================================================
# Cached results
# ============================================================

def try_load_cached(
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

    metrics = evaluate(
        trajectory
    )

    return (
        trajectory,
        metrics,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "LIDAR PREDICTIVE COVARIANCE "
        "MAPPING STRENGTH SWEEP"
    )

    print(
        "=" * 120
    )

    print()
    print(
        "Formula:"
    )

    print(
        "sigma_t = sigma_t0 * "
        "(1 + lambda * risk_score^2)"
    )

    print(
        "sigma_r = sigma_r0 * "
        "(1 + lambda * risk_score^2)"
    )

    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Measurements
    # ========================================================

    measurements = (
        load_physical_measurements()
    )

    n = len(
        measurements.gps_local
    )

    pairs = (
        n - 1
    )

    print()
    print(
        "Frames:",
        n,
    )

    print(
        "LiDAR pairs:",
        pairs,
    )

    # ========================================================
    # Fixed base sigma
    # ========================================================

    sigma_t0 = (
        load_best_fixed_sigma()
    )

    sigma_r0 = (
        sigma_t0
        *
        ROT_TRANS_RATIO
    )

    print()
    print(
        "=" * 120
    )

    print(
        "BASE FIXED LIDAR COVARIANCE"
    )

    print(
        "=" * 120
    )

    print(
        f"translation sigma: "
        f"{sigma_t0:.6f}"
    )

    print(
        f"rotation sigma   : "
        f"{sigma_r0:.6f}"
    )

    # ========================================================
    # Risk
    # ========================================================

    risk = load_risk_score(
        pairs
    )

    print()
    print(
        "=" * 120
    )

    print(
        "RISK SCORE"
    )

    print(
        "=" * 120
    )

    print(
        "min/max/mean:"
    )

    print(
        f"{risk.min():.6f} / "
        f"{risk.max():.6f} / "
        f"{risk.mean():.6f}"
    )

    print(
        "score >= 0.5:",
        int(
            np.sum(
                risk >= 0.5
            )
        ),
    )

    print(
        "score >= 0.9:",
        int(
            np.sum(
                risk >= 0.9
            )
        ),
    )

    # ========================================================
    # Sweep
    # ========================================================

    rows = []

    total = len(
        LAMBDA_CANDIDATES
    )

    for index, lambda_value in enumerate(
        LAMBDA_CANDIDATES,
        start=1,
    ):

        (
            sigma_t,
            sigma_r,
            scale,
        ) = build_dynamic_covariance(
            risk,
            sigma_t0,
            lambda_value,
        )

        print()
        print()
        print(
            "#" * 120
        )

        print(
            f"[{index}/{total}] "
            f"LAMBDA = {lambda_value:.2f}"
        )

        print(
            "#" * 120
        )

        print(
            "Translation sigma min/max/mean:"
        )

        print(
            f"  {sigma_t.min():.6f} / "
            f"{sigma_t.max():.6f} / "
            f"{sigma_t.mean():.6f}"
        )

        print(
            "Rotation sigma min/max/mean:"
        )

        print(
            f"  {sigma_r.min():.6f} / "
            f"{sigma_r.max():.6f} / "
            f"{sigma_r.mean():.6f}"
        )

        print(
            "Scale min/max/mean:"
        )

        print(
            f"  {scale.min():.6f} / "
            f"{scale.max():.6f} / "
            f"{scale.mean():.6f}"
        )

        case_dir = (
            Path(
                OUT_ROOT
            )
            /
            (
                "lambda_"
                +
                lambda_tag(
                    lambda_value
                )
            )
        )

        cached = (
            try_load_cached(
                case_dir
            )
        )

        if cached is not None:

            print()
            print(
                "Cached trajectory found."
            )

            print(
                "Skip optimization."
            )

            _, metrics = cached

        else:

            # =================================================
            # λ=0 -> exact fixed graph
            # =================================================

            if np.isclose(
                lambda_value,
                0.0,
            ):

                trajectory, _, _ = (
                    run_lambda_zero_fixed(
                        measurements,
                        sigma_t0,
                        str(case_dir),
                    )
                )

            # =================================================
            # λ>0 -> genuinely dynamic graph
            # =================================================

            else:

                trajectory, _, _ = (
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

        # ====================================================
        # Metrics
        # ====================================================

        ate = float(
            metrics["ATE3D"]
        )

        improvement = (
            (
                BEST_FIXED_ATE_REFERENCE
                -
                ate
            )
            /
            BEST_FIXED_ATE_REFERENCE
            *
            100.0
        )

        oracle_gap = (
            (
                ate
                -
                GT_ORACLE_ATE_REFERENCE
            )
            /
            GT_ORACLE_ATE_REFERENCE
            *
            100.0
        )

        row = {

            "lambda":
                float(
                    lambda_value
                ),

            "sigma_t_min":
                float(
                    sigma_t.min()
                ),

            "sigma_t_max":
                float(
                    sigma_t.max()
                ),

            "sigma_t_mean":
                float(
                    sigma_t.mean()
                ),

            "sigma_r_min":
                float(
                    sigma_r.min()
                ),

            "sigma_r_max":
                float(
                    sigma_r.max()
                ),

            "sigma_r_mean":
                float(
                    sigma_r.mean()
                ),

            "ATE3D":
                float(
                    metrics["ATE3D"]
                ),

            "ATE2D":
                float(
                    metrics["ATE2D"]
                ),

            "Mean3D":
                float(
                    metrics["Mean3D"]
                ),

            "Max3D":
                float(
                    metrics["Max3D"]
                ),

            "ZRMSE":
                float(
                    metrics["ZRMSE"]
                ),

            "improvement_vs_best_fixed_percent":
                float(
                    improvement
                ),

            "gap_vs_gt_oracle_percent":
                float(
                    oracle_gap
                ),
        }

        rows.append(
            row
        )

        save_csv(
            rows
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            case_dir
            /
            "mapping_strength_result.txt"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            for key, value in row.items():

                file.write(
                    f"{key}: {value}\n"
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
            f"ATE3D            : "
            f"{row['ATE3D']:.6f}"
        )

        print(
            f"ATE2D            : "
            f"{row['ATE2D']:.6f}"
        )

        print(
            f"Mean3D           : "
            f"{row['Mean3D']:.6f}"
        )

        print(
            f"Max3D            : "
            f"{row['Max3D']:.6f}"
        )

        print(
            f"ZRMSE            : "
            f"{row['ZRMSE']:.6f}"
        )

        print(
            f"vs Best Fixed    : "
            f"{improvement:+.2f}%"
        )

        print(
            f"Gap vs GT Oracle : "
            f"{oracle_gap:+.2f}%"
        )

    # ========================================================
    # Best result
    # ========================================================

    best = min(
        rows,
        key=lambda item:
            item["ATE3D"],
    )

    with open(
        BEST_LAMBDA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"{best['lambda']:.10f}\n"
        )

    with open(
        BEST_RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "BEST LIDAR MAPPING STRENGTH\n"
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
    # Final table
    # ========================================================

    print()
    print()
    print(
        "=" * 120
    )

    print(
        "LIDAR MAPPING STRENGTH SWEEP RESULTS"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Lambda':>8s} "
        f"{'SigmaMean':>12s} "
        f"{'ATE3D':>12s} "
        f"{'ATE2D':>12s} "
        f"{'ZRMSE':>12s} "
        f"{'vsFixed':>12s} "
        f"{'GapOracle':>12s}"
    )

    print(
        "-" * 120
    )

    for row in rows:

        marker = ""

        if np.isclose(
            row["lambda"],
            best["lambda"],
        ):

            marker = "  <-- BEST"

        print(
            f"{row['lambda']:8.2f} "
            f"{row['sigma_t_mean']:12.6f} "
            f"{row['ATE3D']:12.6f} "
            f"{row['ATE2D']:12.6f} "
            f"{row['ZRMSE']:12.6f} "
            f"{row['improvement_vs_best_fixed_percent']:11.2f}% "
            f"{row['gap_vs_gt_oracle_percent']:11.2f}%"
            f"{marker}"
        )

    # ========================================================
    # Lambda 0 sanity check
    # ========================================================

    zero_row = next(
        row
        for row in rows
        if np.isclose(
            row["lambda"],
            0.0,
        )
    )

    difference = abs(
        zero_row["ATE3D"]
        -
        BEST_FIXED_ATE_REFERENCE
    )

    print()
    print(
        "=" * 120
    )

    print(
        "LAMBDA=0 SANITY CHECK"
    )

    print(
        "=" * 120
    )

    print(
        f"lambda=0 ATE3D       : "
        f"{zero_row['ATE3D']:.6f}"
    )

    print(
        f"Previous Best Fixed  : "
        f"{BEST_FIXED_ATE_REFERENCE:.6f}"
    )

    print(
        f"Absolute difference  : "
        f"{difference:.8f}"
    )

    if difference < 1e-4:

        print(
            "PASS:"
        )

        print(
            "lambda=0 exactly reproduces "
            "the optimized fixed baseline."
        )

    else:

        print(
            "WARNING:"
        )

        print(
            "lambda=0 does not reproduce "
            "the previous fixed baseline."
        )

    # ========================================================
    # Best
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "BEST MAPPING STRENGTH"
    )

    print(
        "=" * 120
    )

    print(
        f"Best lambda       : "
        f"{best['lambda']:.6f}"
    )

    print(
        f"Mean LiDAR sigma  : "
        f"{best['sigma_t_mean']:.6f}"
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

    print(
        f"Improvement fixed : "
        f"{best['improvement_vs_best_fixed_percent']:+.2f}%"
    )

    print(
        f"Gap GT Oracle     : "
        f"{best['gap_vs_gt_oracle_percent']:+.2f}%"
    )

    # ========================================================
    # Conclusion
    # ========================================================

    print()
    print(
        "=" * 120
    )

    print(
        "FINAL CONCLUSION"
    )

    print(
        "=" * 120
    )

    if (
        best["lambda"] > 0.0
        and
        best["ATE3D"]
        <
        BEST_FIXED_ATE_REFERENCE
    ):

        print(
            "SUCCESS:"
        )

        print(
            "A genuinely dynamic predictive LiDAR "
            "covariance beats the optimized fixed baseline."
        )

    elif np.isclose(
        best["lambda"],
        0.0,
    ):

        print(
            "RESULT:"
        )

        print(
            "The optimized fixed LiDAR covariance "
            "is still the best solution."
        )

    else:

        print(
            "RESULT:"
        )

        print(
            "A dynamic mapping was selected, "
            "but it does not beat the fixed baseline."
        )

    print()
    print(
        "Saved:"
    )

    print(
        CSV_PATH
    )

    print(
        BEST_LAMBDA_PATH
    )

    print(
        BEST_RESULT_PATH
    )


if __name__ == "__main__":
    main()
