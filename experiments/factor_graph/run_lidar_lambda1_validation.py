from __future__ import annotations

import csv
import os
import sys
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
# Reuse the already validated direct-sigma graph runner.
# No graph construction is duplicated here.
# ============================================================

from experiments.factor_graph.run_lidar_calibrated_dynamic_fg import (
    load_physical_measurements,
    run_dynamic_graph,
    evaluate,
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

PHYSICAL_FACTOR_PATH = os.path.join(
    ROOT,
    "results",
    "lidar_physical_oracle_benchmark",
    "physical_lidar_factor_data.npz",
)

OUT_ROOT = os.path.join(
    CALIB_DIR,
    "lambda1_validation",
)

REPORT_PATH = os.path.join(
    OUT_ROOT,
    "lambda1_validation_report.txt",
)

CSV_PATH = os.path.join(
    OUT_ROOT,
    "validation_runs.csv",
)

RISK_ANALYSIS_PATH = os.path.join(
    OUT_ROOT,
    "risk_error_analysis.csv",
)

MATCHED_ORACLE_PATH = os.path.join(
    OUT_ROOT,
    "matched_gt_oracle_mapping.npz",
)


# ============================================================
# Experiment constants
# ============================================================

SIGMA_T0 = 0.150000
SIGMA_R0 = 0.012857142857142857

LAMBDA_MAIN = 1.0
LAMBDA_ANOMALY = 0.5

REPEAT_RUNS = 3

REFERENCE_FIXED_ATE = 1.977235
REFERENCE_LAMBDA1_ATE = 1.728809

# Quantile bins for risk/error diagnosis
RISK_BINS = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0000001,
]


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def build_sigma_from_risk(
    risk,
    lambda_value,
):
    risk = np.asarray(
        risk,
        dtype=np.float64,
    ).reshape(-1)

    risk = np.clip(
        risk,
        0.0,
        1.0,
    )

    scale = (
        1.0
        +
        float(lambda_value)
        *
        np.square(risk)
    )

    sigma_t = (
        SIGMA_T0
        *
        scale
    )

    sigma_r = (
        SIGMA_R0
        *
        scale
    )

    return (
        sigma_t,
        sigma_r,
        scale,
    )


def pearson_corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    mask = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return float("nan")

    if (
        np.std(a) <= 1e-15
        or
        np.std(b) <= 1e-15
    ):
        return float("nan")

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def rankdata_average(x):
    """
    Small scipy-free average-rank implementation.
    Used only for Spearman correlation.
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )

    sorted_x = x[order]

    i = 0

    while i < len(x):
        j = i + 1

        while (
            j < len(x)
            and
            sorted_x[j] == sorted_x[i]
        ):
            j += 1

        # ranks are 1-based for Spearman convention
        avg_rank = (
            (i + 1)
            +
            j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    return ranks


def spearman_corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    mask = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return float("nan")

    return pearson_corr(
        rankdata_average(a),
        rankdata_average(b),
    )


def load_prediction_risk(
    expected_pairs,
):
    if not Path(MAPPING_PATH).exists():
        raise FileNotFoundError(
            MAPPING_PATH
        )

    data = np.load(
        MAPPING_PATH,
        allow_pickle=False,
    )

    if "risk_score_pair" not in data.files:
        raise KeyError(
            "risk_score_pair not found in "
            f"{MAPPING_PATH}\n"
            f"Available: {data.files}"
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

    if not np.all(np.isfinite(risk)):
        raise ValueError(
            "Prediction risk contains NaN/Inf."
        )

    return np.clip(
        risk,
        0.0,
        1.0,
    )


def load_gt_factor_errors(
    expected_pairs,
):
    if not Path(PHYSICAL_FACTOR_PATH).exists():
        raise FileNotFoundError(
            PHYSICAL_FACTOR_PATH
        )

    data = np.load(
        PHYSICAL_FACTOR_PATH,
        allow_pickle=False,
    )

    required = [
        "te",
        "re",
    ]

    for key in required:
        if key not in data.files:
            raise KeyError(
                f"Missing '{key}' in "
                f"{PHYSICAL_FACTOR_PATH}\n"
                f"Available: {data.files}"
            )

    te = np.asarray(
        data["te"],
        dtype=np.float64,
    ).reshape(-1)

    re = np.asarray(
        data["re"],
        dtype=np.float64,
    ).reshape(-1)

    if len(te) != expected_pairs:
        raise ValueError(
            f"te length={len(te)}, "
            f"expected={expected_pairs}"
        )

    if len(re) != expected_pairs:
        raise ValueError(
            f"re length={len(re)}, "
            f"expected={expected_pairs}"
        )

    if not np.all(np.isfinite(te)):
        raise ValueError(
            "GT translation error contains NaN/Inf."
        )

    if not np.all(np.isfinite(re)):
        raise ValueError(
            "GT rotation error contains NaN/Inf."
        )

    return te, re


def normalize_error_to_risk(
    values,
):
    """
    Oracle-only calibration.

    Converts realized GT factor error to [0, 1] using its
    own p50 -> p90 interval:

        error <= p50  -> risk 0
        error >= p90  -> risk 1

    This intentionally uses GT realized error and is therefore
    an ORACLE diagnostic, not a deployable predictor.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    p50 = float(
        np.quantile(
            values,
            0.50,
        )
    )

    p90 = float(
        np.quantile(
            values,
            0.90,
        )
    )

    denom = max(
        p90 - p50,
        1e-12,
    )

    score = (
        values
        -
        p50
    ) / denom

    score = np.clip(
        score,
        0.0,
        1.0,
    )

    return (
        score,
        p50,
        p90,
    )


def build_matched_gt_oracle_risk(
    te,
    re,
):
    t_score, t_p50, t_p90 = (
        normalize_error_to_risk(
            te
        )
    )

    r_score, r_p50, r_p90 = (
        normalize_error_to_risk(
            re
        )
    )

    # Same joint-risk rule as the predictive calibration:
    # if either translation or rotation is risky,
    # the LiDAR factor is treated as risky.
    risk = np.maximum(
        t_score,
        r_score,
    )

    return (
        risk,
        {
            "translation_p50": t_p50,
            "translation_p90": t_p90,
            "rotation_p50": r_p50,
            "rotation_p90": r_p90,
        },
    )


def trajectory_difference(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    n = min(
        len(a),
        len(b),
    )

    delta = (
        a[:n]
        -
        b[:n]
    )

    norm = np.linalg.norm(
        delta,
        axis=1,
    )

    return {
        "max_position_difference":
            float(
                np.max(norm)
            ),
        "mean_position_difference":
            float(
                np.mean(norm)
            ),
        "rmse_position_difference":
            float(
                np.sqrt(
                    np.mean(
                        norm ** 2
                    )
                )
            ),
    }


def run_forced_case(
    measurements,
    risk,
    lambda_value,
    output_dir,
):
    """
    FORCE optimization.

    This function deliberately does not inspect any existing
    trajectory.txt. run_dynamic_graph() is called every time.
    Existing files in output_dir may be overwritten by the new
    optimization result.
    """
    sigma_t, sigma_r, scale = (
        build_sigma_from_risk(
            risk,
            lambda_value,
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        f"FORCED RUN: lambda={lambda_value:.6f}"
    )

    print(
        "=" * 120
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

    trajectory, poses, counts = (
        run_dynamic_graph(
            measurements,
            sigma_t,
            sigma_r,
            output_dir,
        )
    )

    metrics = evaluate(
        trajectory
    )

    return {
        "trajectory": trajectory,
        "poses": poses,
        "counts": counts,
        "metrics": metrics,
        "sigma_t": sigma_t,
        "sigma_r": sigma_r,
        "scale": scale,
    }


def risk_bin_analysis(
    risk,
    te,
    re,
):
    rows = []

    for i in range(
        len(RISK_BINS) - 1
    ):
        lo = RISK_BINS[i]
        hi = RISK_BINS[i + 1]

        if i == len(RISK_BINS) - 2:
            mask = (
                (risk >= lo)
                &
                (risk <= hi)
            )
        else:
            mask = (
                (risk >= lo)
                &
                (risk < hi)
            )

        count = int(
            np.sum(mask)
        )

        if count == 0:
            row = {
                "risk_low": lo,
                "risk_high": min(hi, 1.0),
                "count": 0,
                "translation_error_mean": float("nan"),
                "translation_error_median": float("nan"),
                "translation_error_p90": float("nan"),
                "rotation_error_mean": float("nan"),
                "rotation_error_median": float("nan"),
                "rotation_error_p90": float("nan"),
            }

        else:
            row = {
                "risk_low": lo,
                "risk_high": min(hi, 1.0),
                "count": count,
                "translation_error_mean":
                    float(np.mean(te[mask])),
                "translation_error_median":
                    float(np.median(te[mask])),
                "translation_error_p90":
                    float(np.quantile(te[mask], 0.90)),
                "rotation_error_mean":
                    float(np.mean(re[mask])),
                "rotation_error_median":
                    float(np.median(re[mask])),
                "rotation_error_p90":
                    float(np.quantile(re[mask], 0.90)),
            }

        rows.append(row)

    return rows


def save_risk_analysis(
    rows,
):
    fields = [
        "risk_low",
        "risk_high",
        "count",
        "translation_error_mean",
        "translation_error_median",
        "translation_error_p90",
        "rotation_error_mean",
        "rotation_error_median",
        "rotation_error_p90",
    ]

    with open(
        RISK_ANALYSIS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_run_csv(
    rows,
):
    fields = [
        "name",
        "lambda",
        "ATE3D",
        "ATE2D",
        "Mean3D",
        "Max3D",
        "ZRMSE",
        "sigma_t_min",
        "sigma_t_max",
        "sigma_t_mean",
        "sigma_r_min",
        "sigma_r_max",
        "sigma_r_mean",
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
        writer.writerows(rows)


def make_run_row(
    name,
    lambda_value,
    result,
):
    metrics = result["metrics"]
    sigma_t = result["sigma_t"]
    sigma_r = result["sigma_r"]

    return {
        "name": name,
        "lambda": float(lambda_value),
        "ATE3D": float(metrics["ATE3D"]),
        "ATE2D": float(metrics["ATE2D"]),
        "Mean3D": float(metrics["Mean3D"]),
        "Max3D": float(metrics["Max3D"]),
        "ZRMSE": float(metrics["ZRMSE"]),
        "sigma_t_min": float(np.min(sigma_t)),
        "sigma_t_max": float(np.max(sigma_t)),
        "sigma_t_mean": float(np.mean(sigma_t)),
        "sigma_r_min": float(np.min(sigma_r)),
        "sigma_r_max": float(np.max(sigma_r)),
        "sigma_r_mean": float(np.mean(sigma_r)),
    }


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
        "LIDAR LAMBDA=1.0 VALIDATION / DETERMINISM / "
        "RISK-ERROR / MATCHED GT-ORACLE"
    )

    print(
        "=" * 120
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This script NEVER reads cached trajectories for the "
        "forced validation runs."
    )

    print(
        "Every call below invokes GTSAM optimization again."
    )

    # ========================================================
    # Load measurements
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
    # Prediction risk + GT realized factor errors
    # ========================================================

    pred_risk = (
        load_prediction_risk(
            pairs
        )
    )

    te, re = (
        load_gt_factor_errors(
            pairs
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "PREDICTION RISK / GT FACTOR ERROR DIAGNOSTICS"
    )

    print(
        "=" * 120
    )

    print(
        "Prediction risk min/max/mean:"
    )

    print(
        f"  {pred_risk.min():.6f} / "
        f"{pred_risk.max():.6f} / "
        f"{pred_risk.mean():.6f}"
    )

    print(
        "GT translation error min/max/mean:"
    )

    print(
        f"  {te.min():.6f} / "
        f"{te.max():.6f} / "
        f"{te.mean():.6f}"
    )

    print(
        "GT rotation error min/max/mean:"
    )

    print(
        f"  {re.min():.6f} / "
        f"{re.max():.6f} / "
        f"{re.mean():.6f}"
    )

    pred_t_pearson = pearson_corr(
        pred_risk,
        te,
    )

    pred_r_pearson = pearson_corr(
        pred_risk,
        re,
    )

    pred_t_spearman = spearman_corr(
        pred_risk,
        te,
    )

    pred_r_spearman = spearman_corr(
        pred_risk,
        re,
    )

    print()
    print(
        f"Risk vs translation error Pearson : "
        f"{pred_t_pearson:.6f}"
    )

    print(
        f"Risk vs translation error Spearman: "
        f"{pred_t_spearman:.6f}"
    )

    print(
        f"Risk vs rotation error Pearson    : "
        f"{pred_r_pearson:.6f}"
    )

    print(
        f"Risk vs rotation error Spearman   : "
        f"{pred_r_spearman:.6f}"
    )

    bin_rows = risk_bin_analysis(
        pred_risk,
        te,
        re,
    )

    save_risk_analysis(
        bin_rows
    )

    print()
    print(
        "Risk-bin realized factor errors:"
    )

    for row in bin_rows:
        print(
            f"  risk [{row['risk_low']:.1f}, "
            f"{row['risk_high']:.1f}] "
            f"n={row['count']:4d} | "
            f"tMean={row['translation_error_mean']:.6f} | "
            f"rMean={row['rotation_error_mean']:.6f}"
        )

    # ========================================================
    # 1) λ=1.0 repeated forced optimization
    # ========================================================

    print()
    print()
    print(
        "#" * 120
    )

    print(
        "TEST 1: FORCE λ=1.0 RE-OPTIMIZATION + DETERMINISM"
    )

    print(
        "#" * 120
    )

    lambda1_results = []
    run_rows = []

    for run_id in range(
        1,
        REPEAT_RUNS + 1,
    ):
        case_dir = os.path.join(
            OUT_ROOT,
            f"lambda_1p00_repeat_{run_id}",
        )

        print()
        print(
            f"λ=1.0 forced repeat "
            f"{run_id}/{REPEAT_RUNS}"
        )

        result = run_forced_case(
            measurements,
            pred_risk,
            LAMBDA_MAIN,
            case_dir,
        )

        lambda1_results.append(
            result
        )

        run_rows.append(
            make_run_row(
                f"predictive_lambda1_repeat_{run_id}",
                LAMBDA_MAIN,
                result,
            )
        )

        m = result["metrics"]

        print(
            f"RESULT repeat {run_id}: "
            f"ATE3D={m['ATE3D']:.6f}, "
            f"ATE2D={m['ATE2D']:.6f}, "
            f"ZRMSE={m['ZRMSE']:.6f}"
        )

    lambda1_ates = np.asarray(
        [
            item["metrics"]["ATE3D"]
            for item in lambda1_results
        ],
        dtype=np.float64,
    )

    print()
    print(
        "λ=1.0 ATE3D repeats:"
    )

    print(
        "  "
        +
        ", ".join(
            f"{x:.9f}"
            for x in lambda1_ates
        )
    )

    print(
        f"mean/std/range: "
        f"{lambda1_ates.mean():.9f} / "
        f"{lambda1_ates.std():.9f} / "
        f"{np.ptp(lambda1_ates):.9f}"
    )

    reference_delta = abs(
        float(lambda1_ates[0])
        -
        REFERENCE_LAMBDA1_ATE
    )

    print(
        f"First rerun vs cached 1.728809 difference: "
        f"{reference_delta:.9f} m"
    )

    # Pairwise trajectory determinism
    base_traj = (
        lambda1_results[0][
            "trajectory"
        ]
    )

    determinism_rows = []

    for idx in range(
        1,
        len(lambda1_results),
    ):
        diff = trajectory_difference(
            base_traj,
            lambda1_results[idx][
                "trajectory"
            ],
        )

        determinism_rows.append(
            diff
        )

        print(
            f"Trajectory repeat1 vs repeat{idx + 1}: "
            f"max={diff['max_position_difference']:.12f} m, "
            f"mean={diff['mean_position_difference']:.12f} m"
        )

    # ========================================================
    # 2) Force λ=0.5 anomaly rerun
    # ========================================================

    print()
    print()
    print(
        "#" * 120
    )

    print(
        "TEST 2: FORCE λ=0.5 ANOMALY RE-OPTIMIZATION"
    )

    print(
        "#" * 120
    )

    lambda05_result = run_forced_case(
        measurements,
        pred_risk,
        LAMBDA_ANOMALY,
        os.path.join(
            OUT_ROOT,
            "lambda_0p50_forced",
        ),
    )

    run_rows.append(
        make_run_row(
            "predictive_lambda0p5_forced",
            LAMBDA_ANOMALY,
            lambda05_result,
        )
    )

    m05 = lambda05_result[
        "metrics"
    ]

    print()
    print(
        f"λ=0.5 forced ATE3D: "
        f"{m05['ATE3D']:.6f}"
    )

    print(
        "Previous sweep λ=0.5 reference: "
        "2.613778"
    )

    print(
        f"Absolute difference: "
        f"{abs(m05['ATE3D'] - 2.613778):.9f} m"
    )

    # ========================================================
    # 3) Matched GT-error oracle
    # ========================================================

    print()
    print()
    print(
        "#" * 120
    )

    print(
        "TEST 3: APPLES-TO-APPLES MATCHED GT-ERROR ORACLE"
    )

    print(
        "#" * 120
    )

    (
        gt_risk,
        gt_thresholds,
    ) = build_matched_gt_oracle_risk(
        te,
        re,
    )

    gt_sigma_t, gt_sigma_r, gt_scale = (
        build_sigma_from_risk(
            gt_risk,
            LAMBDA_MAIN,
        )
    )

    print()
    print(
        "GT oracle risk construction:"
    )

    print(
        f"  translation p50/p90: "
        f"{gt_thresholds['translation_p50']:.6f} / "
        f"{gt_thresholds['translation_p90']:.6f}"
    )

    print(
        f"  rotation p50/p90   : "
        f"{gt_thresholds['rotation_p50']:.6f} / "
        f"{gt_thresholds['rotation_p90']:.6f}"
    )

    print(
        f"  GT risk min/max/mean: "
        f"{gt_risk.min():.6f} / "
        f"{gt_risk.max():.6f} / "
        f"{gt_risk.mean():.6f}"
    )

    print(
        f"  GT sigma_t min/max/mean: "
        f"{gt_sigma_t.min():.6f} / "
        f"{gt_sigma_t.max():.6f} / "
        f"{gt_sigma_t.mean():.6f}"
    )

    np.savez_compressed(
        MATCHED_ORACLE_PATH,
        gt_translation_error=te,
        gt_rotation_error=re,
        gt_risk_score_pair=gt_risk,
        translation_sigma_pair=gt_sigma_t,
        rotation_sigma_pair=gt_sigma_r,
        scale_pair=gt_scale,
        translation_p50=
            gt_thresholds["translation_p50"],
        translation_p90=
            gt_thresholds["translation_p90"],
        rotation_p50=
            gt_thresholds["rotation_p50"],
        rotation_p90=
            gt_thresholds["rotation_p90"],
        lambda_value=LAMBDA_MAIN,
        sigma_t0=SIGMA_T0,
        sigma_r0=SIGMA_R0,
    )

    gt_oracle_result = run_forced_case(
        measurements,
        gt_risk,
        LAMBDA_MAIN,
        os.path.join(
            OUT_ROOT,
            "matched_gt_oracle_lambda_1p00",
        ),
    )

    run_rows.append(
        make_run_row(
            "matched_gt_oracle_lambda1",
            LAMBDA_MAIN,
            gt_oracle_result,
        )
    )

    save_run_csv(
        run_rows
    )

    gt_metrics = gt_oracle_result[
        "metrics"
    ]

    # ========================================================
    # Final comparison
    # ========================================================

    predictive_metrics = (
        lambda1_results[0][
            "metrics"
        ]
    )

    pred_ate = float(
        predictive_metrics[
            "ATE3D"
        ]
    )

    oracle_ate = float(
        gt_metrics[
            "ATE3D"
        ]
    )

    improvement_vs_fixed = (
        (
            REFERENCE_FIXED_ATE
            -
            pred_ate
        )
        /
        REFERENCE_FIXED_ATE
        *
        100.0
    )

    pred_vs_matched_oracle = (
        (
            pred_ate
            -
            oracle_ate
        )
        /
        oracle_ate
        *
        100.0
    )

    print()
    print()
    print(
        "=" * 120
    )

    print(
        "FINAL VALIDATION COMPARISON"
    )

    print(
        "=" * 120
    )

    print(
        f"Optimized Fixed LiDAR      : "
        f"{REFERENCE_FIXED_ATE:.6f} m"
    )

    print(
        f"Predictive Dynamic λ=1.0   : "
        f"{pred_ate:.6f} m"
    )

    print(
        f"Matched GT Oracle λ=1.0    : "
        f"{oracle_ate:.6f} m"
    )

    print()
    print(
        f"Predictive improvement vs fixed: "
        f"{improvement_vs_fixed:+.2f}%"
    )

    print(
        f"Predictive gap vs matched oracle: "
        f"{pred_vs_matched_oracle:+.2f}%"
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "Matched GT Oracle uses realized GT factor error to "
        "construct risk, then uses the SAME lambda=1 covariance "
        "formula and SAME sigma0 values as the predictive run."
    )

    print(
        "It is an oracle diagnostic and intentionally uses "
        "ground truth; it is not a deployable method."
    )

    # ========================================================
    # Validation decisions
    # ========================================================

    deterministic = (
        np.ptp(
            lambda1_ates
        )
        <
        1e-6
    )

    cached_reproduced = (
        reference_delta
        <
        1e-4
    )

    lambda05_reproduced = (
        abs(
            m05["ATE3D"]
            -
            2.613778
        )
        <
        1e-4
    )

    dynamic_beats_fixed = (
        pred_ate
        <
        REFERENCE_FIXED_ATE
    )

    print()
    print(
        "=" * 120
    )

    print(
        "VALIDATION CHECKS"
    )

    print(
        "=" * 120
    )

    print(
        f"[{'PASS' if cached_reproduced else 'FAIL'}] "
        "λ=1 forced rerun reproduces cached 1.728809"
    )

    print(
        f"[{'PASS' if deterministic else 'FAIL'}] "
        "Repeated λ=1 optimization is deterministic "
        "(ATE range < 1e-6)"
    )

    print(
        f"[{'PASS' if lambda05_reproduced else 'FAIL'}] "
        "λ=0.5 forced rerun reproduces 2.613778 anomaly"
    )

    print(
        f"[{'PASS' if dynamic_beats_fixed else 'FAIL'}] "
        "Predictive λ=1 beats optimized fixed baseline"
    )

    # ========================================================
    # Save report
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "LIDAR LAMBDA=1 VALIDATION REPORT\n"
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
            f"LiDAR pairs: {pairs}\n\n"
        )

        file.write(
            "PREDICTION RISK VS REALIZED GT FACTOR ERROR\n"
        )

        file.write(
            f"Pearson risk-vs-translation: "
            f"{pred_t_pearson:.10f}\n"
        )

        file.write(
            f"Spearman risk-vs-translation: "
            f"{pred_t_spearman:.10f}\n"
        )

        file.write(
            f"Pearson risk-vs-rotation: "
            f"{pred_r_pearson:.10f}\n"
        )

        file.write(
            f"Spearman risk-vs-rotation: "
            f"{pred_r_spearman:.10f}\n\n"
        )

        file.write(
            "LAMBDA=1 FORCED REPEATS\n"
        )

        for idx, ate in enumerate(
            lambda1_ates,
            start=1,
        ):
            file.write(
                f"repeat_{idx}_ATE3D: "
                f"{ate:.10f}\n"
            )

        file.write(
            f"ATE_mean: "
            f"{lambda1_ates.mean():.10f}\n"
        )

        file.write(
            f"ATE_std: "
            f"{lambda1_ates.std():.10f}\n"
        )

        file.write(
            f"ATE_range: "
            f"{np.ptp(lambda1_ates):.10f}\n"
        )

        file.write(
            f"cached_reference_difference: "
            f"{reference_delta:.10f}\n\n"
        )

        file.write(
            "LAMBDA=0.5 FORCED CHECK\n"
        )

        file.write(
            f"ATE3D: "
            f"{m05['ATE3D']:.10f}\n"
        )

        file.write(
            f"reference_difference: "
            f"{abs(m05['ATE3D'] - 2.613778):.10f}\n\n"
        )

        file.write(
            "MATCHED GT ORACLE\n"
        )

        file.write(
            f"translation_p50: "
            f"{gt_thresholds['translation_p50']:.10f}\n"
        )

        file.write(
            f"translation_p90: "
            f"{gt_thresholds['translation_p90']:.10f}\n"
        )

        file.write(
            f"rotation_p50: "
            f"{gt_thresholds['rotation_p50']:.10f}\n"
        )

        file.write(
            f"rotation_p90: "
            f"{gt_thresholds['rotation_p90']:.10f}\n"
        )

        file.write(
            f"matched_oracle_ATE3D: "
            f"{oracle_ate:.10f}\n\n"
        )

        file.write(
            "FINAL COMPARISON\n"
        )

        file.write(
            f"fixed_ATE3D: "
            f"{REFERENCE_FIXED_ATE:.10f}\n"
        )

        file.write(
            f"predictive_lambda1_ATE3D: "
            f"{pred_ate:.10f}\n"
        )

        file.write(
            f"matched_gt_oracle_ATE3D: "
            f"{oracle_ate:.10f}\n"
        )

        file.write(
            f"improvement_vs_fixed_percent: "
            f"{improvement_vs_fixed:.6f}\n"
        )

        file.write(
            f"gap_vs_matched_oracle_percent: "
            f"{pred_vs_matched_oracle:.6f}\n\n"
        )

        file.write(
            "CHECKS\n"
        )

        file.write(
            f"cached_reproduced: "
            f"{cached_reproduced}\n"
        )

        file.write(
            f"deterministic: "
            f"{deterministic}\n"
        )

        file.write(
            f"lambda05_reproduced: "
            f"{lambda05_reproduced}\n"
        )

        file.write(
            f"dynamic_beats_fixed: "
            f"{dynamic_beats_fixed}\n"
        )

    print()
    print(
        "=" * 120
    )

    print(
        "SAVED"
    )

    print(
        "=" * 120
    )

    print(
        REPORT_PATH
    )

    print(
        CSV_PATH
    )

    print(
        RISK_ANALYSIS_PATH
    )

    print(
        MATCHED_ORACLE_PATH
    )


if __name__ == "__main__":
    main()
