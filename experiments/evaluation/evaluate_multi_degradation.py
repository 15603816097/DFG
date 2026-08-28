"""
Evaluate Multi-Degradation Robustness
=====================================

Produces:
    results/multi_degradation_summary.csv
    results/multi_degradation_summary.txt
"""

from __future__ import annotations

import os
import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

GT_PATH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)

SCENARIO_ROOT = os.path.join(
    ROOT,
    "results",
    "multi_degradation",
)

SCENARIOS = [
    "mild_progressive",
    "medium_progressive",
    "severe_progressive",
    "sudden",
    "bias_drift",
    "intermittent_outlier",
]

METHODS = [
    (
        "Fixed",
        "fixed_fg",
    ),
    (
        "RobustFixed",
        "robust_fixed_fg",
    ),
    (
        "Proposed",
        "proposed_fg",
    ),
    (
        "Oracle",
        "oracle_fg",
    ),
]

CSV_PATH = os.path.join(
    ROOT,
    "results",
    "multi_degradation_summary.csv",
)

TXT_PATH = os.path.join(
    ROOT,
    "results",
    "multi_degradation_summary.txt",
)


def trajectory_metrics(
    gt,
    trajectory,
):
    n = min(
        len(gt),
        len(trajectory),
    )

    error = (
        trajectory[:n]
        -
        gt[:n]
    )

    e3 = np.linalg.norm(
        error,
        axis=1,
    )

    e2 = np.linalg.norm(
        error[:, :2],
        axis=1,
    )

    return {
        "ATE3D":
            float(
                np.sqrt(
                    np.mean(
                        e3 ** 2
                    )
                )
            ),

        "ATE2D":
            float(
                np.sqrt(
                    np.mean(
                        e2 ** 2
                    )
                )
            ),

        "Mean3D":
            float(
                np.mean(
                    e3
                )
            ),

        "Max3D":
            float(
                np.max(
                    e3
                )
            ),

        "ZRMSE":
            float(
                np.sqrt(
                    np.mean(
                        error[:, 2]
                        ** 2
                    )
                )
            ),
    }


def reliability_metrics(
    gt,
    gps,
    predictive_prior,
):
    n = min(
        len(gt),
        len(gps),
        len(predictive_prior),
    )

    gt = gt[:n]
    gps = gps[:n]

    p = np.clip(
        predictive_prior[:n],
        0.0,
        1.0,
    )

    error = np.linalg.norm(
        gps
        -
        gt,
        axis=1,
    )

    oracle_like = np.exp(
        -0.5
        *
        (
            error
            /
            8.0
        )
        ** 2
    )

    valid = np.arange(
        n
    ) >= 66

    y = oracle_like[
        valid
    ]

    p = p[
        valid
    ]

    mae = float(
        np.mean(
            np.abs(
                y
                -
                p
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    y
                    -
                    p
                )
                ** 2
            )
        )
    )

    if (
        np.std(y) > 1e-12
        and
        np.std(p) > 1e-12
    ):
        corr = float(
            np.corrcoef(
                y,
                p,
            )[0, 1]
        )

    else:
        corr = 0.0

    return (
        mae,
        rmse,
        corr,
    )


def main():
    gt = np.loadtxt(
        GT_PATH,
        dtype=np.float64,
    )

    rows = []

    print("=" * 116)
    print(
        "MULTI-DEGRADATION ROBUSTNESS "
        "EVALUATION"
    )
    print("=" * 116)

    for scenario in SCENARIOS:
        folder = os.path.join(
            SCENARIO_ROOT,
            scenario,
        )

        gps = np.loadtxt(
            os.path.join(
                folder,
                "gps_corrupted.txt",
            ),
            dtype=np.float64,
        )

        prior = np.loadtxt(
            os.path.join(
                folder,
                "predictive_prior.txt",
            ),
            dtype=np.float64,
        ).reshape(-1)

        (
            r_mae,
            r_rmse,
            r_corr,
        ) = reliability_metrics(
            gt,
            gps,
            prior,
        )

        scenario_rows = []

        for (
            method_name,
            method_folder,
        ) in METHODS:

            trajectory_path = os.path.join(
                folder,
                method_folder,
                "trajectory.txt",
            )

            if not os.path.exists(
                trajectory_path
            ):
                raise FileNotFoundError(
                    trajectory_path
                )

            trajectory = np.loadtxt(
                trajectory_path,
                dtype=np.float64,
            )

            metrics = (
                trajectory_metrics(
                    gt,
                    trajectory,
                )
            )

            row = {
                "Scenario":
                    scenario,

                "Method":
                    method_name,

                "ATE3D":
                    metrics[
                        "ATE3D"
                    ],

                "ATE2D":
                    metrics[
                        "ATE2D"
                    ],

                "Mean3D":
                    metrics[
                        "Mean3D"
                    ],

                "Max3D":
                    metrics[
                        "Max3D"
                    ],

                "ZRMSE":
                    metrics[
                        "ZRMSE"
                    ],

                "ReliabilityMAE":
                    r_mae,

                "ReliabilityRMSE":
                    r_rmse,

                "ReliabilityCorr":
                    r_corr,
            }

            scenario_rows.append(
                row
            )

            rows.append(
                row
            )

        fixed_ate = next(
            x["ATE3D"]
            for x in scenario_rows
            if x["Method"] == "Fixed"
        )

        robust_ate = next(
            x["ATE3D"]
            for x in scenario_rows
            if x["Method"] == "RobustFixed"
        )

        proposed_ate = next(
            x["ATE3D"]
            for x in scenario_rows
            if x["Method"] == "Proposed"
        )

        oracle_ate = next(
            x["ATE3D"]
            for x in scenario_rows
            if x["Method"] == "Oracle"
        )

        improve_fixed = (
            100.0
            *
            (
                fixed_ate
                -
                proposed_ate
            )
            /
            max(
                fixed_ate,
                1e-12,
            )
        )

        improve_robust = (
            100.0
            *
            (
                robust_ate
                -
                proposed_ate
            )
            /
            max(
                robust_ate,
                1e-12,
            )
        )

        print()
        print(
            f"{scenario:24s}"
            f" Fixed={fixed_ate:8.4f}"
            f" Robust={robust_ate:8.4f}"
            f" Proposed={proposed_ate:8.4f}"
            f" Oracle={oracle_ate:8.4f}"
            f" | vs Fixed={improve_fixed:7.2f}%"
            f" | vs Robust={improve_robust:7.2f}%"
            f" | Rcorr={r_corr:7.3f}"
        )

    # Add improvement columns.
    enriched = []

    for scenario in SCENARIOS:
        current = [
            row
            for row in rows
            if row[
                "Scenario"
            ] == scenario
        ]

        fixed_ate = next(
            x["ATE3D"]
            for x in current
            if x["Method"] == "Fixed"
        )

        robust_ate = next(
            x["ATE3D"]
            for x in current
            if x[
                "Method"
            ] == "RobustFixed"
        )

        for row in current:
            new_row = dict(
                row
            )

            if (
                row["Method"]
                ==
                "Proposed"
            ):
                new_row[
                    "ImproveVsFixed"
                ] = (
                    100.0
                    *
                    (
                        fixed_ate
                        -
                        row[
                            "ATE3D"
                        ]
                    )
                    /
                    max(
                        fixed_ate,
                        1e-12,
                    )
                )

                new_row[
                    "ImproveVsRobust"
                ] = (
                    100.0
                    *
                    (
                        robust_ate
                        -
                        row[
                            "ATE3D"
                        ]
                    )
                    /
                    max(
                        robust_ate,
                        1e-12,
                    )
                )

            else:
                new_row[
                    "ImproveVsFixed"
                ] = np.nan

                new_row[
                    "ImproveVsRobust"
                ] = np.nan

            enriched.append(
                new_row
            )

    with open(
        CSV_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Scenario,Method,ATE3D,ATE2D,"
            "Mean3D,Max3D,ZRMSE,"
            "ReliabilityMAE,ReliabilityRMSE,"
            "ReliabilityCorr,"
            "ImproveVsFixed_percent,"
            "ImproveVsRobust_percent\n"
        )

        for row in enriched:
            file.write(
                f'{row["Scenario"]},'
                f'{row["Method"]},'
                f'{row["ATE3D"]:.8f},'
                f'{row["ATE2D"]:.8f},'
                f'{row["Mean3D"]:.8f},'
                f'{row["Max3D"]:.8f},'
                f'{row["ZRMSE"]:.8f},'
                f'{row["ReliabilityMAE"]:.8f},'
                f'{row["ReliabilityRMSE"]:.8f},'
                f'{row["ReliabilityCorr"]:.8f},'
                f'{row["ImproveVsFixed"]},'
                f'{row["ImproveVsRobust"]}\n'
            )

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Multi-Degradation Robustness Summary\n"
        )

        file.write(
            "=" * 90
            +
            "\n"
        )

        for scenario in SCENARIOS:
            current = [
                x
                for x in enriched
                if x[
                    "Scenario"
                ] == scenario
            ]

            proposed = next(
                x
                for x in current
                if x[
                    "Method"
                ] == "Proposed"
            )

            fixed = next(
                x
                for x in current
                if x[
                    "Method"
                ] == "Fixed"
            )

            robust = next(
                x
                for x in current
                if x[
                    "Method"
                ] == "RobustFixed"
            )

            oracle = next(
                x
                for x in current
                if x[
                    "Method"
                ] == "Oracle"
            )

            file.write(
                f"{scenario:24s} "
                f"Fixed={fixed['ATE3D']:.6f} "
                f"Robust={robust['ATE3D']:.6f} "
                f"Proposed={proposed['ATE3D']:.6f} "
                f"Oracle={oracle['ATE3D']:.6f} "
                f"vsFixed={proposed['ImproveVsFixed']:.2f}% "
                f"vsRobust={proposed['ImproveVsRobust']:.2f}% "
                f"Rcorr={proposed['ReliabilityCorr']:.4f}\n"
            )

    print()
    print(
        "Saved:",
        CSV_PATH,
    )

    print(
        "Saved:",
        TXT_PATH,
    )


if __name__ == "__main__":
    main()
