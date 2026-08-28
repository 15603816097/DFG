"""
Evaluate horizon sensitivity after all horizon models and factor graphs
have finished.

Outputs
-------
results/horizon_sensitivity_summary.csv
results/horizon_sensitivity_summary.txt
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


HORIZONS = [
    1,
    3,
    5,
    10,
    20,
]


GT_PATH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)

CURRENT_LABEL_PATH = os.path.join(
    ROOT,
    "results",
    "predictive_reliability_labels",
    "current_reliability.txt",
)

FIXED_PATH = os.path.join(
    ROOT,
    "results",
    "fair_progressive_fixed_fg",
    "trajectory.txt",
)

CSV_PATH = os.path.join(
    ROOT,
    "results",
    "horizon_sensitivity_summary.csv",
)

TXT_PATH = os.path.join(
    ROOT,
    "results",
    "horizon_sensitivity_summary.txt",
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

    return (
        float(
            np.sqrt(
                np.mean(
                    e3 ** 2
                )
            )
        ),
        float(
            np.sqrt(
                np.mean(
                    e2 ** 2
                )
            )
        ),
        float(
            np.mean(
                e3
            )
        ),
        float(
            np.max(
                e3
            )
        ),
    )


def reliability_metrics(
    current_reliability,
    predictive_prior,
    horizon,
):
    """
    Compare target-aligned prior with actual current reliability.
    This measures whether a prediction made H frames ago correctly
    describes the reliability at the target frame.
    """
    n = min(
        len(current_reliability),
        len(predictive_prior),
    )

    y = np.asarray(
        current_reliability[:n],
        dtype=np.float64,
    )

    p = np.asarray(
        predictive_prior[:n],
        dtype=np.float64,
    )

    valid = np.arange(
        n
    ) >= (
        63
        +
        horizon
    )

    y = y[
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

    current_reliability = np.loadtxt(
        CURRENT_LABEL_PATH,
        dtype=np.float64,
    ).reshape(-1)

    fixed_ate = None

    if os.path.exists(
        FIXED_PATH
    ):
        fixed_trajectory = np.loadtxt(
            FIXED_PATH,
            dtype=np.float64,
        )

        fixed_ate = (
            trajectory_metrics(
                gt,
                fixed_trajectory,
            )[0]
        )

    rows = []

    print("=" * 100)
    print(
        "HORIZON SENSITIVITY EVALUATION"
    )
    print("=" * 100)

    for horizon in HORIZONS:
        folder = os.path.join(
            ROOT,
            "results",
            "horizon_sensitivity",
            f"h{horizon}",
        )

        trajectory_path = os.path.join(
            folder,
            "fg_trajectory.txt",
        )

        prior_path = os.path.join(
            folder,
            "predictive_prior_target_aligned.txt",
        )

        if not (
            os.path.exists(
                trajectory_path
            )
            and
            os.path.exists(
                prior_path
            )
        ):
            print(
                "SKIP H =",
                horizon,
                "(missing outputs)",
            )

            continue

        trajectory = np.loadtxt(
            trajectory_path,
            dtype=np.float64,
        )

        prior = np.loadtxt(
            prior_path,
            dtype=np.float64,
        ).reshape(-1)

        (
            ate3d,
            ate2d,
            mean3d,
            max3d,
        ) = trajectory_metrics(
            gt,
            trajectory,
        )

        (
            r_mae,
            r_rmse,
            r_corr,
        ) = reliability_metrics(
            current_reliability,
            prior,
            horizon,
        )

        if fixed_ate is None:
            improvement = 0.0

        else:
            improvement = (
                100.0
                *
                (
                    fixed_ate
                    -
                    ate3d
                )
                /
                max(
                    fixed_ate,
                    1e-12,
                )
            )

        rows.append(
            {
                "horizon":
                    horizon,
                "lead_time_s":
                    horizon
                    *
                    0.1,
                "ATE3D":
                    ate3d,
                "ATE2D":
                    ate2d,
                "Mean3D":
                    mean3d,
                "Max3D":
                    max3d,
                "ReliabilityMAE":
                    r_mae,
                "ReliabilityRMSE":
                    r_rmse,
                "ReliabilityCorr":
                    r_corr,
                "ImprovementVsFixed":
                    improvement,
            }
        )

    if not rows:
        raise RuntimeError(
            "No horizon results found."
        )

    print(
        f'{"H":>4s}'
        f'{"Lead(s)":>10s}'
        f'{"ATE3D":>12s}'
        f'{"ATE2D":>12s}'
        f'{"R-Corr":>12s}'
        f'{"vs Fixed":>14s}'
    )

    print(
        "-" * 100
    )

    for row in rows:
        print(
            f'{row["horizon"]:4d}'
            f'{row["lead_time_s"]:10.2f}'
            f'{row["ATE3D"]:12.6f}'
            f'{row["ATE2D"]:12.6f}'
            f'{row["ReliabilityCorr"]:12.6f}'
            f'{row["ImprovementVsFixed"]:13.2f}%'
        )

    best = min(
        rows,
        key=lambda x:
            x["ATE3D"],
    )

    print()
    print(
        "Best horizon:",
        best["horizon"],
        "frames",
    )

    print(
        "Best lead time:",
        best[
            "lead_time_s"
        ],
        "s",
    )

    print(
        "Best ATE3D:",
        best["ATE3D"],
    )

    print(
        "Improvement vs Fixed:",
        best[
            "ImprovementVsFixed"
        ],
        "%",
    )

    header = (
        "Horizon,LeadTime_s,ATE3D,ATE2D,"
        "Mean3D,Max3D,ReliabilityMAE,"
        "ReliabilityRMSE,ReliabilityCorr,"
        "ImprovementVsFixed_percent\n"
    )

    with open(
        CSV_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            header
        )

        for row in rows:
            f.write(
                f'{row["horizon"]},'
                f'{row["lead_time_s"]:.4f},'
                f'{row["ATE3D"]:.8f},'
                f'{row["ATE2D"]:.8f},'
                f'{row["Mean3D"]:.8f},'
                f'{row["Max3D"]:.8f},'
                f'{row["ReliabilityMAE"]:.8f},'
                f'{row["ReliabilityRMSE"]:.8f},'
                f'{row["ReliabilityCorr"]:.8f},'
                f'{row["ImprovementVsFixed"]:.4f}\n'
            )

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Horizon Sensitivity Summary\n"
        )

        f.write(
            "=" * 70
            +
            "\n"
        )

        for row in rows:
            f.write(
                f'H={row["horizon"]:2d} '
                f'Lead={row["lead_time_s"]:.2f}s '
                f'ATE3D={row["ATE3D"]:.6f} '
                f'Rcorr={row["ReliabilityCorr"]:.6f} '
                f'vsFixed={row["ImprovementVsFixed"]:.2f}%\n'
            )

        f.write(
            "\nBest horizon: "
            f'{best["horizon"]}\n'
        )

        f.write(
            "Best lead time: "
            f'{best["lead_time_s"]:.2f}s\n'
        )

        f.write(
            "Best ATE3D: "
            f'{best["ATE3D"]:.8f}\n'
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
