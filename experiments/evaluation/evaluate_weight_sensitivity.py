"""
Evaluate prediction/feedback weight sensitivity.

Expected inputs:
    results/weight_sensitivity/alpha_0p50/
    results/weight_sensitivity/alpha_0p60/
    ...
    results/weight_sensitivity/alpha_1p00/

Output:
    results/weight_sensitivity_summary.csv
    results/weight_sensitivity_summary.txt
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


ALPHAS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]


GT_PATH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
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
    "weight_sensitivity_summary.csv",
)


TXT_PATH = os.path.join(
    ROOT,
    "results",
    "weight_sensitivity_summary.txt",
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

    error_3d = np.linalg.norm(
        error,
        axis=1,
    )

    error_2d = np.linalg.norm(
        error[:, :2],
        axis=1,
    )

    return {
        "ATE3D":
            float(
                np.sqrt(
                    np.mean(
                        error_3d ** 2
                    )
                )
            ),

        "ATE2D":
            float(
                np.sqrt(
                    np.mean(
                        error_2d ** 2
                    )
                )
            ),

        "Mean3D":
            float(
                np.mean(
                    error_3d
                )
            ),

        "Max3D":
            float(
                np.max(
                    error_3d
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


def main():
    gt = np.loadtxt(
        GT_PATH,
        dtype=np.float64,
    )

    fixed_ate = None

    if os.path.exists(
        FIXED_PATH
    ):
        fixed_trajectory = (
            np.loadtxt(
                FIXED_PATH,
                dtype=np.float64,
            )
        )

        fixed_ate = (
            trajectory_metrics(
                gt,
                fixed_trajectory,
            )[
                "ATE3D"
            ]
        )

    rows = []

    print("=" * 96)

    print(
        "PREDICTION / FEEDBACK "
        "WEIGHT SENSITIVITY"
    )

    print("=" * 96)

    for alpha in ALPHAS:
        tag = (
            f"{alpha:.2f}"
            .replace(
                ".",
                "p",
            )
        )

        folder = os.path.join(
            ROOT,
            "results",
            "weight_sensitivity",
            f"alpha_{tag}",
        )

        trajectory_path = os.path.join(
            folder,
            "trajectory.txt",
        )

        reliability_path = os.path.join(
            folder,
            "final_reliability.txt",
        )

        sigma_path = os.path.join(
            folder,
            "sigma.txt",
        )

        if not os.path.exists(
            trajectory_path
        ):
            print(
                "SKIP alpha =",
                alpha,
            )

            continue

        trajectory = np.loadtxt(
            trajectory_path,
            dtype=np.float64,
        )

        final_reliability = (
            np.loadtxt(
                reliability_path,
                dtype=np.float64,
            )
        )

        sigma = np.loadtxt(
            sigma_path,
            dtype=np.float64,
        )

        result = trajectory_metrics(
            gt,
            trajectory,
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
                    result[
                        "ATE3D"
                    ]
                )
                /
                max(
                    fixed_ate,
                    1e-12,
                )
            )

        rows.append(
            {
                "alpha":
                    alpha,

                "feedback_weight":
                    1.0
                    -
                    alpha,

                "ATE3D":
                    result[
                        "ATE3D"
                    ],

                "ATE2D":
                    result[
                        "ATE2D"
                    ],

                "Mean3D":
                    result[
                        "Mean3D"
                    ],

                "Max3D":
                    result[
                        "Max3D"
                    ],

                "ZRMSE":
                    result[
                        "ZRMSE"
                    ],

                "ReliabilityMean":
                    float(
                        np.mean(
                            final_reliability
                        )
                    ),

                "SigmaMean":
                    float(
                        np.mean(
                            sigma
                        )
                    ),

                "ImprovementVsFixed":
                    improvement,
            }
        )

    if not rows:
        raise RuntimeError(
            "No weight sensitivity "
            "results found."
        )

    print(
        f'{"Alpha":>8s}'
        f'{"Feedback":>12s}'
        f'{"ATE3D":>12s}'
        f'{"ATE2D":>12s}'
        f'{"SigmaMean":>14s}'
        f'{"vs Fixed":>14s}'
    )

    print(
        "-" * 96
    )

    for row in rows:
        print(
            f'{row["alpha"]:8.2f}'
            f'{row["feedback_weight"]:12.2f}'
            f'{row["ATE3D"]:12.6f}'
            f'{row["ATE2D"]:12.6f}'
            f'{row["SigmaMean"]:14.6f}'
            f'{row["ImprovementVsFixed"]:13.2f}%'
        )

    best = min(
        rows,
        key=lambda row:
            row["ATE3D"],
    )

    print()
    print(
        "Best alpha:",
        best["alpha"],
    )

    print(
        "Best prediction weight:",
        best["alpha"],
    )

    print(
        "Best feedback weight:",
        best[
            "feedback_weight"
        ],
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

    with open(
        CSV_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Alpha,FeedbackWeight,"
            "ATE3D,ATE2D,Mean3D,"
            "Max3D,ZRMSE,"
            "ReliabilityMean,"
            "SigmaMean,"
            "ImprovementVsFixed_percent\n"
        )

        for row in rows:
            file.write(
                f'{row["alpha"]:.4f},'
                f'{row["feedback_weight"]:.4f},'
                f'{row["ATE3D"]:.8f},'
                f'{row["ATE2D"]:.8f},'
                f'{row["Mean3D"]:.8f},'
                f'{row["Max3D"]:.8f},'
                f'{row["ZRMSE"]:.8f},'
                f'{row["ReliabilityMean"]:.8f},'
                f'{row["SigmaMean"]:.8f},'
                f'{row["ImprovementVsFixed"]:.4f}\n'
            )

    with open(
        TXT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Prediction / Feedback "
            "Weight Sensitivity\n"
        )

        file.write(
            "=" * 70
            +
            "\n"
        )

        for row in rows:
            file.write(
                f'alpha={row["alpha"]:.2f} '
                f'feedback='
                f'{row["feedback_weight"]:.2f} '
                f'ATE3D='
                f'{row["ATE3D"]:.6f} '
                f'SigmaMean='
                f'{row["SigmaMean"]:.6f} '
                f'vsFixed='
                f'{row["ImprovementVsFixed"]:.2f}%\n'
            )

        file.write(
            "\nBest alpha: "
            f'{best["alpha"]:.2f}\n'
        )

        file.write(
            "Best feedback weight: "
            f'{best["feedback_weight"]:.2f}\n'
        )

        file.write(
            "Best ATE3D: "
            f'{best["ATE3D"]:.8f}\n'
        )

        file.write(
            "Improvement vs Fixed: "
            f'{best["ImprovementVsFixed"]:.4f}%\n'
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
