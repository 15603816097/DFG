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

METHODS = [
    (
        "Fixed Covariance",
        "fair_progressive_fixed_fg",
    ),
    (
        "Old Predictive Only",
        "fair_predictive_only_fg",
    ),
    (
        "Old Predictive + Feedback",
        "fair_predictive_feedback_fg",
    ),
    (
        "Calibrated Predictive Only",
        "calibrated_predictive_only_fg",
    ),
    (
        "Calibrated Predictive + FG Feedback",
        "calibrated_predictive_feedback_fg",
    ),
    (
        "Oracle",
        "fair_oracle_progressive_fg",
    ),
]

OUTPUT_PATH = os.path.join(
    ROOT,
    "results",
    "calibrated_feedback_evaluation.txt",
)


def calculate(
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


def main():
    gt = np.loadtxt(
        GT_PATH,
        dtype=np.float64,
    )

    rows = []

    print("=" * 88)
    print(
        "CALIBRATED RELIABILITY + "
        "FACTOR-GRAPH FEEDBACK EVALUATION"
    )
    print("=" * 88)

    for name, folder in METHODS:
        path = os.path.join(
            ROOT,
            "results",
            folder,
            "trajectory.txt",
        )

        if not os.path.exists(
            path
        ):
            print(
                "SKIP:",
                name,
                "(missing)",
            )

            continue

        trajectory = np.loadtxt(
            path,
            dtype=np.float64,
        )

        result = calculate(
            gt,
            trajectory,
        )

        rows.append(
            (
                name,
                result,
            )
        )

        print()
        print(name)

        for key, value in (
            result.items()
        ):
            print(
                f"{key:10s}: "
                f"{value:.6f}"
            )

    if not rows:
        raise RuntimeError(
            "No results to evaluate."
        )

    fixed_value = None

    for name, result in rows:
        if (
            name
            ==
            "Fixed Covariance"
        ):
            fixed_value = (
                result["ATE3D"]
            )

    print()
    print("=" * 88)
    print(
        f'{"Method":40s}'
        f'{"ATE3D":>12s}'
        f'{"ATE2D":>12s}'
        f'{"vs Fixed":>14s}'
    )
    print("-" * 88)

    lines = []

    for name, result in rows:
        if fixed_value is None:
            improvement = 0.0
        else:
            improvement = (
                100.0
                *
                (
                    fixed_value
                    -
                    result[
                        "ATE3D"
                    ]
                )
                /
                max(
                    fixed_value,
                    1e-12,
                )
            )

        print(
            f'{name:40s}'
            f'{result["ATE3D"]:12.6f}'
            f'{result["ATE2D"]:12.6f}'
            f'{improvement:13.2f}%'
        )

        lines.append(
            f"{name},"
            f'{result["ATE3D"]:.8f},'
            f'{result["ATE2D"]:.8f},'
            f'{result["Mean3D"]:.8f},'
            f'{result["Max3D"]:.8f},'
            f'{result["ZRMSE"]:.8f},'
            f"{improvement:.4f}\n"
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Method,ATE3D,ATE2D,"
            "Mean3D,Max3D,ZRMSE,"
            "ImprovementVsFixed\n"
        )

        f.writelines(
            lines
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
