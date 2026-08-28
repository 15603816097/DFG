"""
Complete Prediction / Feedback Weight Sensitivity Experiment
============================================================

Fixed:
    H = 3
    model = existing H=3 trained Mamba
    covariance mapping = sigma_min 3, sigma_max 20, gamma 3

Sweep:
    alpha = 0.50, 0.60, 0.70, 0.80, 0.90, 1.00

Meaning:
    R_final =
        alpha * R_pred
        +
        (1-alpha) * R_feedback
"""

import os
import subprocess
import sys


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
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


def main():
    print("=" * 80)
    print(
        "DFG PREDICTION / FEEDBACK "
        "WEIGHT SENSITIVITY"
    )
    print("=" * 80)

    for i, alpha in enumerate(
        ALPHAS,
        start=1,
    ):
        print()
        print(
            "#" * 80
        )

        print(
            f"[{i}/{len(ALPHAS)}] "
            f"alpha = {alpha:.2f}, "
            f"feedback = "
            f"{1.0-alpha:.2f}"
        )

        print(
            "#" * 80
        )

        subprocess.run(
            [
                sys.executable,
                os.path.join(
                    ROOT,
                    "experiments",
                    "factor_graph",
                    "run_weight_fg.py",
                ),
                "--alpha",
                str(
                    alpha
                ),
            ],
            cwd=ROOT,
            check=True,
        )

    print()
    print(
        "#" * 80
    )

    print(
        "FINAL WEIGHT SENSITIVITY "
        "EVALUATION"
    )

    print(
        "#" * 80
    )

    subprocess.run(
        [
            sys.executable,
            os.path.join(
                ROOT,
                "experiments",
                "evaluation",
                "evaluate_weight_sensitivity.py",
            ),
        ],
        cwd=ROOT,
        check=True,
    )

    print()
    print("=" * 80)

    print(
        "Weight sensitivity finished."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
