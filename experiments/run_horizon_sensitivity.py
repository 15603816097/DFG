"""
Complete Horizon Sensitivity Experiment.

Runs:
    H = 1, 3, 5, 10, 20

For each H:
    1. train dual-head Mamba
    2. create target-aligned predictive prior
    3. run predictive + feedback factor graph

Finally:
    evaluate all horizons
"""

import os
import subprocess
import sys


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


HORIZONS = [
    1,
    3,
    5,
    10,
    20,
]


def run(
    relative_path,
    horizon=None,
):
    command = [
        sys.executable,
        os.path.join(
            ROOT,
            relative_path,
        ),
    ]

    if horizon is not None:
        command.extend(
            [
                "--horizon",
                str(
                    horizon
                ),
            ]
        )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():
    print("=" * 80)
    print(
        "DFG HORIZON SENSITIVITY EXPERIMENT"
    )
    print("=" * 80)

    for i, horizon in enumerate(
        HORIZONS,
        start=1,
    ):
        print()
        print(
            "#" * 80
        )

        print(
            f"HORIZON {horizon} "
            f"({horizon * 0.1:.1f} s)"
        )

        print(
            f"[{i}/{len(HORIZONS)}]"
        )

        print(
            "#" * 80
        )

        print()
        print(
            "Step A: Training"
        )

        run(
            "experiments/reliability/"
            "train_horizon_model.py",
            horizon=horizon,
        )

        print()
        print(
            "Step B: Factor Graph"
        )

        run(
            "experiments/factor_graph/"
            "run_horizon_fg.py",
            horizon=horizon,
        )

    print()
    print(
        "#" * 80
    )
    print(
        "FINAL EVALUATION"
    )
    print(
        "#" * 80
    )

    run(
        "experiments/evaluation/"
        "evaluate_horizon_sensitivity.py"
    )

    print()
    print("=" * 80)
    print(
        "Horizon sensitivity finished."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
