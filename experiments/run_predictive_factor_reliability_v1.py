import os
import subprocess
import sys


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


STEPS = [
    (
        "Build factor-reliability training labels",
        "experiments/reliability/"
        "build_predictive_factor_reliability_data.py",
    ),

    (
        "Train Shared-Mamba on future factor reliability",
        "experiments/reliability/"
        "train_predictive_factor_reliability_v1.py",
    ),

    (
        "Strict test-only reliability evaluation",
        "experiments/reliability/"
        "evaluate_predictive_factor_reliability_v1.py",
    ),

    (
        "Run predicted factor-reliability FG",
        "experiments/factor_graph/"
        "run_predicted_factor_reliability_fg.py",
    ),

    (
        "Evaluate Fixed vs old predictive vs learned vs Oracle",
        "experiments/evaluation/"
        "evaluate_predictive_factor_reliability_fg.py",
    ),
]


def main():
    print("=" * 116)
    print(
        "DFG PREDICTIVE FACTOR RELIABILITY V1 PIPELINE"
    )
    print("=" * 116)

    for i, (
        title,
        relative_path,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print()
        print(
            "#" * 116
        )

        print(
            f"[{i}/{len(STEPS)}] "
            f"{title}"
        )

        print(
            "#" * 116
        )

        subprocess.run(
            [
                sys.executable,
                os.path.join(
                    ROOT,
                    relative_path,
                ),
            ],
            cwd=ROOT,
            check=True,
        )

    print()
    print(
        "Predictive Factor Reliability V1 pipeline finished."
    )


if __name__ == "__main__":
    main()
