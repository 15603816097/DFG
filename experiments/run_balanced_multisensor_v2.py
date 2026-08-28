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
        "Create balanced degradation plan V2",
        "experiments/multisensor/"
        "create_balanced_multisensor_degradation_v2.py",
    ),

    (
        "Build balanced reliability data V2",
        "experiments/multisensor/"
        "build_balanced_multisensor_reliability_data_v2.py",
    ),

    (
        "Check train/val/test balance",
        "experiments/multisensor/"
        "check_balanced_distribution_v2.py",
    ),

    (
        "Train Shared-Mamba on V2",
        "experiments/multisensor/"
        "train_multisensor_predictive_reliability_v2.py",
    ),

    (
        "Strict test-only evaluation",
        "experiments/multisensor/"
        "evaluate_balanced_multisensor_test_v2.py",
    ),
]


def main():
    print("=" * 100)
    print(
        "DFG BALANCED MULTI-SENSOR "
        "RELIABILITY V2 PIPELINE"
    )
    print("=" * 100)

    for i, (
        title,
        relative_path,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print()
        print(
            "#" * 100
        )

        print(
            f"[{i}/{len(STEPS)}] "
            f"{title}"
        )

        print(
            "#" * 100
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
    print("=" * 100)
    print(
        "Balanced Multi-Sensor V2 "
        "pipeline finished."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
