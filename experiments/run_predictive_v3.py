import os
import sys
import subprocess


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


STEPS = [
    (
        "Train dual-head predictive reliability",
        "experiments/reliability/"
        "train_predictive_reliability.py",
    ),
    (
        "Run temporally aligned predictive FG",
        "experiments/factor_graph/"
        "run_predictive_feedback_fg.py",
    ),
    (
        "Evaluate V3",
        "experiments/evaluation/"
        "evaluate_predictive_v3.py",
    ),
]


def main():
    print("=" * 72)
    print(
        "PREDICTIVE RELIABILITY V3 "
        "COMPLETE PIPELINE"
    )
    print("=" * 72)

    for i, (
        title,
        relative_path,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print()
        print(
            f"[{i}/{len(STEPS)}] "
            f"{title}"
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
        "Predictive Reliability V3 "
        "pipeline finished."
    )


if __name__ == "__main__":
    main()
