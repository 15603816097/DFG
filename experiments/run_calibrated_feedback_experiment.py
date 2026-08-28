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
        "Calibrated Predictive Only",
        "experiments/factor_graph/"
        "run_calibrated_predictive_only_fg.py",
    ),
    (
        "Calibrated Predictive + FG Feedback",
        "experiments/factor_graph/"
        "run_calibrated_predictive_feedback_fg.py",
    ),
    (
        "Evaluation",
        "experiments/evaluation/"
        "evaluate_calibrated_feedback.py",
    ),
]


def main():
    print("=" * 70)
    print(
        "CALIBRATED RELIABILITY + "
        "FG FEEDBACK EXPERIMENT"
    )
    print("=" * 70)

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
        "Experiment finished."
    )


if __name__ == "__main__":
    main()
