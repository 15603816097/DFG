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
        "Fixed four-sensor graph",
        "experiments/factor_graph/"
        "run_four_sensor_fixed_fg.py",
    ),
    (
        "Fixed sensor contribution ablation",
        "experiments/factor_graph/"
        "run_four_sensor_ablation.py",
    ),
    (
        "Predictive four-sensor graph",
        "experiments/factor_graph/"
        "run_four_sensor_predictive_fg.py",
    ),
    (
        "Evaluation",
        "experiments/evaluation/"
        "evaluate_four_sensor_fg.py",
    ),
]


def main():
    print("=" * 100)
    print(
        "DFG FOUR-SENSOR FACTOR GRAPH PIPELINE"
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
        "Four-sensor FG pipeline finished."
    )


if __name__ == "__main__":
    main()
