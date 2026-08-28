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
        "Build oracle factor reliability",
        "experiments/reliability/"
        "build_oracle_factor_reliability.py",
    ),
    (
        "Run oracle four-sensor FG",
        "experiments/factor_graph/"
        "run_oracle_four_sensor_fg.py",
    ),
    (
        "Evaluate fixed vs predictive vs oracle",
        "experiments/evaluation/"
        "evaluate_oracle_factor_reliability.py",
    ),
]


def main():
    print("=" * 112)
    print(
        "DFG ORACLE FACTOR RELIABILITY V1"
    )
    print("=" * 112)

    for i, (
        title,
        relative_path,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print()
        print(
            "#" * 112
        )

        print(
            f"[{i}/{len(STEPS)}] "
            f"{title}"
        )

        print(
            "#" * 112
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
        "Oracle factor-reliability experiment finished."
    )


if __name__ == "__main__":
    main()
