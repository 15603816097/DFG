from __future__ import annotations

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
        "Run GPS Z soft-dynamic covariance sweep",
        os.path.join(
            ROOT,
            "experiments",
            "factor_graph",
            "run_gps_z_soft_dynamic_sweep.py",
        ),
    ),
    (
        "Evaluate GPS Z soft-dynamic covariance sweep",
        os.path.join(
            ROOT,
            "experiments",
            "evaluation",
            "evaluate_gps_z_soft_dynamic_sweep.py",
        ),
    ),
]


def main():
    print("=" * 120)
    print(
        "DFG GPS Z SOFT-DYNAMIC COVARIANCE PIPELINE"
    )
    print("=" * 120)

    for index, (title, script) in enumerate(
        STEPS,
        start=1,
    ):
        print()
        print("#" * 120)
        print(
            f"[{index}/{len(STEPS)}] {title}"
        )
        print("#" * 120)

        subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            check=True,
        )

    print()
    print(
        "GPS Z soft-dynamic covariance pipeline finished."
    )


if __name__ == "__main__":
    main()
