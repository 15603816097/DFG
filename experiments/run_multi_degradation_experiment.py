"""
Complete Multi-Degradation Robustness Experiment
================================================

All algorithm parameters are frozen.

Pipeline:
1. Generate six degradation scenarios.
2. Run the already-trained H=3 Mamba on each scenario.
3. Run Fixed / RobustFixed / Proposed / Oracle FG.
4. Evaluate all scenarios.

No model retraining.
No parameter tuning.
"""

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
        "Generate degradation scenarios",
        os.path.join(
            ROOT,
            "experiments",
            "degradation",
            "create_multi_degradation_scenarios.py",
        ),
    ),

    (
        "Frozen H=3 Mamba inference",
        os.path.join(
            ROOT,
            "experiments",
            "reliability",
            "infer_frozen_h3_reliability.py",
        ),
    ),

    (
        "Run factor graphs",
        os.path.join(
            ROOT,
            "experiments",
            "factor_graph",
            "run_multi_scenario_fg.py",
        ),
    ),

    (
        "Evaluate robustness",
        os.path.join(
            ROOT,
            "experiments",
            "evaluation",
            "evaluate_multi_degradation.py",
        ),
    ),
]


def main():
    print("=" * 96)
    print(
        "DFG MULTI-DEGRADATION "
        "ROBUSTNESS EXPERIMENT"
    )
    print("=" * 96)

    print(
        "Frozen parameters:"
    )

    print(
        "H=3, alpha=1.0, "
        "sigma_min=3, sigma_max=30, gamma=3"
    )

    print(
        "Mamba is NOT retrained."
    )

    for i, (
        title,
        path,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print()
        print(
            "#" * 96
        )

        print(
            f"[{i}/{len(STEPS)}] "
            f"{title}"
        )

        print(
            "#" * 96
        )

        subprocess.run(
            [
                sys.executable,
                path,
            ],
            cwd=ROOT,
            check=True,
        )

    print()
    print("=" * 96)
    print(
        "Multi-degradation robustness "
        "experiment finished."
    )
    print(
        "See:"
    )
    print(
        os.path.join(
            ROOT,
            "results",
            "multi_degradation_summary.txt",
        )
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
