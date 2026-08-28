import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("Build factor-oriented V2 feature dataset", "experiments/reliability/build_factor_reliability_v2_data.py"),
    ("Train Shared-Mamba V2", "experiments/reliability/train_predictive_factor_reliability_v2.py"),
    ("Strict test-only V2 reliability evaluation", "experiments/reliability/evaluate_predictive_factor_reliability_v2.py"),
    ("Run V2 predictive reliability FG", "experiments/factor_graph/run_predicted_factor_reliability_v2_fg.py"),
    ("Evaluate V1 vs V2 vs Oracle", "experiments/evaluation/evaluate_predictive_factor_reliability_v2_fg.py"),
]

def main():
    print("=" * 118)
    print("DFG PREDICTIVE FACTOR RELIABILITY V2 PIPELINE")
    print("=" * 118)

    for i, (title, relative_path) in enumerate(STEPS, start=1):
        print()
        print("#" * 118)
        print(f"[{i}/{len(STEPS)}] {title}")
        print("#" * 118)

        subprocess.run(
            [sys.executable, os.path.join(ROOT, relative_path)],
            cwd=ROOT,
            check=True,
        )

    print()
    print("Predictive Factor Reliability V2 pipeline finished.")

if __name__ == "__main__":
    main()
