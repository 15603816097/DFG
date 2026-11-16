from __future__ import annotations

"""Summarize *convergence-controlled* Stage-7 ISJ supplementary ATE3D.

Reads (rather than reruns or mutates) the original exact machine-written
isj_convergence_core_results.csv and isj_convergence_ablations_results.csv
for each frozen test sequence. This is exploratory/post-hoc evidence.

Crucially, fixed2 on drive 0034 can be NOT_STABLE under the strict 1-mm
trajectory-change criterion. The script preserves this warning rather than
silently describing all methods as converged.
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence.stage7_config import OUT_ROOT, TEST_SEQUENCES

MODELS = {
    "GRU": {
        "dynamic": "FINAL_S1_v5_gru",
        "reliability": "ABL_V5_GRU_RELIABILITY_ONLY",
        "mean": "ISJ_CTRL_GRU_MEAN_U",
        "permutation_prefix": "ISJ_CTRL_GRU_PERM_U_SEED_",
    },
    "Mamba": {
        "dynamic": "FINAL_S2_v6_all_mamba",
        "reliability": "ABL_V6_MAMBA_RELIABILITY_ONLY",
        "mean": "ISJ_CTRL_MAMBA_MEAN_U",
        "permutation_prefix": "ISJ_CTRL_MAMBA_PERM_U_SEED_",
    },
}
SEEDS = (11, 23, 47)
FIXED_CASE = "ISJ_CTRL_FIXED_SECOND_PASS"


def read_csv(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required convergence CSV missing: {path}")
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for line in csv.DictReader(stream):
            name = line.get("case")
            if not name:
                raise ValueError(f"Missing 'case' field in {path}")
            if name in rows:
                raise ValueError(f"Duplicate case {name!r} in {path}")
            rows[name] = line
    if not rows:
        raise ValueError(f"Empty results CSV: {path}")
    return rows


def get(rows: dict, case: str) -> tuple[float, bool]:
    if case not in rows:
        raise KeyError(f"Missing case {case!r}")
    row = rows[case]
    val = float(row["continued_ATE3D"])
    if not math.isfinite(val) or val <= 0:
        raise ValueError(f"Bad continued_ATE3D for {case}: {val}")
    flag = row["stationary_under_rule"].strip().lower()
    if flag not in ("true", "false"):
        raise ValueError(f"Invalid stationary status for {case}: {flag!r}")
    return val, flag == "true"


def gain(reference: float, dynamic: float) -> float:
    return 100.0 * (reference - dynamic) / reference


def summarize_sequence(directory: Path, seq: str) -> list[dict]:
    root = directory / seq
    core = read_csv(root / "isj_convergence_core_results.csv")
    ablations = read_csv(root / "isj_convergence_ablations_results.csv")
    fixed, fixed_stable = get(core, FIXED_CASE)
    rows = []
    for model, names in MODELS.items():
        dynamic, dynamic_stable = get(core, names["dynamic"])
        rel, rel_stable = get(ablations, names["reliability"])
        mean, mean_stable = get(ablations, names["mean"])
        pvals = [
            get(ablations, names["permutation_prefix"] + str(seed))
            for seed in SEEDS
        ]
        pmean = statistics.mean(x for x, _ in pvals)
        pstd = statistics.stdev(x for x, _ in pvals)
        all_stable = (fixed_stable and dynamic_stable and rel_stable
                      and mean_stable and all(flag for _, flag in pvals))
        rows.append({
            "sequence": seq,
            "model": model,
            "fixed2_ATE3D_m": fixed,
            "fixed2_stable": fixed_stable,
            "reliability_only_ATE3D_m": rel,
            "reliability_only_stable": rel_stable,
            "constant_U_ATE3D_m": mean,
            "constant_U_stable": mean_stable,
            "shuffled_U_3seed_mean_ATE3D_m": pmean,
            "shuffled_U_3seed_std_ATE3D_m": pstd,
            "shuffled_U_all_stable": all(flag for _, flag in pvals),
            "dynamic_U_ATE3D_m": dynamic,
            "dynamic_U_stable": dynamic_stable,
            "dynamic_improvement_vs_reliability_pct": gain(rel, dynamic),
            "dynamic_improvement_vs_constant_pct": gain(mean, dynamic),
            "dynamic_improvement_vs_shuffled_mean_pct": gain(pmean, dynamic),
            "dynamic_improvement_vs_fixed2_pct": gain(fixed, dynamic),
            "all_cases_stable_under_rule": all_stable,
            "experimental_status": "post_hoc_exploratory",
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=OUT_ROOT,
                   help="Directory holding the three sequence result folders")
    p.add_argument("--output", type=Path,
                   help="Output CSV; default root/isj_converged_summary.csv")
    args = p.parse_args()

    rows = []
    for seq in TEST_SEQUENCES:
        rows += summarize_sequence(args.root, seq)
    path = args.output or args.root / "isj_converged_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")
    print(f"{'SEQ':4} {'model':6} {'fixed2':>9} {'reliability':>12} "
          f"{'const-U':>9} {'shuffle-U':>10} {'dynamic-U':>10} "
          f"{'vs-rel':>8} {'status':>10}")
    for row in rows:
        print(f"{row['sequence'].split('_drive_')[1][:4]:4} "
              f"{row['model']:6} "
              f"{row['fixed2_ATE3D_m']:9.4f} "
              f"{row['reliability_only_ATE3D_m']:12.4f} "
              f"{row['constant_U_ATE3D_m']:9.4f} "
              f"{row['shuffled_U_3seed_mean_ATE3D_m']:10.4f} "
              f"{row['dynamic_U_ATE3D_m']:10.4f} "
              f"{row['dynamic_improvement_vs_reliability_pct']:+7.2f}% "
              f"{'STABLE' if row['all_cases_stable_under_rule'] else 'PARTIAL':>10}")
    print("\nWarning: SHUFFLED seeds are repeated permutations, NOT independent")
    print("training runs. The Stage-7 experiments and analysis are POST-HOC.")
    print("The OXTS/GPS-derived reference is not independent of GPS input.")
    if any(not r["all_cases_stable_under_rule"] for r in rows):
        print("PARTIAL = one or more cases did NOT meet the strict convergence")
        print("criterion; see *_stable columns. In particular check 0034 fixed2.")


if __name__ == "__main__":
    main()
