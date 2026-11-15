from __future__ import annotations

"""Summarize actual Stage-7 original and ISJ-control CSVs without rerunning FGO.

The output uses exact machine-recorded metrics rather than numbers rounded in
the manuscript. Differences are descriptive, not inferential statistics.
"""

import argparse
import csv
import math
from pathlib import Path

from experiments.cross_sequence.stage7_config import TEST_SEQUENCES, OUT_ROOT

MODELS = {
    "GRU": ("S1_v5_gru", "V5_GRU"),
    "Mamba": ("S2_v6_all_mamba", "V6_MAMBA"),
}
SEEDS = (11, 23, 47)


def load_rows(path: Path, key: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot summarize; missing {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    out = {}
    for row in rows:
        name = row.get(key)
        if not name:
            raise ValueError(f"CSV {path} has no '{key}' value")
        if name in out:
            raise ValueError(f"Duplicate {key}={name} in {path}")
        out[name] = row
    return out


def ate(row: dict) -> float:
    x = float(row["ATE3D"])
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"Invalid ATE3D: {row}")
    return x


def pct_drop(before: float, after: float) -> float:
    return 100.0 * (before - after) / before


def sample_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((x - avg) ** 2 for x in vals) / (len(vals) - 1))


def summarize_sequence(seq: str) -> list[dict]:
    directory = OUT_ROOT / seq
    original = load_rows(directory / "stage7_two_stage_results.csv", "case")
    controls = load_rows(directory / "isj_controls_results.csv", "case")
    ablation_path = directory / "stage7_uncertainty_ablation.csv"
    ablation = load_rows(ablation_path, "method") if ablation_path.is_file() else {}

    s0 = ate(original["S0_fixed"])
    fixed2 = ate(controls["ISJ_CTRL_FIXED_SECOND_PASS"])
    output = []
    for model, (case, ablation_key) in MODELS.items():
        full = ate(original[case])
        mean = ate(controls[f"ISJ_CTRL_{model.upper()}_MEAN_U"])
        shuffled = [
            ate(controls[f"ISJ_CTRL_{model.upper()}_PERM_U_SEED_{seed}"])
            for seed in SEEDS
        ]
        sh_mean = sum(shuffled) / len(shuffled)
        reliability_only = ""
        improvement_vs_rel = ""
        if ablation_key in ablation:
            reliability_only = float(ablation[ablation_key]["reliability_only_ATE3D"])
            improvement_vs_rel = pct_drop(reliability_only, full)
        output.append({
            "sequence": seq,
            "encoder": model,
            "fixed_stage1_ATE3D": s0,
            "fixed_stage2_ATE3D": fixed2,
            "fixed2_improvement_vs_fixed1_pct": pct_drop(s0, fixed2),
            "reliability_only_ATE3D": reliability_only,
            "original_dynamic_ATE3D": full,
            "constant_mean_U_ATE3D": mean,
            "permuted_U_mean_ATE3D": sh_mean,
            "permuted_U_sample_std_ATE3D": sample_std(shuffled),
            "permuted_U_min_ATE3D": min(shuffled),
            "permuted_U_max_ATE3D": max(shuffled),
            "dynamic_vs_fixed2_pct": pct_drop(fixed2, full),
            "dynamic_vs_constant_U_pct": pct_drop(mean, full),
            "dynamic_vs_permuted_U_mean_pct": pct_drop(sh_mean, full),
            "dynamic_vs_reliability_only_pct": improvement_vs_rel,
            "dynamic_better_than_all_3_permutations": full < min(shuffled),
            "dynamic_better_than_constant_U": full < mean,
            "protocol": "post_hoc_exploratory",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUT_ROOT / "isj_supplement_summary.csv"))
    args = parser.parse_args()
    all_rows = []
    for seq in TEST_SEQUENCES:
        all_rows.extend(summarize_sequence(seq))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print("ISJ exploratory supplement: Stage-7 exact CSV comparison")
    print(f"Saved: {destination}")
    print(f"{'Sequence':8} {'Model':6} {'Fixed1':>8} {'Fixed2':>8}"
          f" {'Dynamic':>8} {'Mean U':>8} {'Perm mean':>9}"
          f" {'vs Mean':>8} {'vs Perm':>8}")
    for r in all_rows:
        seq = r["sequence"].split("_drive_")[1][:4]
        print(f"{seq:8} {r['encoder']:6} "
              f"{r['fixed_stage1_ATE3D']:8.4f} "
              f"{r['fixed_stage2_ATE3D']:8.4f} "
              f"{r['original_dynamic_ATE3D']:8.4f} "
              f"{r['constant_mean_U_ATE3D']:8.4f} "
              f"{r['permuted_U_mean_ATE3D']:9.4f} "
              f"{r['dynamic_vs_constant_U_pct']:+7.2f}% "
              f"{r['dynamic_vs_permuted_U_mean_pct']:+7.2f}%")
    print("\nNOTES: 3 shuffled seeds are not independent training repeats.")
    print("The experiments were defined post hoc after viewing the test results.")
    print("The OXTS/GPS reference is not independent of GPS measurements.")


if __name__ == "__main__":
    main()
