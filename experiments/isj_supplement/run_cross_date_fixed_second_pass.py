from __future__ import annotations

"""Cross-date frozen FIXED-noise second pass and residual-baseline audit.

IMPORTANT: Existing cross-date residual baselines both produced ATE3D 2.779791
on 0016, versus Stage-1 fixed 2.811762. A second optimization alone may
explain the apparent improvement. This script isolates that confound by
running EXACTLY the original fixed graph from its own fixed Stage-1 poses.

No re-training, no new measurements, no U/r, and no reference positions used
for initialization or optimization. Each new output has a unique name and
old model/factor/pose files are NEVER overwritten. 0016 remains exploratory.

Stages:
    audit    Print saved residual sigma inflation stats (read-only)
    fgo      Run an additional fixed-noise GTSAM pass from saved Fixed Stage 1
    converge Continue the saved fixed2 graph under the uniform stopping rule
    compare  Compare fixed1, fixed2, one-step residual baselines and convergence
"""

import argparse
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.isj_supplement.run_cross_date import (
    RESULT_ROOT, crossdate_bindings, date_for, require_processed,
    result_is_complete, validate_destination,
)
from experiments.isj_supplement.run_cross_date_controls import (
    cached_continuation,
)
from experiments.isj_supplement.run_cross_date_beta_swap import (
    save_protocol,
)
from experiments.isj_supplement.run_cross_date_residual_baseline import (
    CASES,
)

FIXED_NAME = "ISJ_CTRL_CROSSDATE_FIXED_SECOND_PASS"
FIXED_SPEC = {"name": FIXED_NAME, "type": "fixed"}
WINDOW = 64


def load_saved_ate(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("final_metrics", data.get("metrics", data))
    val = float(metrics["ATE3D"])
    if not math.isfinite(val) or val < 0:
        raise ValueError(f"Invalid ATE3D in {path}")
    return val


def residual_sigma_stats(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    stats = meta["sigma_statistics"]
    result = {}
    for sensor in ("gps", "imu", "camera"):
        fields = stats[sensor]
        normalized = {}
        for key in ("mean_sigma_multiplier", "max_sigma_multiplier",
                    "inflated_fraction"):
            value = float(fields[key])
            if not math.isfinite(value):
                raise ValueError(f"Invalid {sensor}/{key} in {path}")
            normalized[key] = value
        if not (1 - 1e-8 <= normalized["mean_sigma_multiplier"] <= 4 + 1e-8):
            raise ValueError(f"{sensor} mean multiplier out of expected range")
        if not (1 - 1e-8 <= normalized["max_sigma_multiplier"] <= 4 + 1e-8):
            raise ValueError(f"{sensor} max multiplier out of expected range")
        if not (0 <= normalized["inflated_fraction"] <= 1):
            raise ValueError(f"{sensor} inflated fraction out of range")
        result[sensor] = normalized
    return result


def fixed_output(out: Path, name: str) -> Path:
    return out / "factor_graph" / name / "metrics.json"


def preflight(out: Path) -> tuple[int, np.ndarray]:
    n = require_processed(out)
    if not result_is_complete(out, "FINAL_S0_fixed", n):
        raise FileNotFoundError(
            f"Stage-1 Fixed FGO missing: {out}/factor_graph/FINAL_S0_fixed"
        )
    poses = np.load(
        out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy",
        allow_pickle=False,
    )
    return n, poses


def run_fgo(seq: str, out: Path) -> None:
    import experiments.cross_sequence.run_stage7_factor_graph as s7

    n, fixed_poses = preflight(out)
    destination = out / "factor_graph" / FIXED_NAME
    protocol = out / "isj_cross_date_fixed2_protocol.json"
    expected = {
        "sequence": seq,
        "method": FIXED_SPEC,
        "original_graph": "fixed sigma from Stage-7, no sensor reweighting",
        "warm_start": "factor_graph/FINAL_S0_fixed/poses.npy",
        "classification": "exploratory_post_hoc",
        "reference_for_optimization": False,
    }

    if destination.exists():
        if not result_is_complete(out, FIXED_NAME, n):
            raise RuntimeError(f"Incomplete fixed2 output, refusing overwrite: {destination}")
        if not protocol.is_file() or json.loads(protocol.read_text()) != expected:
            raise RuntimeError("Existing fixed2 output has no matching protocol")
        status = "cached"
    else:
        def fixed_start(gtsam, measurements, initial_pose):
            if len(measurements.gps_local) != n:
                raise ValueError("Fixed stage shape mismatch")
            return [s7._pose3_from_matrix(gtsam, T) for T in fixed_poses]

        with patch.object(s7, "initial_trajectory_from_pose", fixed_start):
            s7.run_case(seq=seq, name=FIXED_NAME)
        if not result_is_complete(out, FIXED_NAME, n):
            raise RuntimeError("Fixed second stage did not save complete output")
        save_protocol(protocol, expected)
        status = "computed"

    first_ate = load_saved_ate(fixed_output(out, "FINAL_S0_fixed"))
    fixed2_ate = load_saved_ate(fixed_output(out, FIXED_NAME))
    print(f"Fixed Stage-1 : {first_ate:.9f} m")
    print(f"Fixed Stage-2 : {fixed2_ate:.9f} m ({status})")
    print(f"Change solely from additional fixed optimization: "
          f"{first_ate-fixed2_ate:+.9f} m")


def run_converge(seq: str, out: Path, max_extra_passes: int) -> None:
    import gtsam
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    from experiments.isj_supplement.run_uniform_convergence import refine_case

    n, _ = preflight(out)
    if not result_is_complete(out, FIXED_NAME, n):
        raise FileNotFoundError(
            f"First run --stage fgo to produce {FIXED_NAME}"
        )
    row = cached_continuation(out, FIXED_SPEC, n, max_extra_passes)
    if row is None:
        convergence_dir = out / "factor_graph" / f"ISJ_CONVERGENCE_{FIXED_NAME}"
        if convergence_dir.exists():
            raise RuntimeError(
                f"Partial/inconsistent convergence output: {convergence_dir}"
            )
        m = s7.load_measurements(seq)
        row = refine_case(
            gtsam, seq, m, FIXED_SPEC, max_extra_passes, 1e-7, 1e-3
        )
    print(
        f"Fixed Stage-2 continuation: "
        f"{row['source_ATE3D']:.9f} -> "
        f"{row['continued_ATE3D']:.9f} m; "
        f"stable={bool(row['stationary_under_rule'])}"
    )


def run_audit(seq: str, out: Path) -> None:
    n, _ = preflight(out)
    print("=" * 96)
    print(f"READ-ONLY BASELINE ADAPTATION CHECK | {seq}, n={n}")
    print("First-stage Fixed FGO ATE:",
          f"{load_saved_ate(fixed_output(out, 'FINAL_S0_fixed')):.9f} m")
    print("=" * 96)
    for mode, spec in CASES.items():
        path = fixed_output(out, spec["name"])
        if not path.is_file():
            print(f"{spec['name']}: NOT YET COMPUTED")
            continue
        ate = load_saved_ate(path)
        stats = residual_sigma_stats(path)
        print(f"\n{spec['name']} ATE={ate:.9f} m")
        for sensor, values in stats.items():
            print(
                f"  {sensor:7s}: inflated={values['inflated_fraction']*100:.3f}% "
                f"mean_sigma_mult={values['mean_sigma_multiplier']:.9f} "
                f"max_sigma_mult={values['max_sigma_multiplier']:.9f}"
            )
    print("\nInterpretation: identical inflation stats for IMU/Camera can "
          "explain identical GPS-only and all-sensor results.")
    print("An additional fixed Stage-2 pass is required to isolate "
          "optimization-only improvement.")


def run_compare(seq: str, out: Path) -> None:
    n, _ = preflight(out)
    print("=" * 96)
    print(f"READ-ONLY CROSS-DATE BASELINE COMPARISON | {seq}")
    print("=" * 96)
    cases = [
        ("Fixed Stage-1", "FINAL_S0_fixed"),
        ("Fixed Stage-2 (no reweighting)", FIXED_NAME),
        ("Residual GPS only", CASES["gps"]["name"]),
        ("Residual GPS+IMU+Camera", CASES["all"]["name"]),
    ]
    reference = None
    fixed2 = None
    for label, name in cases:
        file = fixed_output(out, name)
        if not file.exists():
            print(f"{label:<36} MISSING")
            continue
        ate = load_saved_ate(file)
        if name == FIXED_NAME:
            fixed2 = ate
        if name == "FINAL_S0_fixed":
            reference = ate
        print(f"{label:<36} ATE3D={ate:.9f} m")

    print("\nUniform convergence copies (not the first/second-stage raw values):")
    conv_fixed2 = None
    for label, name in cases[1:]:
        dest = fixed_output(out, f"ISJ_CONVERGENCE_{name}")
        if not dest.exists():
            print(f"{label:<36} MISSING")
            continue
        data = json.loads(dest.read_text(encoding="utf-8"))
        if "final_metrics" not in data:
            raise ValueError(f"Missing continuation metrics {dest}")
        ate = load_saved_ate(dest)
        stable = bool(data.get("stationary_under_rule"))
        if name == FIXED_NAME:
            conv_fixed2 = ate
        print(f"{label:<36} continued_ATE={ate:.9f} m, stable={stable}")

    if fixed2 is not None:
        print("\nReweighting improvement over the SAME-second-pass fixed anchor:")
        for label, name in cases[2:]:
            f = fixed_output(out, name)
            if f.is_file():
                diff = fixed2 - load_saved_ate(f)
                print(f"  {label}: {diff:+.9f} m")
    if reference is not None and fixed2 is not None:
        print(f"\nFixed first-to-second-pass-only improvement: "
              f"{reference-fixed2:+.9f} m")
    if conv_fixed2 is not None:
        print("Uniform continuation comparison must be done at comparable "
              "documented stability, not by mixing raw and continued ATE.")
    print("\nWARNING: 0016 is repeatedly inspected exploratory data; "
          "the OXTS/GPS reference is not independent.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    p.add_argument("--stage", required=True,
                   choices=("audit", "fgo", "converge", "compare"))
    p.add_argument("--max-extra-passes", type=int, default=4)
    a = p.parse_args()
    date_for(a.sequence)
    if a.max_extra_passes <= 0:
        p.error("--max-extra-passes must be positive")
    out = validate_destination(a.sequence, RESULT_ROOT)
    with crossdate_bindings(a.sequence, RESULT_ROOT,
                            "converge" if a.stage == "converge" else "fgo"):
        if a.stage == "audit":
            run_audit(a.sequence, out)
        elif a.stage == "fgo":
            run_fgo(a.sequence, out)
        elif a.stage == "converge":
            run_converge(a.sequence, out, a.max_extra_passes)
        else:
            run_compare(a.sequence, out)


if __name__ == "__main__":
    main()
