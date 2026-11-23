from __future__ import annotations

"""Beta-valid permutation of frozen evidential uncertainty on cross-date KITTI.

Preserves every reliability r[t], exact U multiset per sensor, and every
warmup value t<63. Swaps uncertainty values only if BOTH exchanged
(r[t], U[t]) pairs still correspond to alpha>=1,beta>=1.

This controls for the obvious invalid-evidence confound of unconstrained
U-only permutation. It does NOT guarantee preservation of the full
(r,U) joint distribution or prove uncertainty calibration/causal effects.

Three disjoint phases: audit (no write), fgo (new cases only), converge
(separate continuations under the existing uniform LM stopping rule).
Never retrains a checkpoint or overwrites previous cross-date cases.
"""

import argparse
import csv
import json
import math
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.isj_supplement.run_cross_date import (
    RESULT_ROOT,
    MODELS,
    SENSORS,
    crossdate_bindings,
    date_for,
    validate_destination,
    result_is_complete,
)
from experiments.isj_supplement.run_cross_date_controls import (
    WINDOW,
    preflight,
    cached_continuation,
    parse_seeds,
)

# Pre-specified. Swaps are proposals rather than the desired number of
# accepted exchanges; accepted exchanges depend on evidence constraints.
PROPOSALS_PER_VALID_FRAME = 200
BETA_TOL = 1e-7
MIN_CHANGED_FRACTION = 0.05


def beta_invalid_fraction(reliability: np.ndarray,
                          uncertainty: np.ndarray,
                          tolerance: float = 1e-3) -> float:
    r = np.asarray(reliability, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    if r.shape != u.shape or r.size == 0:
        raise ValueError("Expected matching nonempty r and U arrays")
    bad = (~np.isfinite(r) | ~np.isfinite(u) |
           (r < 0.0) | (r > 1.0) | (u <= 0.0) |
           (2.0 * np.minimum(r, 1.0 - r) + tolerance < u))
    return float(np.mean(bad))


def beta_valid_swap_uncertainty(
    r: np.ndarray, u: np.ndarray, *, seed: int,
    source: str, sensor: str, window: int = WINDOW,
    proposals_per_frame: int = PROPOSALS_PER_VALID_FRAME,
) -> tuple[np.ndarray, dict]:
    """Permutation with *exact* U multiset and beta-validity at each timestamp.

    It is a local-swap randomization, not a uniform sample of all constrained
    permutations; accepted/moved stats must accompany results.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    start = window - 1
    if r.shape != u.shape or len(r) <= start:
        raise ValueError("r/U shape or warmup window mismatch")
    if proposals_per_frame <= 0:
        raise ValueError("proposals_per_frame must be positive")
    rv, uv = r[start:], u[start:]
    original_invalid = beta_invalid_fraction(rv, uv)
    if original_invalid > 0:
        raise ValueError(
            f"{source}/{sensor}: original frozen predictions violate beta "
            f"constraints at {100 * original_invalid:.3f}% of timestamps"
        )
    # For original evidential heads: alpha=2*r/U, beta=2*(1-r)/U.
    cap = 2.0 * np.minimum(rv, 1.0 - rv)
    scrambled = uv.copy()
    rng = np.random.default_rng(
        int(seed) + zlib.crc32(f"{source}/{sensor}".encode("utf-8"))
    )
    accepted = 0
    proposals = len(rv) * proposals_per_frame
    for _ in range(proposals):
        i, j = rng.integers(0, len(rv), size=2)
        if i == j:
            continue
        ui, uj = scrambled[i], scrambled[j]
        # Skip swaps that have no mathematical effect on the U series.
        if ui == uj:
            continue
        if uj <= cap[i] + BETA_TOL and ui <= cap[j] + BETA_TOL:
            scrambled[i], scrambled[j] = uj, ui
            accepted += 1

    transformed = u.copy()
    transformed[start:] = scrambled

    if not np.array_equal(transformed[:start], u[:start]):
        raise AssertionError("Warmup U values changed")
    if not np.array_equal(np.sort(scrambled), np.sort(uv)):
        raise AssertionError("Uncertainty empirical distribution changed")
    after_invalid = beta_invalid_fraction(rv, scrambled)
    if after_invalid != 0.0:
        raise AssertionError("Invalid beta evidence after valid swaps")
    changed = float(np.mean(scrambled != uv))
    if changed < MIN_CHANGED_FRACTION:
        raise RuntimeError(
            f"{source}/{sensor} seed {seed}: only {100 * changed:.2f}% U "
            "timestamps changed; do not call this a meaningful control."
        )

    summary = {
        "source": source,
        "sensor": sensor,
        "seed": seed,
        "frames": len(u),
        "warmup_frames": start,
        "valid_frames": len(rv),
        "proposals": proposals,
        "accepted_swaps": accepted,
        "changed_fraction": changed,
        "mean_absolute_delta_u": float(np.mean(np.abs(scrambled - uv))),
        "original_beta_invalid_fraction": original_invalid,
        "controlled_beta_invalid_fraction": after_invalid,
        "exact_U_multiset_preserved": True,
        "reliability_unchanged": True,
    }
    return transformed, summary


def specifications(models: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict]:
    specs = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")
        source = MODELS[model]["source"]
        label = "GRU" if model == "gru" else "MAMBA"
        for seed in seeds:
            specs.append({
                "name": f"ISJ_CTRL_{label}_BETA_SWAP_U_SEED_{seed}",
                "type": "dynamic",
                "source": source,
                "mode": "beta_valid_pair_swap",
                "seed": seed,
            })
    return specs


def control_predictions(original_pred, spec: dict, audit: list | None = None):
    def wrapped(seq, source, sensor, n):
        if source != spec["source"] or sensor not in SENSORS:
            raise ValueError(f"Unexpected prediction {source}/{sensor}")
        r, u = original_pred(seq, source, sensor, n)
        if len(u) != n or len(r) != n:
            raise ValueError("Frozen prediction length mismatch")
        swapped_u, stats = beta_valid_swap_uncertainty(
            r, u, seed=spec["seed"], source=source, sensor=sensor
        )
        if audit is not None:
            audit.append(stats)
        return r, swapped_u
    return wrapped


def num_frames_from_frozen(seq: str, s7) -> int:
    return len(s7.load_measurements(seq).gps_local)


def write_new_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No summary rows")
    # Reruns may revisit the same isolated summary. Never replace an existing
    # record with different content without an explicit code/protocol review.
    from io import StringIO
    buff = StringIO()
    writer = csv.DictWriter(buff, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    candidate = buff.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != candidate:
            raise RuntimeError(f"Refusing to replace existing summary: {path}")
        print("SUMMARY CACHED:", path)
        return
    path.write_text(candidate, encoding="utf-8")
    print("Saved:", path)


def save_protocol(path: Path, protocol: dict) -> None:
    candidate = json.dumps(protocol, indent=2, ensure_ascii=False)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError(f"Refusing to replace existing protocol: {path}")
        return
    path.write_text(candidate, encoding="utf-8")


def run_audit(seq: str, specs: list[dict]) -> None:
    import experiments.cross_sequence.run_stage7_factor_graph as s7

    n = num_frames_from_frozen(seq, s7)
    print(f"Audit: {seq} | frames={n} | valid_window_starts_at={WINDOW - 1}")
    for spec in specs:
        for sensor in SENSORS:
            r, u = s7.pred(seq, spec["source"], sensor, n)
            _, stats = beta_valid_swap_uncertainty(
                r, u, seed=spec["seed"],
                source=spec["source"], sensor=sensor
            )
            print(f'{spec["name"]:<43} {sensor:<7} '
                  f'changed={100 * stats["changed_fraction"]:.1f}% '
                  f'mean_dU={stats["mean_absolute_delta_u"]:.5f} '
                  f'accepted={stats["accepted_swaps"]} '
                  f'invalid={100 * stats["controlled_beta_invalid_fraction"]:.1f}%')
    print("BETA VALID SWAP AUDIT PASS")


def run_fgo(seq: str, out: Path, models: tuple[str, ...],
            specs: list[dict]) -> None:
    import experiments.cross_sequence.run_stage7_factor_graph as s7

    n, fixed_poses, fixed_ate = preflight(out, models)
    original_pred = s7.pred

    def fixed_start(gtsam, measurements, initial_pose):
        if len(measurements.gps_local) != n:
            raise ValueError("Fixed warm start length mismatch")
        return [s7._pose3_from_matrix(gtsam, pose) for pose in fixed_poses]

    rows = []
    with patch.object(s7, "initial_trajectory_from_pose", fixed_start):
        for spec in specs:
            name = spec["name"]
            output_dir = out / "factor_graph" / name
            if result_is_complete(out, name, n):
                status = "cached"
            else:
                if output_dir.exists():
                    raise RuntimeError(
                        f"Incomplete existing folder; refusing overwrite: {output_dir}"
                    )
                with patch.object(
                    s7, "pred", control_predictions(original_pred, spec)
                ):
                    s7.run_case(
                        seq=seq, name=name,
                        gps_src=spec["source"],
                        imu_src=spec["source"],
                        cam_src=spec["source"]
                    )
                status = "computed"

            metric = json.loads((output_dir / "metrics.json").read_text())
            ate = float(metric["ATE3D"])
            if not math.isfinite(ate):
                raise ValueError(f"{name}: invalid ATE")
            rows.append({
                "sequence": seq,
                "case": name,
                "source": spec["source"],
                "mode": spec["mode"],
                "seed": spec["seed"],
                "ATE3D_m": ate,
                "vs_fixed_pct": 100.0 * (fixed_ate - ate) / fixed_ate,
            })
            print(f"{name:<43} ATE3D={ate:.6f} m ({status})", flush=True)

    # It is possible to first run one model and later both; avoid replacing
    # the CSV in this case by distinguishing models and predeclared seeds.
    label = "_".join(models) + "_" + "_".join(
        str(seed) for seed in sorted({spec["seed"] for spec in specs})
    )
    path = out / f"isj_cross_date_beta_swap_fgo_{label}.csv"
    write_new_summary(path, rows)
    save_protocol(out / f"isj_cross_date_beta_swap_fgo_protocol_{label}.json", {
        "protocol": "frozen_beta_valid_uncertainty_swap_control",
        "sequence": seq,
        "model_set": list(models),
        "seeds": sorted({spec["seed"] for spec in specs}),
        "common_warm_start": "FINAL_S0_fixed",
        "reliability_modified": False,
        "U_multiset_exactly_preserved_per_sensor": True,
        "first_63_uncertainty_values_preserved": True,
        "numerical_constraint": "U <= 2 * min(r, 1-r)",
        "classification": "exploratory_post_hoc_non_blind",
        "reference_caveat": "OXTS/GPS reference not independent of GPS input",
    })


def run_converge(seq: str, out: Path, models: tuple[str, ...],
                 specs: list[dict], max_extra_passes: int) -> None:
    import gtsam
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    from experiments.isj_supplement.run_uniform_convergence import refine_case

    n, _, _ = preflight(out, models)
    for spec in specs:
        if not result_is_complete(out, spec["name"], n):
            raise FileNotFoundError(f"{spec['name']} FGO missing; run --stage fgo")
    original_pred = s7.pred
    m = s7.load_measurements(seq)
    rows = []
    for spec in specs:
        existing = cached_continuation(out, spec, n, max_extra_passes)
        if existing is None:
            dest = out / "factor_graph" / ("ISJ_CONVERGENCE_" + spec["name"])
            if dest.exists():
                raise RuntimeError(f"Incomplete convergence folder: {dest}")
            print(f"\n{seq} | {spec['name']}", flush=True)
            with patch.object(s7, "pred", control_predictions(original_pred, spec)):
                row = refine_case(gtsam, seq, m, spec, max_extra_passes, 1e-7, 1e-3)
        else:
            row = existing
        rows.append(row)
        print(f'  {spec["name"]:<43} '
              f'{row["source_ATE3D"]:.6f} -> {row["continued_ATE3D"]:.6f}m '
              f'{"STABLE" if row["stationary_under_rule"] else "NOT_STABLE"}',
              flush=True)
    label = "_".join(models) + "_" + "_".join(
        str(seed) for seed in sorted({spec["seed"] for spec in specs})
    )
    write_new_summary(
        out / f"isj_cross_date_beta_swap_convergence_{label}.csv", rows
    )
    save_protocol(
        out / f"isj_cross_date_beta_swap_convergence_protocol_{label}.json",
        {
            "protocol": "frozen_beta_valid_uncertainty_swap_convergence",
            "sequence": seq,
            "model_set": list(models),
            "seeds": sorted({spec["seed"] for spec in specs}),
            "max_extra_passes": max_extra_passes,
            "outer_objective_relative_tolerance": 1e-7,
            "outer_trajectory_shift_rmse_tolerance_m": 0.001,
            "all_stable": all(row["stationary_under_rule"] for row in rows),
            "reference_caveat": "OXTS/GPS reference not independent of GPS input",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    p.add_argument("--stage", required=True, choices=("audit", "fgo", "converge"))
    p.add_argument("--model", choices=("gru", "mamba", "both"), default="both")
    p.add_argument("--seeds", default="11,23,47")
    p.add_argument("--max-extra-passes", type=int, default=4)
    args = p.parse_args()

    date_for(args.sequence)
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        p.error(str(exc))
    if args.max_extra_passes < 1:
        p.error("--max-extra-passes must be positive")
    models = ("gru", "mamba") if args.model == "both" else (args.model,)
    specs = specifications(models, seeds)
    out = validate_destination(args.sequence, RESULT_ROOT)
    # The audit uses the standard FGO bindings for *reading* frozen predictions.
    with crossdate_bindings(args.sequence, RESULT_ROOT,
                            "converge" if args.stage == "converge" else "fgo"):
        if args.stage == "audit":
            run_audit(args.sequence, specs)
        elif args.stage == "fgo":
            run_fgo(args.sequence, out, models, specs)
        else:
            run_converge(seq=args.sequence, out=out, models=models,
                         specs=specs, max_extra_passes=args.max_extra_passes)


if __name__ == "__main__":
    main()
