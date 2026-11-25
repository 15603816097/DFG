"""V8.4.3 fixed smoke and V8.4.4 uniform DEV-only C8 replay.

Same offline aligned seed and same native-warm/FD-continuation policy
for every covariance case. Never evaluate clean GPS or select by RMSE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.cross_sequence_v7 import _positions
from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, END, DEGRADED, subgraph_measurements,
)
from experiments.isj_v8.backend_c8_numeric import (
    build_c8_graphs, directional_audit, verify_objective, yaw_aligned_seed,
)
from experiments.isj_v8.diagnose_dev_local_solutions import load_sealed_source_report
from experiments.isj_v8.run_dev_feedforward_fgo import prepare_cases
from experiments.isj_v8.solver_health_v8 import solve_v8_health
from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/isj_v8"
OLD = BASE / "dev_0027_solver_health_replay_v1/v8_dev_solver_health_results.json"
OLD_HASH = "e2f27959069e2e9932384dde599212d4a8dfcf51f0d512d4c1d14beee8184afd"
COUNTS = {"gps": 680, "imu": 679, "lidar": 679, "camera": 670}
TOKEN = "V8_C8_NUMERIC_0027_DEV_ONLY"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    if sha(OLD) != OLD_HASH:
        raise ValueError("V8.3.3 archive hash mismatch")
    frozen, summary = load_sealed_source_report()
    meta, sigmas = prepare_cases()
    if (not summary["all_stable"] or len(sigmas) != 16 or "axis_fixed" not in sigmas
            or meta["all_cases_use_same_frame_range"] != [3180, 3860]
            or meta["heldout_0016_0020_accessed"] is not False):
        raise ValueError("frozen DEV inputs mismatch")
    for key in ("gps_degraded_sha256", "gps_candidate_sha256",
                "lidar_factors_sha256", "camera_factors_sha256"):
        if meta[key] != frozen[key]:
            raise ValueError(f"source mismatch: {key}")
    for case, arr in sigmas.items():
        a = np.asarray(arr, dtype=float)
        s = frozen["case_sigmas"][case]
        if (a.shape != (680, 3) or not np.isfinite(a).all() or np.any(a <= 0)
                or not np.isclose(a.min(), s["min_m"], atol=1e-8, rtol=0)
                or not np.isclose(a.max(), s["max_m"], atol=1e-8, rtol=0)
                or not np.allclose(a.mean(axis=0), s["mean_xyz_m"], atol=1e-8, rtol=0)):
            raise ValueError(f"frozen sigma mismatch: {case}")
    return sigmas


def summary_of(solved):
    return {k: solved[k] for k in (
        "status", "objective", "initial_objective", "initial_gradient_l2",
        "final_gradient_l2", "passes")}


def run(mode):
    import gtsam
    import importlib.metadata
    sigmas = preflight()
    if importlib.metadata.version("gtsam") != "4.2":
        raise ValueError("use GTSAM 4.2 as in V8.4.2")
    out = BASE / f"dev_0027_c8_numeric_{mode}_v1"
    if out.exists():
        raise FileExistsError(f"refuse overwrite: {out}")
    if mode == "all":
        smoke = BASE / "dev_0027_c8_numeric_fixed_v1/report.json"
        report = json.loads(smoke.read_text())
        if (report.get("status") != "COMPLETE"
                or report.get("source_sha256") != OLD_HASH
                or not report["cases"]["axis_fixed"]["engineering_candidate_and_derivative_check"]):
            raise ValueError("fixed smoke must pass before full replay")
    cases = ["axis_fixed"] if mode == "fixed" else list(sigmas)
    sub = subgraph_measurements(load_degraded_four_sensor_measurements(DEGRADED),
                               begin=BEGIN, end=END)
    n = len(sub.gps_local)
    if n != 680:
        raise ValueError("unexpected frame count")
    gps = np.asarray(sub.gps_local, float)
    records, pose_arrays = {}, {}
    common_poses, seed, alignment = None, None, None

    for case in cases:
        print("CASE_START", case, flush=True)
        legacy, initial, counts = run_four_sensor_factor_graph(
            sub, out / "NO_GRAPH_WRITES", mode="fixed",
            gps_sigmas_override=sigmas[case], build_graph_only=True)
        if counts != COUNTS or legacy.size() != 2709:
            raise ValueError("legacy factor count mismatch")
        poses = _positions(gtsam, initial, n)
        if common_poses is None:
            common_poses = poses.copy()
            seed, alignment = yaw_aligned_seed(gtsam, poses, gps)
            pose_arrays["common_aligned_seed"] = _positions(gtsam, seed, n)
            print("ALIGNMENT", json.dumps(alignment), flush=True)
        elif not np.array_equal(common_poses, poses):
            raise ValueError("common seed drift")
        native, corrected, keepalive, count = build_c8_graphs(
            gtsam, legacy, gps, sigmas[case])
        if count != 2028 or len(keepalive) != 2029:
            raise ValueError("corrected factor count mismatch")
        verify_objective(native, corrected, seed)
        warm = solve_v8_health(native, seed, n)
        v0 = warm["values"]
        before = verify_objective(native, corrected, v0)
        if case == "axis_fixed" and not np.isclose(
                before, 903.2550475676794, atol=1e-4, rtol=1e-6):
            raise ValueError("native fixed C8 reproduction mismatch")
        check0 = directional_audit(gtsam, native, corrected, v0, n)
        if not check0["passed"]:
            print("DERIVATIVE_AUDIT_FAILURE", case, json.dumps(check0), flush=True)
            raise RuntimeError("corrected warm-state derivative consistency failed")
        final = solve_v8_health(corrected, v0, n)
        v1 = final["values"]
        after = verify_objective(native, corrected, v1)
        check1 = directional_audit(gtsam, native, corrected, v1, n)
        if not check1["passed"]:
            print("DERIVATIVE_AUDIT_FAILURE", case, json.dumps(check1), flush=True)
            raise RuntimeError("corrected terminal derivative consistency failed")
        if after > before + 1e-8 * max(before, 1.):
            raise RuntimeError("same objective increased")
        qualified = final["status"] == "CONVERGED_CANDIDATE"
        row = {
            "native": summary_of(warm), "corrected": summary_of(final),
            "same_objective_before": before, "same_objective_after": after,
            "corrected_gradient_before": check0["gradient_l2"],
            "corrected_gradient_after": check1["gradient_l2"],
            "warm_derivative_audit": check0, "terminal_derivative_audit": check1,
            "engineering_candidate_and_derivative_check": qualified,
            "stationarity_certified": False,
        }
        records[case] = row
        pose_arrays[case + "__native"] = warm["poses"]
        pose_arrays[case + "__corrected"] = final["poses"]
        print("CASE_RESULT", case, json.dumps({
            "native_status": warm["status"], "corrected_status": final["status"],
            "objective_before": before, "objective_after": after,
            "gradient_before": check0["gradient_l2"],
            "gradient_after": check1["gradient_l2"],
            "derivative_audits_passed": True, "qualified": qualified,
        }), flush=True)

    if sha(OLD) != OLD_HASH:
        raise ValueError("original result changed")
    report = {
        "status": "COMPLETE", "mode": mode, "source_sha256": OLD_HASH,
        "development_only": True, "heldout_accessed": False,
        "clean_reference_evaluated": False, "independent_position_gt": False,
        "offline_batch_lm": True, "alignment": alignment,
        "selection_by_reference": False, "stationarity_certified": False,
        "cases": records,
    }
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "poses.npz", **pose_arrays)
    report["poses_sha256"] = sha(out / "poses.npz")
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("REPORT", path, sha(path), flush=True)
    print("TOTAL_CASES", len(records), flush=True)
    print("QUALIFIED_CASES", sum(r["engineering_candidate_and_derivative_check"]
                                for r in records.values()), flush=True)
    print("V8_C8_NUMERIC_REPLAY_COMPLETE", mode, "DEV_ONLY_NO_HELDOUT", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", choices=("fixed", "all"))
    p.add_argument("--token")
    args = p.parse_args()
    if args.preflight:
        if args.token:
            p.error("preflight is read-only")
        sigmas = preflight()
        print("V8_C8_NUMERIC_PREFLIGHT_OK", len(sigmas), "NO_SOLVES_NO_WRITES")
    else:
        if args.token != TOKEN:
            p.error("explicit DEV-only token required")
        run(args.run)


if __name__ == "__main__":
    main()
