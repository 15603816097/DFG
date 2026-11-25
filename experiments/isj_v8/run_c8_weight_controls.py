"""V8.5.2 bounded post-hoc 0027 DEV control study.

Eight new GPS covariance controls, frozen C8 backend and V8.4.5 policy.
No training, no hyperparameter search, no heldout, no independent GT.
Constant/shuffled controls use the complete DEV weight stream and are
retrospective diagnostics, not causal deployable baselines.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.isj_v8.evaluate_c8_refined_dev import (
    verify_sources, SOURCE, PREVIOUS, OUT as EVAL, metrics, PRIMARY,
)
from experiments.isj_v8.run_c8_numeric_replay import preflight
from experiments.isj_v8.backend_c8_numeric import (
    poses_to_values, build_c8_graphs, verify_objective, directional_audit,
)
from experiments.isj_v8.solver_health_v8 import solve_v8_health
from experiments.isj_v8.audit_forecast_dev import CASES, SEED
from experiments.isj_v8.causal_prior_dev import preobservation_candidates
from experiments.isj_v7.compare_dev_gps import read_candidate, read_run, sha256
from experiments.isj_v7.supervision_gate import read_policy, validate_policy
from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, END, DEGRADED, subgraph_measurements,
    load_train_axis_noise, axis_sigma_from_prediction, cap_axis_information,
)
from experiments.isj_v7.cross_sequence_v7 import _positions
from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph

OUT = EVAL.parent / "dev_0027_c8_numeric_weight_controls_v1"
EVAL_HASH = "1afb4a168866b1890cd95885d59686bd7419e3b36141db72e46175fc8f17dc7d"
COUNTS = {"gps": 680, "imu": 679, "lidar": 679, "camera": 670}
PREFIX = 2
N = 680
MAMBA = "mamba_r__"
H0 = MAMBA + "h0_current_observation_available"
P1 = MAMBA + "h0_tminus1_persistence"
H1 = MAMBA + "h1_tminus1_causal_prior"
H2 = MAMBA + "h2_tminus2_causal_prior"
H12 = MAMBA + "h1h2_equal_causal_prior"


def verify():
    verify_sources()
    if sha256(EVAL / "report.json") != EVAL_HASH:
        raise ValueError("V8.5.0 report hash mismatch")


def build_controls(sigmas):
    # Joint XYZ permutation preserves each row and the full weight distribution.
    # Identical deterministic permutation for H0/P1/H1; no reference selection.
    perm = np.random.default_rng(20261125).permutation(N-PREFIX)
    out = {}
    for name, case in (("h0", H0), ("p1", P1), ("h1", H1)):
        s = sigmas[case]
        constant = s.copy()
        constant[PREFIX:] = 1. / np.sqrt(np.mean(1. / s[PREFIX:]**2, axis=0))
        shuffled = s.copy()
        shuffled[PREFIX:] = s[PREFIX:][perm]
        for a in (constant, shuffled):
            if not np.array_equal(a[:PREFIX], s[:PREFIX]):
                raise ValueError("startup differs")
            if not np.allclose(np.mean(1./a[PREFIX:]**2, axis=0),
                               np.mean(1./s[PREFIX:]**2, axis=0), rtol=1e-12, atol=0):
                raise ValueError("mean information differs")
        out[name + "_constant_mean_information"] = constant
        out[name + "_permuted_information"] = shuffled

    candidate = read_candidate()
    root, backend, _, _, expected = CASES["mamba_r"]
    if sha256(Path(root) / backend / f"seed_{SEED}" / "best.pt") != expected:
        raise ValueError("frozen checkpoint mismatch")
    a = read_run(root, backend, SEED, candidate)["arrays"]
    _, p = preobservation_candidates(a["anchors"], a["reliability"])
    policy = read_policy()
    validate_policy(policy)
    fitted = load_train_axis_noise(policy)
    fixed = sigmas["axis_fixed"][0]
    def mapped(tail):
        r = np.full(N, .5)
        r[PREFIX:] = tail
        s = cap_axis_information(axis_sigma_from_prediction(r, fitted), fixed)
        s[:PREFIX] = fixed
        return s
    # Verify mapping consistency for original schedules before new controls.
    for c, stream in ((P1, p["persistence_h0_t_minus_1"]),
                      (H1, p["forecast_h1_t_minus_1"]),
                      (H2, p["forecast_h2_t_minus_2"]),
                      (H12, p["forecast_h1h2_equal_mean"])):
        if not np.array_equal(mapped(stream), sigmas[c]):
            raise ValueError("existing mapping drift: " + c)
    out["p2_same_age_persistence"] = mapped(p["persistence_h0_t_minus_2"])
    out["p12_equal_persistence"] = mapped(
        .5 * (p["persistence_h0_t_minus_1"] + p["persistence_h0_t_minus_2"]))
    if len(out) != 8:
        raise ValueError("unexpected controls")
    return out


def shifts(a, b):
    t = float(np.sqrt(np.mean(np.sum((a[:, :3, 3]-b[:, :3, 3])**2, axis=1))))
    r = a[:, :3, :3].transpose(0, 2, 1) @ b[:, :3, :3]
    angle = np.arccos(np.clip((np.trace(r, axis1=1, axis2=2)-1)/2, -1, 1))
    return t, float(np.sqrt(np.mean(angle**2)))


def main():
    import gtsam
    import importlib.metadata
    if importlib.metadata.version("gtsam") != "4.2":
        raise ValueError("GTSAM version drift")
    if OUT.exists():
        raise FileExistsError(f"refuse overwrite: {OUT}")
    verify()
    sigmas = preflight()
    controls = build_controls(sigmas)
    old = json.loads((EVAL / "report.json").read_text())
    refined = json.loads((SOURCE / "report.json").read_text())
    if any(not (r["derivative_audit"]["passed"] and
                r["within_previous_motion_objective_tolerances"])
           for r in refined["cases"].values()):
        raise ValueError("refined stability failed")
    with np.load(PREVIOUS / "poses.npz", allow_pickle=False) as z:
        seed_poses = z["common_aligned_seed"].copy()
    if seed_poses.shape != (N, 4, 4) or not np.isfinite(seed_poses).all():
        raise ValueError("bad archived seed")
    seed = poses_to_values(gtsam, seed_poses)
    sub = subgraph_measurements(load_degraded_four_sensor_measurements(DEGRADED),
                               begin=BEGIN, end=END)
    gps = np.asarray(sub.gps_local, float)
    records, poses = {}, {}
    for case, sigma in controls.items():
        print("CASE_START", case, flush=True)
        legacy, _, counts = run_four_sensor_factor_graph(
            sub, OUT / "NO_GRAPH_WRITES", mode="fixed",
            gps_sigmas_override=sigma, build_graph_only=True)
        if counts != COUNTS or legacy.size() != 2709:
            raise ValueError("factor count drift")
        native, corrected, keepalive, count = build_c8_graphs(gtsam, legacy, gps, sigma)
        if count != 2028 or len(keepalive) != 2029:
            raise ValueError("wrapper count drift")
        verify_objective(native, corrected, seed)
        warm = solve_v8_health(native, seed, N)
        v0 = warm["values"]
        j0 = verify_objective(native, corrected, v0)
        audit0 = directional_audit(gtsam, native, corrected, v0, N)
        if not audit0["passed"]:
            raise RuntimeError(f"warm derivative audit failed: {case}")
        mid = solve_v8_health(corrected, v0, N)
        v = mid["values"]
        j1 = verify_objective(native, corrected, v)
        audit1 = directional_audit(gtsam, native, corrected, v, N)
        if not audit1["passed"] or j1 > j0 + 1e-8*max(j0, 1.):
            raise RuntimeError(f"corrected continuation check failed: {case}")
        passes = []
        for i in range(2):
            before = _positions(gtsam, v, N)
            jb = verify_objective(native, corrected, v)
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(100)
            params.setRelativeErrorTol(1e-12)
            params.setAbsoluteErrorTol(1e-10)
            v = gtsam.LevenbergMarquardtOptimizer(corrected, v, params).optimize()
            ja = verify_objective(native, corrected, v)
            after = _positions(gtsam, v, N)
            if not np.isfinite(after).all() or ja > jb + 1e-8*max(jb, 1.):
                raise RuntimeError(f"invalid refinement: {case}")
            dt, dr = shifts(before, after)
            passes.append(dict(objective_before=jb, objective_after=ja,
                               relative_gain=(jb-ja)/max(jb, 1.),
                               translation_shift_m=dt, rotation_shift_rad=dr))
        audit = directional_audit(gtsam, native, corrected, v, N)
        last = passes[-1]
        stable = bool(abs(last["relative_gain"]) <= 1e-6
                      and last["translation_shift_m"] <= .01
                      and last["rotation_shift_rad"] <= .01)
        if not audit["passed"]:
            raise RuntimeError(f"terminal derivative audit failed: {case}")
        poses[case] = _positions(gtsam, v, N)
        records[case] = dict(native_status=warm["status"], corrected_status=mid["status"],
                             refinement=passes, final_gradient=audit["gradient_l2"],
                             derivative_audit=audit, last_step_stable=stable,
                             stationarity_certified=False)
        print("CASE_SOLVED", case, "gradient", audit["gradient_l2"], "stable", stable, flush=True)

    # Only after all new trajectories are solved and audited, access DEV reference.
    with np.load(DEGRADED / "degraded_sensor_data.npz", allow_pickle=False) as z:
        clean = np.asarray(z["clean_gps_local"][BEGIN:END], float)
        degraded = np.asarray(z["degraded_gps_local"][BEGIN:END], float)
    if clean.shape != (N, 3) or degraded.shape != (N, 3):
        raise ValueError("invalid reference")
    comparisons = []
    for case, a in poses.items():
        records[case]["evaluation"] = metrics(a[:, :3, 3], clean)
        print("CONTROL_RMSE", case, records[case]["evaluation"][PRIMARY], flush=True)
    for name, original in (("h0", H0), ("p1", P1), ("h1", H1)):
        for control in (name+"_constant_mean_information", name+"_permuted_information"):
            a = old["results"][original][PRIMARY]
            b = records[control]["evaluation"][PRIMARY]
            row = dict(candidate=original, baseline=control, candidate_m=a, baseline_m=b,
                       improvement_m=b-a, improvement_pct=100*(b-a)/b if b else None)
            comparisons.append(row)
            print("COMPARE", json.dumps(row), flush=True)
    for original, control in ((H2, "p2_same_age_persistence"), (H12, "p12_equal_persistence")):
        a = old["results"][original][PRIMARY]
        b = records[control]["evaluation"][PRIMARY]
        row = dict(candidate=original, baseline=control, candidate_m=a, baseline_m=b,
                   improvement_m=b-a, improvement_pct=100*(b-a)/b if b else None)
        comparisons.append(row)
        print("COMPARE", json.dumps(row), flush=True)

    # Descriptive measurement-domain diagnostic, not a calibrated likelihood:
    # use raw degraded-minus-clean error (not the re-zeroed trajectory).
    error = degraded[PREFIX:] - clean[PREFIX:]
    magnitude = np.linalg.norm(error, axis=1)
    large = magnitude >= np.quantile(magnitude, .9)
    fixed = sigmas["axis_fixed"][0]
    quality = {}
    for case, s in {k: sigmas[k] for k in (H0, P1, H1, H2, H12)}.items():
        ratio = (fixed / s[PREFIX:])**2
        quality[case] = dict(
            top_decile_error_count=int(large.sum()),
            mean_information_ratio_xyz=ratio.mean(axis=0).tolist(),
            large_error_information_ratio_xyz=ratio[large].mean(axis=0).tolist(),
            large_error_cap4_fraction_xyz=np.isclose(ratio[large], 4., atol=1e-8, rtol=0).mean(axis=0).tolist())
        print("LARGE_ERROR_WEIGHT", case, json.dumps(quality[case]), flush=True)

    verify()
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUT / "poses.npz", **poses)
    np.savez_compressed(OUT / "sigmas.npz", **controls)
    payload = dict(status="COMPLETE_0027_DEV_WEIGHT_CONTROLS",
                   evaluation_sha256=EVAL_HASH, development_only=True,
                   heldout_accessed=False, independent_ground_truth=False,
                   stationarity_certified=False, retrospective_full_window_controls=True,
                   cases=records, comparisons=comparisons, large_error_weight=quality,
                   poses_sha256=sha256(OUT / "poses.npz"),
                   sigmas_sha256=sha256(OUT / "sigmas.npz"))
    (OUT / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print("STABLE_CASES", sum(r["last_step_stable"] for r in records.values()), "/ 8")
    print("REPORT_SHA256", sha256(OUT / "report.json"))
    print("V8_5_2_CONTROLS_COMPLETE 8_CASES DEV_ONLY NO_TRAINING NO_HELDOUT")


if __name__ == "__main__":
    main()
