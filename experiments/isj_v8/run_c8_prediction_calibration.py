"""V8.5.3 single bounded Mamba prediction-conditioned calibration pilot.

Fit only in-sample predictions on original 0027 TRAIN; no validation fit.
Not cross-fitted, not a validated physical covariance model. Same five
equal-count bins, weighted monotone second moments, original floor/Cap4
and identical frozen solver policy for H0/P1/H1. No search or heldout.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from experiments.isj_v8.run_c8_weight_controls import (
    verify, shifts, COUNTS, PREFIX, N, H0, P1, H1, OUT as CONTROLS,
)
from experiments.isj_v8.evaluate_c8_refined_dev import (
    OUT as EVAL, PREVIOUS, metrics, PRIMARY,
)
from experiments.isj_v8.run_c8_numeric_replay import preflight
from experiments.isj_v8.backend_c8_numeric import (
    poses_to_values, build_c8_graphs, verify_objective, directional_audit,
)
from experiments.isj_v8.solver_health_v8 import solve_v8_health
from experiments.isj_v8.audit_forecast_dev import CASES, SEED
from experiments.isj_v8.causal_prior_dev import preobservation_candidates
from experiments.isj_v7.compare_dev_gps import read_candidate, read_run, sha256
from experiments.isj_v7.train_gps_dev import (
    load_dev_data, load_frozen_v7_gps_checkpoint,
    predict_frozen_gps_features_causally,
)
from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, END, DEGRADED, AXIS_STD_FLOOR_M, subgraph_measurements,
    cap_axis_information, axis_sigma_from_prediction,
)
from experiments.isj_v7.cross_sequence_v7 import _positions
from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph

OUT = EVAL.parent / "dev_0027_c8_prediction_calibration_v1"
CONTROL_HASH = "18330f105f22e07f7a8a208dc5411d4e45ac05d4fae7eebb6a55ea079dd2339d"
SCHEDULES = (H0, P1, H1)
TRAIN_START, TRAIN_END = 65, 3178


def verify_all():
    verify()
    if sha256(CONTROLS / "report.json") != CONTROL_HASH:
        raise ValueError("weight control report hash mismatch")


def monotone_decreasing(y, weights):
    """Weighted PAVA for nonincreasing second moments."""
    blocks = []
    for i, (v, w) in enumerate(zip(y, weights)):
        blocks.append([i, i+1, float(v*w), float(w)])
        while len(blocks) > 1:
            a, b = blocks[-2:]
            if a[2]/a[3] >= b[2]/b[3]:
                break
            blocks[-2:] = [[a[0], b[1], a[2]+b[2], a[3]+b[3]]]
    out = np.empty(len(y))
    for i, j, s, w in blocks:
        out[i:j] = s/w
    return out


def fit_prediction_bins(score, error):
    score, error = np.asarray(score, float), np.asarray(error, float)
    if (score.shape != (TRAIN_END-TRAIN_START,) or error.shape != (len(score), 3)
            or not np.isfinite(score).all() or not np.isfinite(error).all()
            or np.any((score < 0) | (score > 1))):
        raise ValueError("invalid train calibration data")
    groups = np.array_split(np.argsort(score, kind="stable"), 5)
    centers = np.array([score[g].mean() for g in groups])
    if not np.all(np.diff(centers) > 0):
        raise ValueError("degenerate calibration score centers")
    counts = np.array([len(g) for g in groups])
    second = np.stack([np.mean(error[g]**2, axis=0) for g in groups])
    fitted = np.column_stack([monotone_decreasing(second[:, a], counts)
                              for a in range(3)])
    bins = np.maximum(np.sqrt(fitted), AXIS_STD_FLOOR_M)
    return dict(centers=centers, sigma_bins_xyz=bins, counts=counts,
                raw_second_moment=second)


def main():
    import gtsam
    import importlib.metadata
    if importlib.metadata.version("gtsam") != "4.2":
        raise ValueError("GTSAM version drift")
    if OUT.exists():
        raise FileExistsError(f"refuse overwrite: {OUT}")
    verify_all()
    original = preflight()
    fixed = original["axis_fixed"][0]
    data, _ = load_dev_data()
    model, med, iqr, provenance = load_frozen_v7_gps_checkpoint(
        "mamba_r", seed=SEED, device="cuda")
    if provenance["checkpoint_sha256"] != CASES["mamba_r"][4]:
        raise ValueError("frozen Mamba checkpoint hash mismatch")
    ids, pred, _ = predict_frozen_gps_features_causally(
        model, data["gps_features"][:TRAIN_END], med, iqr,
        window=64, clip=provenance["input_clip"], batch_size=64)
    if not np.array_equal(ids, np.arange(63, TRAIN_END)):
        raise ValueError("training inference index mismatch")
    rows = np.arange(TRAIN_START, TRAIN_END)-63
    train_scores = {H0: pred[rows, 0], P1: pred[rows-1, 0], H1: pred[rows-1, 1]}
    with np.load(DEGRADED / "degraded_sensor_data.npz", allow_pickle=False) as z:
        train_error = (np.asarray(z["degraded_gps_local"][TRAIN_START:TRAIN_END], float)
                       - np.asarray(z["clean_gps_local"][TRAIN_START:TRAIN_END], float))
    fits = {c: fit_prediction_bins(train_scores[c], train_error) for c in SCHEDULES}
    # No validation targets are used in fitting; archived predictions only.
    candidate = read_candidate()
    root, backend, _, _, _ = CASES["mamba_r"]
    a = read_run(root, backend, SEED, candidate)["arrays"]
    _, streams = preobservation_candidates(a["anchors"], a["reliability"])
    val_scores = {H0: streams["post_observation_h0_t_diagnostic_only"],
                  P1: streams["persistence_h0_t_minus_1"],
                  H1: streams["forecast_h1_t_minus_1"]}
    calibrated = {}
    for c in SCHEDULES:
        r = np.full(N, .5)
        r[PREFIX:] = val_scores[c]
        sigma = cap_axis_information(axis_sigma_from_prediction(r, fits[c]), fixed)
        sigma[:PREFIX] = fixed
        calibrated[c] = sigma
        print("TRAIN_FIT", c, json.dumps({k:v.tolist() for k,v in fits[c].items()}), flush=True)
        print("CAP4_FRACTION", c, np.isclose((fixed/sigma[PREFIX:])**2, 4., atol=1e-8, rtol=0).mean(axis=0), flush=True)

    del model
    with np.load(PREVIOUS / "poses.npz", allow_pickle=False) as z:
        seed_poses = z["common_aligned_seed"].copy()
    if seed_poses.shape != (N,4,4) or not np.isfinite(seed_poses).all():
        raise ValueError("bad archived initialization")
    sub = subgraph_measurements(load_degraded_four_sensor_measurements(DEGRADED),
                               begin=BEGIN, end=END)
    gps = np.asarray(sub.gps_local, float)
    records, poses = {}, {}
    for c in SCHEDULES:
        print("CASE_START", c, flush=True)
        seed = poses_to_values(gtsam, seed_poses)
        legacy, _, counts = run_four_sensor_factor_graph(
            sub, OUT / "NO_GRAPH_WRITES", mode="fixed",
            gps_sigmas_override=calibrated[c], build_graph_only=True)
        if counts != COUNTS or legacy.size() != 2709:
            raise ValueError("factor count drift")
        native, corrected, keepalive, count = build_c8_graphs(
            gtsam, legacy, gps, calibrated[c])
        if count != 2028 or len(keepalive) != 2029:
            raise ValueError("wrapper count drift")
        verify_objective(native, corrected, seed)
        warm = solve_v8_health(native, seed, N)
        v = warm["values"]
        j0 = verify_objective(native, corrected, v)
        check0 = directional_audit(gtsam, native, corrected, v, N)
        if not check0["passed"]:
            raise RuntimeError("warm derivative check failed")
        middle = solve_v8_health(corrected, v, N)
        v = middle["values"]
        j1 = verify_objective(native, corrected, v)
        check1 = directional_audit(gtsam, native, corrected, v, N)
        if not check1["passed"] or j1 > j0 + 1e-8*max(j0,1.):
            raise RuntimeError("corrected continuation check failed")
        passes = []
        for _ in range(2):
            before = _positions(gtsam, v, N)
            jb = verify_objective(native, corrected, v)
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(100)
            params.setRelativeErrorTol(1e-12)
            params.setAbsoluteErrorTol(1e-10)
            v = gtsam.LevenbergMarquardtOptimizer(corrected, v, params).optimize()
            ja = verify_objective(native, corrected, v)
            after = _positions(gtsam, v, N)
            if ja > jb + 1e-8*max(jb,1.) or not np.isfinite(after).all():
                raise RuntimeError("invalid refinement")
            dt, dr = shifts(before, after)
            passes.append(dict(objective_before=jb, objective_after=ja,
                               relative_gain=(jb-ja)/max(jb,1.),
                               translation_shift_m=dt, rotation_shift_rad=dr))
        audit = directional_audit(gtsam, native, corrected, v, N)
        last = passes[-1]
        stable = bool(abs(last["relative_gain"]) <= 1e-6
                      and last["translation_shift_m"] <= .01
                      and last["rotation_shift_rad"] <= .01)
        if not audit["passed"]:
            raise RuntimeError("final derivative check failed")
        poses[c] = _positions(gtsam, v, N)
        records[c] = dict(native_status=warm["status"], corrected_status=middle["status"],
                          refinement=passes, derivative_audit=audit,
                          last_step_stable=stable, stationarity_certified=False)
        print("CASE_SOLVED", c, "stable", stable, flush=True)

    # Only now read DEV reference; no tuning/selection in this script.
    with np.load(DEGRADED / "degraded_sensor_data.npz", allow_pickle=False) as z:
        clean = np.asarray(z["clean_gps_local"][BEGIN:END], float)
        raw = np.asarray(z["degraded_gps_local"][BEGIN:END], float)
    error = raw[PREFIX:] - clean[PREFIX:]
    if error.shape != (N-PREFIX, 3) or not np.isfinite(error).all():
        raise ValueError("invalid DEV reference")
    large = np.linalg.norm(error, axis=1) >= np.quantile(np.linalg.norm(error, axis=1), .9)
    old = json.loads((EVAL / "report.json").read_text())["results"]
    def diagnostic(sigma):
        s = sigma[PREFIX:]
        # Zero-mean diagonal Gaussian score on raw synthetic error, not
        # graph likelihood; constants included, lower is better.
        nll = .5*np.sum((error/s)**2 + 2*np.log(s) + np.log(2*np.pi), axis=1)
        ratio = (fixed/s)**2
        return dict(mean_gaussian_nll=float(nll.mean()),
                    top_decile_mean_information_ratio_xyz=ratio[large].mean(axis=0).tolist(),
                    cap4_fraction_xyz=np.isclose(ratio,4.,atol=1e-8,rtol=0).mean(axis=0).tolist())
    for c in SCHEDULES:
        r = records[c]
        r["evaluation"] = metrics(poses[c][:,:3,3], clean)
        r["original_evaluation"] = old[c]
        r["original_measurement_diagnostic"] = diagnostic(original[c])
        r["calibrated_measurement_diagnostic"] = diagnostic(calibrated[c])
        print("RESULT", c, json.dumps(dict(
            original_rmse=old[c][PRIMARY], calibrated_rmse=r["evaluation"][PRIMARY],
            original_diagnostic=r["original_measurement_diagnostic"],
            calibrated_diagnostic=r["calibrated_measurement_diagnostic"],
            stable=r["last_step_stable"])), flush=True)
    delta = records[P1]["evaluation"][PRIMARY] - records[H1]["evaluation"][PRIMARY]
    print("CALIBRATED_H1_VS_P1_IMPROVEMENT_M", delta, flush=True)
    verify_all()
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(OUT / "poses.npz", **poses)
    np.savez_compressed(OUT / "sigmas.npz", **calibrated)
    report = dict(status="COMPLETE_0027_DEV_PREDICTION_CALIBRATION",
                  development_only=True, heldout_accessed=False,
                  independent_ground_truth=False, stationarity_certified=False,
                  train_in_sample_calibration=True, cross_fitted=False,
                  train_target_range=[TRAIN_START,TRAIN_END],
                  checkpoint=provenance, control_report_sha256=CONTROL_HASH,
                  fits={c:{k:v.tolist() for k,v in f.items()} for c,f in fits.items()},
                  cases=records, calibrated_h1_vs_p1_improvement_m=delta,
                  poses_sha256=sha256(OUT/"poses.npz"), sigmas_sha256=sha256(OUT/"sigmas.npz"))
    (OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("REPORT_SHA256", sha256(OUT/"report.json"))
    print("STABLE_CASES", sum(r["last_step_stable"] for r in records.values()), "/ 3")
    print("V8_5_3_CALIBRATION_COMPLETE 3_CASES NO_TRAINING NO_HELDOUT")


if __name__ == "__main__":
    main()

