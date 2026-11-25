"""V8.5.0: retrospective DEV-only evaluation of frozen V8.4.5 poses.

No solver, no training, no covariance selection, no held-out drives.
OXTS-correlated synthetic-degradation reference, not independent GT.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, END, DEGRADED, reference_diagnostics,
)
from experiments.isj_v8.run_c8_numeric_replay import preflight

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/isj_v8"
SOURCE = BASE / "dev_0027_c8_numeric_refined_v1"
PREVIOUS = BASE / "dev_0027_c8_numeric_all_v1"
OUT = BASE / "dev_0027_c8_numeric_evaluation_v1"
HASHES = {
    SOURCE / "report.json": "33a26cbe829a57f2224e1698880e09676bb9b1b869386205299c6c081c8a022c",
    SOURCE / "poses.npz": "57abe5fdffea063254eed0c73f493ee960abbdcffaf94df0323e99ad0ba8891b",
    PREVIOUS / "report.json": "02de40a022ef3463d547f79d3709e1d560c0dff27122103c89a8b871caafc10d",
    PREVIOUS / "poses.npz": "52f91b0e6de7b7b1e4bab146d7e2ef762f0346b41bb0135f870420298fea9ca0",
}
MODELS = ("gru96_r", "gru106_r", "mamba_r")
METHODS = {
    "H0": "h0_current_observation_available",
    "P": "h0_tminus1_persistence",
    "H1": "h1_tminus1_causal_prior",
    "H2": "h2_tminus2_causal_prior",
    "H1H2": "h1h2_equal_causal_prior",
}
N = 680


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    for p, digest in HASHES.items():
        if sha(p) != digest:
            raise ValueError(f"archive SHA256 mismatch: {p}")


def key(model, method):
    return model + "__" + METHODS[method]


def metrics(trajectory, clean):
    result = reference_diagnostics(trajectory, clean)
    error = trajectory - (clean - clean[0])
    distance = np.linalg.norm(error, axis=1)
    result.update(
        mean_3d_m=float(distance.mean()),
        median_3d_m=float(np.median(distance)),
        p95_3d_m=float(np.percentile(distance, 95)),
        max_3d_m=float(distance.max()),
        vertical_rmse_m=float(np.sqrt(np.mean(error[:, 2] ** 2))),
    )
    return result


PRIMARY = "origin_aligned_oxTS_relative_3d_rmse_m"


def main():
    if OUT.exists():
        raise FileExistsError(f"refuse overwrite: {OUT}")
    verify_sources()
    sigmas = preflight()
    report = json.loads((SOURCE / "report.json").read_text())
    cases = ["axis_fixed"] + [key(m, h) for m in MODELS for h in METHODS]
    if (report["status"] != "COMPLETE" or report["development_only"] is not True
            or report["heldout_accessed"] is not False
            or report["clean_reference_evaluated"] is not False
            or report["source_report_sha256"] != HASHES[PREVIOUS / "report.json"]
            or report["source_poses_sha256"] != HASHES[PREVIOUS / "poses.npz"]
            or report["poses_sha256"] != HASHES[SOURCE / "poses.npz"]
            or set(cases) != set(report["cases"]) or set(cases) != set(sigmas)
            or END - BEGIN != N):
        raise ValueError("refined DEV archive provenance mismatch")
    for c in cases:
        row = report["cases"][c]
        if not (row["derivative_audit"]["passed"]
                and row["within_previous_motion_objective_tolerances"]):
            raise ValueError(f"numerical stability check failed: {c}")
    with np.load(SOURCE / "poses.npz", allow_pickle=False) as z:
        if set(z.files) != set(cases):
            raise ValueError("pose archive keys mismatch")
        trajectories = {}
        for c in cases:
            a = np.asarray(z[c], float)
            if a.shape != (N, 4, 4) or not np.isfinite(a).all():
                raise ValueError(f"invalid pose array: {c}")
            trajectories[c] = a[:, :3, 3].copy()
    # Reference opened only after all pose and provenance checks.
    with np.load(DEGRADED / "degraded_sensor_data.npz", allow_pickle=False) as z:
        clean = np.asarray(z["clean_gps_local"][BEGIN:END], float)
        degraded = np.asarray(z["degraded_gps_local"][BEGIN:END], float)
    if (clean.shape != (N, 3) or degraded.shape != (N, 3)
            or not np.isfinite(clean).all() or not np.isfinite(degraded).all()):
        raise ValueError("invalid frozen 0027 DEV reference")
    results = {"degraded_gps_only": metrics(degraded - degraded[0], clean)}
    for c in cases:
        results[c] = metrics(trajectories[c], clean)

    comparisons = []
    def compare(name, candidate, baseline):
        a, b = results[candidate][PRIMARY], results[baseline][PRIMARY]
        row = dict(comparison=name, candidate=candidate, baseline=baseline,
                   candidate_rmse_m=a, baseline_rmse_m=b,
                   improvement_m=b-a,
                   improvement_pct=100*(b-a)/b if b > 0 else None)
        comparisons.append(row)
        print("COMPARE", json.dumps(row), flush=True)

    for m in MODELS:
        for h in METHODS:
            compare("vs_axis_fixed", key(m, h), "axis_fixed")
        compare("one_step_vs_same_age_persistence", key(m, "H1"), key(m, "P"))
        for h in ("H2", "H1H2"):
            compare("vs_one_step_persistence_NOT_age_matched", key(m, h), key(m, "P"))
        for h in ("H1", "H2", "H1H2"):
            compare("vs_current_observation_NOT_same_information", key(m, h), key(m, "H0"))
    for h in METHODS:
        for m in ("gru96_r", "gru106_r"):
            compare("same_schedule_mamba_vs_gru", key("mamba_r", h), key(m, h))

    print("\nCASE RMSE3D RMSE2D DISP1 DISP10 P95", flush=True)
    for c, r in results.items():
        print(c, *(f"{r[k]:.6f}" for k in (
            PRIMARY, "origin_aligned_oxTS_relative_2d_rmse_m",
            "relative_displacement_1_frame_rmse_m",
            "relative_displacement_10_frame_rmse_m", "p95_3d_m")), flush=True)
    verify_sources()
    payload = dict(
        status="COMPLETE_0027_DEV_RETROSPECTIVE_ONLY",
        source_hashes={str(p.relative_to(ROOT)): h for p, h in HASHES.items()},
        frame_range=[BEGIN, END], primary_metric=PRIMARY,
        independent_ground_truth=False, heldout_accessed=False,
        offline_full_window_alignment=True, new_solver_runs=0,
        selection_or_tuning_performed=False,
        limitations=[
            "Correlated OXTS reference and synthetic degradation; not independent ATE/RPE.",
            "Single development window and frozen checkpoint/realization; descriptive only.",
            "H0 uses current GPS; H1 is age-matched to t-1 persistence.",
            "No age-matched t-2 or combined persistence in this archive.",
        ],
        results=results, comparisons=comparisons,
    )
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (OUT / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case"] + list(next(iter(results.values()))))
        writer.writeheader()
        for c, r in results.items():
            writer.writerow(dict(case=c, **r))
    with (OUT / "comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print("REPORT_SHA256", sha(OUT / "report.json"))
    print("TOTAL_FGO_CASES", len(cases))
    print("V8_5_0_DEV_EVALUATION_COMPLETE NO_NEW_SOLVES NO_HELDOUT NOT_INDEPENDENT")


if __name__ == "__main__":
    main()
