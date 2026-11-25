from __future__ import annotations

"""V8.3.4 DEV-only read-only numerical factor audit, no optimizer calls.

Compare analytic linearized gradients to central differences of actual
factor.error() on the Pose3 retraction chart. Inspect frozen V8.3.3
states, not OXTS reference or independently held-out drives.
The legacy GPS pseudo-Pose3 prior is compared to a position GPSFactor
at the SAME saved poses. This is a diagnostic, not a method benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, END, DEGRADED, subgraph_measurements,
)
from experiments.isj_v8.diagnose_dev_local_solutions import (
    load_sealed_source_report, build_values_from_pose_matrices,
)
from experiments.isj_v8.run_dev_feedforward_fgo import prepare_cases
from experiments.isj_v8.run_dev_solver_health_replay import (
    METADATA, POSE_ARCHIVE, SEED_NAMES, CASES, FACTORS,
)
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph

REPORT_SHA = "e2f27959069e2e9932384dde599212d4a8dfcf51f0d512d4c1d14beee8184afd"
POSE_SHA = "126a65765a5aa9234eaff9e7ba6bcfd104e1ba1ad103292ca726a91fe9d72a8f"
TARGETS = (
    "axis_fixed",
    "mamba_r__h0_current_observation_available",
    "mamba_r__h1h2_equal_causal_prior",
)
EPS = (1e-5, 1e-6)
SAMPLE_COUNT = 16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def factor_gradient_check(gtsam, factor, poses):
    """Use only the factor's keys, avoiding full trajectory copies."""
    keys = [int(k) for k in factor.keys()]
    if len(keys) not in (1, 2):
        raise ValueError("unexpected factor arity")

    def values(delta_key=None, delta=None):
        state = gtsam.Values()
        for key in keys:
            pose = poses.atPose3(key)
            if key == delta_key:
                pose = pose.retract(delta)
            state.insert(key, pose)
        return state

    at = values()
    single = gtsam.NonlinearFactorGraph()
    single.add(factor)
    vector = single.linearize(at).gradientAtZero()
    analytic = np.concatenate([
        np.asarray(vector.at(k), dtype=float).reshape(6) for k in keys
    ])

    def central(step):
        out = []
        for key in keys:
            for axis in range(6):
                d = np.zeros(6)
                d[axis] = step
                plus = float(factor.error(values(key, d)))
                minus = float(factor.error(values(key, -d)))
                out.append((plus-minus)/(2*step))
        return np.array(out)

    fd1, fd2 = (central(e) for e in EPS)
    if not all(np.isfinite(a).all() for a in (analytic, fd1, fd2)):
        raise ValueError("nonfinite factor gradient")
    scale = max(float(np.linalg.norm(fd2)), 1e-6)
    return {
        "factor_error": float(factor.error(at)),
        "analytic_norm": float(np.linalg.norm(analytic)),
        "fd_norm": float(np.linalg.norm(fd2)),
        "abs_gradient_gap": float(np.linalg.norm(analytic-fd2)),
        "relative_gradient_gap": float(np.linalg.norm(analytic-fd2)/scale),
        "fd_step_disagreement": float(np.linalg.norm(fd1-fd2)/scale),
    }


def gps_factor(gtsam, key, gps, sigma):
    return gtsam.GPSFactor(
        int(key), np.asarray(gps, dtype=float),
        gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigma, dtype=float)),
    )


def selfcheck(gtsam):
    """Validate finite-difference signs and bindings on a known 3D factor."""
    state = gtsam.Values()
    state.insert(0, gtsam.Pose3(gtsam.Rot3.Rz(np.pi/2),
                               np.array([2., 3., 1.])))
    sigma = np.array([1., 2., 3.])
    gps = np.zeros(3)
    correct = gps_factor(gtsam, 0, gps, sigma)
    row = factor_gradient_check(gtsam, correct, state)
    if row["abs_gradient_gap"] > 1e-6 or row["fd_step_disagreement"] > 1e-5:
        raise RuntimeError("GPSFactor numerical-gradient selfcheck failed")
    legacy = gtsam.PriorFactorPose3(
        0, gtsam.Pose3(gtsam.Rot3(), gps),
        gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.full(3, 1e6), sigma]),
    )
    print("SYNTHETIC_GPS_GRADIENT_CHECK", json.dumps({
        "position_GPSFactor": row,
        "legacy_PriorFactorPose3": factor_gradient_check(gtsam, legacy, state),
    }), flush=True)


def summarize(rows):
    gap = np.array([r["relative_gradient_gap"] for r in rows])
    fd_gap = np.array([r["fd_step_disagreement"] for r in rows])
    return {
        "samples": len(rows),
        "median_relative_gradient_gap": float(np.median(gap)),
        "max_relative_gradient_gap": float(gap.max()),
        "max_abs_gradient_gap": max(r["abs_gradient_gap"] for r in rows),
        "max_fd_step_disagreement": float(fd_gap.max()),
        "worst": rows[int(np.argmax(gap))],
    }


def sample(items):
    if not items:
        raise ValueError("missing expected factor group")
    idx = np.linspace(0, len(items)-1, min(SAMPLE_COUNT, len(items)), dtype=int)
    return [items[i] for i in idx]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args(argv)
    import gtsam

    if sha(METADATA) != REPORT_SHA or sha(POSE_ARCHIVE) != POSE_SHA:
        raise ValueError("V8.3.3 frozen report or pose archive SHA mismatch")
    data = json.loads(METADATA.read_text(encoding="utf-8"))
    if (data["status"] != "COMPLETE_0027_DEV_NUMERICAL_HEALTH_REPLAY"
            or data["heldout_0016_0020_read"] is not False
            or set(data["case_results"]) != set(CASES)
            or data["pose_archive_sha256"] != POSE_SHA):
        raise ValueError("unexpected V8.3.3 provenance")
    selfcheck(gtsam)
    print("FROZEN_RESULT_SHA_VERIFIED", REPORT_SHA)
    print("FROZEN_POSE_SHA_VERIFIED", POSE_SHA)
    print("ALL_32_TERMINAL_STATES: case, seed, status, objective, gradient, last_step")
    for case in CASES:
        for seed in SEED_NAMES:
            row = data["case_results"][case]["initializers"][seed]
            print("TERMINAL", case, seed, row["status"],
                  row.get("objective"), row.get("terminal_gradient_l2"),
                  json.dumps(row.get("passes", [])[-1:]))

    frozen, _ = load_sealed_source_report()
    fresh, sigmas = prepare_cases()
    for key in ("gps_degraded_sha256", "gps_candidate_sha256",
                "lidar_factors_sha256", "camera_factors_sha256", "methods"):
        if fresh.get(key) != frozen.get(key):
            raise ValueError(f"changed frozen source: {key}")
    measurements = subgraph_measurements(
        load_degraded_four_sensor_measurements(DEGRADED), begin=BEGIN, end=END)
    with np.load(POSE_ARCHIVE, allow_pickle=False) as archive:
        expected_keys = {"seed_lidar", "seed_degraded_gps"} | {
            f"{case}__{seed}" for case in CASES for seed in SEED_NAMES
        }
        if set(archive.files) != expected_keys:
            raise ValueError("incomplete V8.3.3 pose archive")
        for case in TARGETS:
            graph, _, counts = run_four_sensor_factor_graph(
                measurements, METADATA.parent / "NO_WRITES", mode="fixed",
                gps_sigmas_override=sigmas[case], build_graph_only=True,
            )
            if counts != FACTORS or graph.size() != 2709:
                raise ValueError("factor count mismatch")
            unary, between = [], []
            for i in range(1, graph.size()):  # factor 0 is the initial pose prior
                f = graph.at(i)
                keys = list(f.keys())
                if len(keys) == 1:
                    unary.append((i, int(keys[0])))
                elif len(keys) == 2:
                    between.append((i, int(keys[0])))
                else:
                    raise ValueError("unexpected factor layout")
            if len(unary) != 680 or [k for _, k in unary] != list(range(680)):
                raise ValueError("unexpected GPS factor ordering")
            if len(between) != 2028:
                raise ValueError("unexpected relative factor count")
            for seed in SEED_NAMES:
                poses = build_values_from_pose_matrices(
                    gtsam, archive[f"{case}__{seed}"])
                row = data["case_results"][case]["initializers"][seed]
                objective = float(graph.error(poses))
                if not np.isclose(objective, row["objective"], rtol=1e-6, atol=1e-6):
                    raise ValueError("saved pose objective reproduction failed")
                groups = {"legacy_gps": [], "position_gps": [], "relative": []}
                for i, k in sample(unary):
                    old = graph.at(i)
                    new = gps_factor(gtsam, k, measurements.gps_local[k], sigmas[case][k])
                    for name, f in (("legacy_gps", old), ("position_gps", new)):
                        rec = factor_gradient_check(gtsam, f, poses)
                        groups[name].append(dict(rec, factor_index=i, frame=k))
                for i, k in sample(between):
                    rec = factor_gradient_check(gtsam, graph.at(i), poses)
                    groups["relative"].append(dict(rec, factor_index=i, frame=k))
                print("FACTOR_GRADIENT_AUDIT", json.dumps({
                    "case": case, "seed": seed, "reproduced_objective": objective,
                    "groups": {k: summarize(v) for k, v in groups.items()},
                }), flush=True)
    print("V8_3_4_FACTOR_GRADIENT_AUDIT_COMPLETE; NO_SOLVER; NO_WRITES; NO_HELDOUT")


if __name__ == "__main__":
    main()
