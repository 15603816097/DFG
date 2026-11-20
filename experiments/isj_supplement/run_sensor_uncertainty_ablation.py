from __future__ import annotations

"""Frozen, targeted sensor-uncertainty ablation for Stage-7 FGO.

All GPS/IMU/Camera reliability predictions are ALWAYS retained. Only the
uncertainty time series used by selected sensor factors changes:
    DROP sensor:  u_s(t) = 0 for that sensor; other sensors keep their u(t).
    ONLY sensor:  only that sensor keeps u(t); all others have u_s(t) = 0.

With u=0, confidence(u)=1 and the sensor uses reliability-only noise mapping.
This is a *controlled intervention on the FGO mapping*, not an intervention
on physical observations. Sensor contributions can interact in graph
optimization; differences are not additive nor causal sensor attribution.

Every case uses the saved Stage-1 fixed-FGO poses as warm start. Does not
modify original Stage-7 results, networks, predictions, or reference data.
Raw KITTI is not needed. Exploratory/post-hoc analysis.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence import run_stage7_factor_graph as s7
from experiments.cross_sequence.stage7_config import TEST_SEQUENCES, sequence_out
from experiments.isj_supplement.run_controls import preflight

SENSORS = ("gps", "imu", "camera")
MODELS = (("GRU", "v5"), ("MAMBA", "v6"))
STRATEGIES = ("drop", "only")


def target_sensors(strategy: str, sensor: str) -> frozenset[str]:
    """Returns which sensors retain their original uncertainty time-series."""
    if strategy not in STRATEGIES or sensor not in SENSORS:
        raise ValueError(f"Unknown sensor strategy {strategy}/{sensor}")
    return (frozenset(SENSORS) - {sensor}
            if strategy == "drop" else frozenset((sensor,)))


def select_uncertainty(
    sensor: str,
    uncertainty: np.ndarray,
    active_uncertainty_sensors: frozenset[str],
) -> np.ndarray:
    """Defensive copy: original predictions are never modified."""
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor {sensor}")
    if not active_uncertainty_sensors.issubset(set(SENSORS)):
        raise ValueError(f"Unknown active sensors {active_uncertainty_sensors}")
    arr = np.asarray(uncertainty, dtype=float).reshape(-1)
    if not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
        raise ValueError("Uncertainty must be finite in [0,1]")
    if sensor in active_uncertainty_sensors:
        return arr.copy()
    return np.zeros_like(arr)


def case_name(model: str, strategy: str, sensor: str) -> str:
    if model not in dict(MODELS):
        raise ValueError(f"Unknown model {model}")
    target_sensors(strategy, sensor)
    return f"ISJ_SENSU_{model}_{strategy.upper()}_{sensor.upper()}"


def available_specs(strategy: str = "all") -> list[dict]:
    if strategy != "all" and strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}")
    selected_strategies = STRATEGIES if strategy == "all" else (strategy,)
    return [
        {
            "name": case_name(model, mode, sensor),
            "model": model,
            "source": source,
            "type": "sensor_u",
            "strategy": mode,
            "sensor": sensor,
            "active_uncertainty_sensors": tuple(sorted(target_sensors(mode, sensor))),
        }
        for model, source in MODELS
        for mode in selected_strategies
        for sensor in SENSORS
    ]


def run(seq: str, spec: dict, fixed_poses: np.ndarray, rerun: bool) -> dict:
    path = sequence_out(seq) / "factor_graph" / spec["name"] / "metrics.json"
    if path.is_file() and not rerun:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        status = "cached"
        elapsed = None
    else:
        source = spec["source"]
        active = frozenset(spec["active_uncertainty_sensors"])
        original_pred = s7.pred
        original_init = s7.initial_trajectory_from_pose

        def pred_patch(sequence, model_source, sensor, n):
            r, u = original_pred(sequence, model_source, sensor, n)
            return r, select_uncertainty(sensor, u, active)

        def init_patch(gtsam, measurement, pose0):
            if len(measurement.gps_local) != len(fixed_poses):
                raise ValueError("Fixed stage pose length mismatch")
            return [s7._pose3_from_matrix(gtsam, T) for T in fixed_poses]

        start = time.perf_counter()
        try:
            s7.pred = pred_patch
            s7.initial_trajectory_from_pose = init_patch
            metrics = s7.run_case(
                seq=seq, name=spec["name"],
                gps_src=source, imu_src=source, cam_src=source,
            )
        finally:
            s7.pred = original_pred
            s7.initial_trajectory_from_pose = original_init
        elapsed = time.perf_counter() - start
        status = "computed"

    return {
        "sequence": seq,
        "case": spec["name"],
        "model": spec["model"],
        "strategy": spec["strategy"],
        "sensor": spec["sensor"],
        "active_uncertainty_sensors":
            ";".join(spec["active_uncertainty_sensors"]),
        "ATE3D": float(metrics["ATE3D"]),
        "ATE2D": float(metrics["ATE2D"]),
        "ZRMSE": float(metrics["ZRMSE"]),
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else "",
        "status": status,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_10_03_drive_0047_sync",
                   choices=("all",) + TEST_SEQUENCES)
    p.add_argument("--strategy", choices=("all",) + STRATEGIES, default="drop")
    p.add_argument("--rerun", action="store_true")
    args = p.parse_args()
    sequences = TEST_SEQUENCES if args.sequence == "all" else (args.sequence,)
    specs = available_specs(args.strategy)

    for seq in sequences:
        fixed, _reference = preflight(seq)
        rows = []
        for spec in specs:
            print(f"\nSequence {seq}: {spec['name']}", flush=True)
            row = run(seq, spec, fixed, args.rerun)
            rows.append(row)
            print(f"  ATE3D={row['ATE3D']:.6f} m ({row['status']})", flush=True)
        out = sequence_out(seq)
        # Include family in filename: running --strategy drop does not
        # overwrite a previous all/only experiment summary.
        path = out / f"isj_sensor_uncertainty_{args.strategy}_results.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        protocol = {
            "sequence": seq,
            "strategy": args.strategy,
            "classification": "exploratory_post_hoc_sensorwise_U_intervention",
            "source_fixed_warmstart": "FINAL_S0_fixed/poses.npy",
            "predictions_retrained": False,
            "reliability_predictions_modified": False,
            "uncertainty_dropped_sensor": "set u=0 for all frames",
            "full_sigma_mapping_unchanged": True,
            "lidar_sigma": "fixed",
            "note": "Each factor graph couples sensors and time; do not interpret the ATE differences as additive independent causal effects.",
            "cases": [row["case"] for row in rows],
        }
        (out / f"isj_sensor_uncertainty_{args.strategy}_protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved: {path}", flush=True)


if __name__ == "__main__":
    main()
