from __future__ import annotations

"""Combine the completed 2^3 sensor-uncertainty interventions without reruns.

Each factor graph has 3 binary U switches (GPS, IMU, Camera). All reliability
signals are always on, LiDAR noise is fixed; only u_s(t) is zeroed for some
sensors. This post-hoc study is descriptive, not a blind causal experiment.

Inputs: existing isj_convergence_{core,ablations,sensor_u_drop,sensor_u_only}
results.csv from the original 3 sequence subfolders.

Outputs: results/cross_sequence/isj_sensor_u_factorial_summary.csv (6 rows),
         results/cross_sequence/isj_sensor_u_factorial_long.csv (48 rows).
All source data and original FGO results remain untouched.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence.stage7_config import OUT_ROOT, TEST_SEQUENCES

MODEL_CASES = {
    "GRU": ("FINAL_S1_v5_gru", "ABL_V5_GRU_RELIABILITY_ONLY"),
    "Mamba": ("FINAL_S2_v6_all_mamba", "ABL_V6_MAMBA_RELIABILITY_ONLY"),
}

# U on/off order: GPS, IMU, Camera.
U_MASKS = {
    "000": "reliability_only",
    "100": "only_gps",
    "010": "only_imu",
    "001": "only_camera",
    "011": "drop_gps",
    "101": "drop_imu",
    "110": "drop_camera",
    "111": "full_dynamic",
}
MASK_TO_GROUP = {
    "000": "ablations",
    "111": "core",
    "100": "sensor_u_only",
    "010": "sensor_u_only",
    "001": "sensor_u_only",
    "011": "sensor_u_drop",
    "101": "sensor_u_drop",
    "110": "sensor_u_drop",
}


def read_case_csv(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    result = {}
    for row in data:
        key = row.get("case", "")
        if not key or key in result:
            raise ValueError(f"{path}: empty or duplicate case {key!r}")
        result[key] = row
    if not result:
        raise ValueError(f"{path}: empty CSV")
    return result


def source_case(model: str, mask: str) -> str:
    if model not in MODEL_CASES or mask not in U_MASKS:
        raise ValueError(f"Unknown model/mask {model}/{mask}")
    full, reliability = MODEL_CASES[model]
    if mask == "000":
        return reliability
    if mask == "111":
        return full
    prefix = "GRU" if model == "GRU" else "MAMBA"
    if mask.count("1") == 1:
        return f"ISJ_SENSU_{prefix}_ONLY_{('GPS','IMU','CAMERA')[mask.index('1')]}"
    return f"ISJ_SENSU_{prefix}_DROP_{('GPS','IMU','CAMERA')[mask.index('0')]}"


def extract(row: dict, label: str) -> tuple[float, bool]:
    try:
        ate = float(row["continued_ATE3D"])
        status = row["stationary_under_rule"].strip().lower()
    except (KeyError, ValueError, TypeError) as err:
        raise ValueError(f"{label}: malformed result row") from err
    if not (math.isfinite(ate) and ate > 0):
        raise ValueError(f"{label}: invalid ATE3D: {ate}")
    if status not in ("true", "false"):
        raise ValueError(f"{label}: invalid stationary status {status}")
    return ate, status == "true"


def process_sequence(root: Path, seq: str) -> tuple[list[dict], list[dict]]:
    files = {
        group: read_case_csv(root / seq / f"isj_convergence_{group}_results.csv")
        for group in ("core", "ablations", "sensor_u_only", "sensor_u_drop")
    }
    summaries, long_rows = [], []
    for model in MODEL_CASES:
        ate = {}
        stable = {}
        for mask, label in U_MASKS.items():
            group = MASK_TO_GROUP[mask]
            case = source_case(model, mask)
            if case not in files[group]:
                raise KeyError(f"{seq}/{model}/{mask}: missing {case!r} in {group}")
            value, converged = extract(files[group][case], f"{seq}/{case}")
            ate[mask] = value
            stable[mask] = converged
            long_rows.append({
                "sequence": seq, "model": model,
                "U_GPS": int(mask[0]), "U_IMU": int(mask[1]),
                "U_CAMERA": int(mask[2]), "mask": mask,
                "configuration": label, "original_case": case,
                "ATE3D_m": value,
                "stationary_under_rule": converged,
                "classification": "post_hoc_exploratory",
            })

        # Signed ATE gain: positive means adding U lowers the error.
        def gain(without: str, with_: str) -> float:
            return (ate[without] - ate[with_]) / ate[without] * 100

        summaries.append({
            "sequence": seq,
            "model": model,
            **{f"ATE3D_{label}_m": ate[mask] for mask, label in U_MASKS.items()},
            "GPS_U_gain_when_IMU_CAMERA_off_pct": gain("000", "100"),
            "GPS_U_gain_when_IMU_CAMERA_on_pct": gain("011", "111"),
            "Camera_U_gain_when_GPS_IMU_off_pct": gain("000", "001"),
            "Camera_U_gain_when_GPS_IMU_on_pct": gain("110", "111"),
            "IMU_U_gain_when_GPS_CAMERA_off_pct": gain("000", "010"),
            "IMU_U_gain_when_GPS_CAMERA_on_pct": gain("101", "111"),
            "all_8_cases_stationary": all(stable.values()),
            "classification": "post_hoc_exploratory",
        })
    return summaries, long_rows


def save(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No results")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    summaries, details = [], []
    for seq in TEST_SEQUENCES:
        a, b = process_sequence(args.root, seq)
        summaries.extend(a)
        details.extend(b)
    summary_path = args.root / "isj_sensor_u_factorial_summary.csv"
    long_path = args.root / "isj_sensor_u_factorial_long.csv"
    save(summary_path, summaries)
    save(long_path, details)
    print(f"{'seq':4} {'model':6} {'R-only':>9} {'Only GPS':>9} "
          f"{'Full U':>9} {'GPS alone':>11} {'GPS with others':>16}  status")
    for row in summaries:
        seq = row["sequence"].split("_drive_")[1][:4]
        print(
            f"{seq:4} {row['model']:6} "
            f"{row['ATE3D_reliability_only_m']:9.4f} "
            f"{row['ATE3D_only_gps_m']:9.4f} "
            f"{row['ATE3D_full_dynamic_m']:9.4f} "
            f"{row['GPS_U_gain_when_IMU_CAMERA_off_pct']:+10.2f}% "
            f"{row['GPS_U_gain_when_IMU_CAMERA_on_pct']:+15.2f}%  "
            f"{'STABLE' if row['all_8_cases_stationary'] else 'NOT_ALL_STABLE'}"
        )
    print("Saved:", summary_path)
    print("Saved:", long_path)
    print("NOTE: Percentages are *conditional on other U switches* and are not")
    print("additive or causal attribution. 3 sequences share one acquisition")
    print("date; evaluation reference OXTS/GPS is not independent of GPS input.")
    print("Experiments designed after test inspection: exploratory/post-hoc.")


if __name__ == "__main__":
    main()
