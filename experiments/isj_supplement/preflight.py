from __future__ import annotations
"""Read-only preflight: identify missing files before GPU or FGO runs."""
import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEQS = (
    "2011_10_03_drive_0034_sync",
    "2011_10_03_drive_0042_sync",
    "2011_10_03_drive_0047_sync",
)
DATE = ROOT / "dataset" / "kitti" / "2011_10_03"
OUT = ROOT / "results" / "cross_sequence"

def check():
    groups = {}
    groups["repository"] = [
        ROOT / "experiments" / "cross_sequence" / "run_stage7_factor_graph.py",
        ROOT / "experiments" / "cross_sequence" / "run_stage7_two_stage.py",
    ]
    groups["frozen_checkpoints"] = [
        *[ROOT / "results" / "sensor_specific_reliability_v5_uncertainty" /
          s / "model.pt" for s in ("gps", "imu", "camera")],
        *[ROOT / "results" / "v6_mamba_uncertainty" /
          s / "best.pt" for s in ("gps", "imu", "camera")],
    ]
    raw = [
        DATE / f for f in (
            "calib_cam_to_cam.txt", "calib_velo_to_cam.txt",
            "calib_imu_to_velo.txt"
        )
    ]
    for seq in SEQS:
        for kind in ("oxts", "velodyne_points", "image_02", "image_03"):
            raw.append(DATE / seq / kind / "data")
    groups["kitti_raw"] = raw
    processed = []
    for seq in SEQS:
        p = OUT / seq
        processed += [
            p / name for name in (
                "features.npz", "sensor_data.npz", "lidar_factor_data.npz",
                "camera_factor_data.npz", "ground_truth.txt",
                "stage7_two_stage_results.csv",
                "factor_graph/FINAL_S0_fixed/poses.npy",
            )
        ]
        for model in ("v5", "v6"):
            for sensor in ("gps", "imu", "camera"):
                processed += [
                    p / "predictions" / model / f"{sensor}_{kind}.txt"
                    for kind in ("reliability", "uncertainty")
                ]
    groups["stage7_processed"] = processed
    return {group: [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
            for group, paths in groups.items()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require", choices=("clone", "raw", "ready"), default="clone")
    p.add_argument("--json", action="store_true", help="Machine-readable report")
    args = p.parse_args()
    missing = check()
    present = [name for name in ("numpy", "scipy", "torch", "cv2", "gtsam", "mamba_ssm")
               if importlib.util.find_spec(name) is not None]
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "present_modules": present,
        "missing": missing,
        "required_state": args.require,
    }
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"Python {info['python']} on {info['platform']}")
        print("Available modules:", ", ".join(present) or "(none)")
        for group, items in missing.items():
            print(f"{group}: {'OK' if not items else str(len(items)) + ' missing'}")
            for item in items[:8]:
                print("  -", item)
            if len(items) > 8:
                print(f"  ... {len(items) - 8} more")
    needed = {
        "clone": ("repository", "frozen_checkpoints"),
        "raw": ("repository", "frozen_checkpoints", "kitti_raw"),
        "ready": ("repository", "frozen_checkpoints", "stage7_processed"),
    }[args.require]
    failures = [group for group in needed if missing[group]]
    if failures:
        print("FAIL: Missing prerequisites:", ", ".join(failures), file=sys.stderr)
        raise SystemExit(2)
    print(f"PASS: --require {args.require}")

if __name__ == "__main__":
    main()
