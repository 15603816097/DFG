from __future__ import annotations

"""Read-only inventory of KITTI Raw drives on other acquisition dates.

Does not preprocess, train, infer, or alter Stage-7/ISJ outputs. It only
checks whether the layout required by existing Stage-7 loaders is present:
dataset/kitti/<date>/<date>_drive_<id>_sync/{oxts,velodyne_points,image_02,
image_03}/{data,timestamps.txt} and the date-level 3 calibration files.

NOTE: "READY_RAW" means only that the expected files/counts are present.
It does NOT establish calibration validity, clock synchronization, camera
readability, unbiased ground truth, or absence of training leakage. Existing
Stage-7 scripts are hard-coded to 2011_10_03 and cannot run cross-date without
an isolated generalization of the pipeline and protocol checks.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KITTI_ROOT = ROOT / "dataset" / "kitti"
DEV_DATE = "2011_10_03"
DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
DRIVE_RE = re.compile(r"^(\d{4}_\d{2}_\d{2})_drive_\d+_sync$")

REQUIRED_CALIBRATION = (
    "calib_cam_to_cam.txt",
    "calib_imu_to_velo.txt",
    "calib_velo_to_cam.txt",
)
STREAMS = {
    "oxts": ".txt",
    "velodyne_points": ".bin",
    "image_02": ".png",
    "image_03": ".png",
}


def line_count(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def inspect_drive(
    date_dir: Path, drive: Path, min_frames: int = 100
) -> dict:
    """Validate known file layout without opening large images/point clouds."""
    if min_frames < 1:
        raise ValueError("min_frames must be positive")
    match = DRIVE_RE.fullmatch(drive.name)
    if match is None or match.group(1) != date_dir.name:
        raise ValueError(f"Invalid synchronized drive directory: {drive}")
    missing = []
    counts = {}
    for calib in REQUIRED_CALIBRATION:
        if not (date_dir / calib).is_file():
            missing.append(str(date_dir / calib))
    for kind, extension in STREAMS.items():
        data_dir = drive / kind / "data"
        timestamp = drive / kind / "timestamps.txt"
        if not data_dir.is_dir():
            missing.append(str(data_dir))
            frames = []
        else:
            frames = sorted(data_dir.glob("*" + extension))
            if not frames:
                missing.append(str(data_dir / ("*" + extension)))
        if not timestamp.is_file():
            missing.append(str(timestamp))
        ts_count = line_count(timestamp)
        counts[kind] = {
            "frames": len(frames),
            "timestamps": ts_count,
            "file_timestamp_count_match": len(frames) == ts_count,
        }

    all_frames = [row["frames"] for row in counts.values()]
    matches = (len(set(all_frames)) == 1
               and all(x["file_timestamp_count_match"] for x in counts.values()))
    min_ok = all(x >= min_frames for x in all_frames)
    cross_date = date_dir.name != DEV_DATE
    status = ("READY_RAW"
              if not missing and matches and min_ok and cross_date
              else "NOT_READY_RAW")
    issues = []
    if missing:
        issues.append(f"{len(missing)} missing paths")
    if not matches:
        issues.append("modality frame/timestamp count mismatch")
    if not min_ok:
        issues.append(f"< {min_frames} synchronized frames")
    if not cross_date:
        issues.append("same date as Stage-7 development and prior tests")
    return {
        "date": date_dir.name,
        "drive": drive.name,
        "cross_date_candidate": cross_date,
        "raw_layout_status": status,
        "counts": counts,
        "missing": missing,
        "issues": issues,
        "caveat": "File-count readiness only. Stage-7 code still hardcodes 2011_10_03.",
    }


def discover(root: Path, min_frames: int = 100) -> list[dict]:
    if not root.is_dir():
        return []
    records = []
    for date in sorted(p for p in root.iterdir()
                       if p.is_dir() and DATE_RE.fullmatch(p.name)):
        for drive in sorted(p for p in date.iterdir()
                            if p.is_dir() and DRIVE_RE.fullmatch(p.name)):
            records.append(inspect_drive(date, drive, min_frames))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=KITTI_ROOT)
    parser.add_argument("--min-frames", type=int, default=100)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional independent JSON report (no original files modified)")
    args = parser.parse_args()
    if args.min_frames < 1:
        parser.error("--min-frames must be >= 1")
    rows = discover(args.root, args.min_frames)
    candidates = [r for r in rows if r["cross_date_candidate"]]
    ready = [r for r in candidates if r["raw_layout_status"] == "READY_RAW"]
    print("KITTI raw root:", args.root)
    print(f"Total synchronized drives found: {len(rows)}")
    print(f"Different-date drives found: {len(candidates)}")
    print(f"Different-date RAW-layout-ready drives: {len(ready)}")
    for row in rows:
        counts = " ".join(
            f"{name}:{v['frames']}/{v['timestamps']}"
            for name, v in row["counts"].items()
        )
        print(f"  {row['date']} {row['drive']} "
              f"{row['raw_layout_status']} {counts}")
        if row["cross_date_candidate"] and row["issues"]:
            print("   Issues:", "; ".join(row["issues"]))
            for path in row["missing"][:5]:
                print("     Missing:", path)
    if not candidates:
        print("No different-date KITTI Raw drive found locally.")
        print("Obtain a legitimately accessible synchronized KITTI Raw")
        print("drive from another acquisition date, with its OWN calibrations")
        print("and timestamp files, before generalizing the Stage-7 pipeline.")
    elif not ready:
        print("Cross-date drive directories exist, but none passes the")
        print("current loader-path and frame-count preflight.")
    if ready:
        print("READY_RAW is a file-layout check only; do not run old")
        print("Stage-7 scripts against new dates without code generalization.")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({
                "root": str(args.root),
                "min_frames": args.min_frames,
                "records": rows,
                "ready_cross_date": [r["drive"] for r in ready],
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Saved:", args.json_out)


if __name__ == "__main__":
    main()
