from __future__ import annotations

"""Safely inspect/install one newly uploaded KITTI Raw *sync.zip* archive.

First run WITHOUT --install for read-only preview. With --install it
extracts atomically to:
  dataset/kitti/<date>/<date>_drive_<id>_sync/
and refuses any existing target folder. It never overwrites training
0027, exploratory 0016, original results, or original archive contents.

Accepted archive member layouts:
  2011_09_30_drive_0020_sync/oxts/data/...
  2011_09_30/2011_09_30_drive_0020_sync/oxts/data/...
Only a single selected drive is permitted. ZIP path traversal, symlinks,
nonregular entries, repeated filenames, and compressed bomb ratios are
rejected. Files are CRC-checked as they stream-extract.

KITTI's date-level calib_*.txt are often SEPARATE from a drive ZIP. If
missing, this script prints them; it does not borrow calibration from
a different date or invent values. A previous 0016 drive on the same
date should have supplied the same date's own calibrations.

This is a data installation utility, NOT model inference or evaluation.
"""

import argparse
from collections import Counter
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
KITTI = ROOT / "dataset" / "kitti"
DRIVE_RE = re.compile(r"^(\d{4}_\d{2}_\d{2})_drive_\d+_sync$")
REQUIRED = ("oxts", "velodyne_points", "image_02", "image_03")
CALIB = (
    "calib_cam_to_cam.txt",
    "calib_imu_to_velo.txt",
    "calib_velo_to_cam.txt",
)


def safe_parts(name: str, drive: str) -> tuple[str, ...]:
    """Return path *inside* the desired drive, or reject member."""
    m = DRIVE_RE.fullmatch(drive)
    if m is None:
        raise ValueError(f"Bad KITTI drive name: {drive}")
    if "\\" in name or name.startswith(("/", "~")) or "\x00" in name:
        raise ValueError(f"Unsafe ZIP path: {name!r}")
    path = PurePosixPath(name)
    parts = path.parts
    if any(part in ("", ".", "..") for part in name.split("/")):
        # The final slash of a directory entry is normal.
        tokens = name.rstrip("/").split("/")
        if any(t in ("", ".", "..") for t in tokens):
            raise ValueError(f"Unsafe ZIP path components: {name!r}")
    if not parts:
        raise ValueError("Empty ZIP member")
    if parts[0] == drive:
        stripped = parts[1:]
    elif len(parts) >= 2 and parts[:2] == (m.group(1), drive):
        stripped = parts[2:]
    else:
        raise ValueError(
            f"ZIP member outside expected {drive} root: {name!r}"
        )
    return stripped


def inventory(zip_path: Path, drive: str) -> tuple[list, int, dict]:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    seen = set()
    rows = []
    total = 0
    counts = Counter()
    with zipfile.ZipFile(zip_path) as z:
        if len(z.infolist()) == 0:
            raise ValueError("Archive is empty")
        for info in z.infolist():
            inner = safe_parts(info.filename, drive)
            if not inner:
                if not info.is_dir():
                    raise ValueError("Non-directory ZIP drive root")
                continue
            if any(part in (".", "..") for part in inner):
                raise ValueError(f"Unsafe traversal path: {info.filename!r}")
            mode = (info.external_attr >> 16) & 0xFFFF
            ftype = stat.S_IFMT(mode)
            if ftype not in (0, stat.S_IFDIR, stat.S_IFREG):
                raise ValueError(f"Unsupported ZIP entry type: {info.filename!r}")
            if info.is_dir() and ftype == stat.S_IFREG:
                raise ValueError(f"Directory marked as file: {info.filename!r}")
            if (not info.is_dir()) and ftype == stat.S_IFDIR:
                raise ValueError(f"File marked as directory: {info.filename!r}")
            key = "/".join(inner).rstrip("/")
            if key in seen:
                raise ValueError(f"Duplicate ZIP path: {key}")
            seen.add(key)
            if not info.is_dir():
                total += info.file_size
                # KITTI png/bin compression varies; avoid extreme zip bombs.
                if info.file_size > 100_000_000_000:
                    raise ValueError(f"Overlarge member: {key}")
                if info.file_size > 1_000_000_000 and info.compress_size < info.file_size/200:
                    raise ValueError(f"Suspicious compression ratio: {key}")
                if inner[0] in REQUIRED:
                    counts[inner[0]] += 1
            rows.append((info, inner))
        if total > 500_000_000_000:
            raise ValueError("ZIP uncompressed total exceeds 500 GB limit")
    return rows, total, dict(counts)


def install(zip_path: Path, drive: str, kitti_root: Path, do_install: bool) -> None:
    m = DRIVE_RE.fullmatch(drive)
    if not m:
        raise ValueError("Expected name like 2011_09_30_drive_0020_sync")
    date_dir = (kitti_root / m.group(1)).resolve()
    destination = date_dir / drive
    rows, total, counts = inventory(zip_path, drive)
    print("="*90)
    print("KITTI SYNC ZIP - SAFE INSTALL PREVIEW")
    print("archive:", zip_path.resolve())
    print("destination:", destination)
    print("ZIP members:", len(rows))
    print("uncompressed GiB:", round(total / 1024**3, 3))
    print("ZIP per-family files:", counts)
    print("destination already exists:", destination.exists())
    missing = [c for c in CALIB if not (date_dir / c).is_file()]
    print("date-level calibration missing:", missing or "none")
    print("Note: calib files are NOT automatically copied from a different date.")

    # Only check whether files exist in the ZIP, not whether all frame
    # counts match; the established raw audit performs the full check.
    for sensor in REQUIRED:
        wanted = f"{sensor}/timestamps.txt"
        has_times = any("/".join(parts) == wanted for _, parts in rows)
        data_count = sum(
            1 for info, parts in rows if len(parts) == 3
            and parts[0] == sensor and parts[1] == "data"
            and not info.is_dir()
        )
        print(f"  {sensor:16s}: data={data_count:<6d}, timestamps={has_times}")
        if data_count < 100 or not has_times:
            raise ValueError(
                f"Incomplete sync ZIP: {sensor} missing >=100 files or timestamps"
            )

    if not do_install:
        print("DRY RUN ONLY. To unpack without overwriting anything, add --install.")
        return

    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; never merge/overwrite: {destination}"
        )
    date_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(date_dir).free
    if free < total + 1024**3:
        raise OSError(
            f"Insufficient disk free: free={free/1024**3:.3f} GiB, "
            f"required>={total/1024**3+1:.3f} GiB"
        )
    staging = date_dir / (f".{drive}.install-{uuid.uuid4().hex[:12]}")
    staging.mkdir(mode=0o700)
    success = False
    try:
        with zipfile.ZipFile(zip_path) as z:
            for info, parts in rows:
                dest = staging.joinpath(*parts)
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info, "r") as src, dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
        # Check required file presence before atomic move.
        for sensor in REQUIRED:
            base = staging / sensor
            if not (base / "timestamps.txt").is_file():
                raise RuntimeError(f"Missing {sensor}/timestamps.txt after extraction")
            if not (base / "data").is_dir():
                raise RuntimeError(f"Missing {sensor}/data after extraction")
        if destination.exists():
            raise FileExistsError(f"Target created concurrently: {destination}")
        os.rename(staging, destination)
        success = True
    finally:
        if not success and staging.exists():
            shutil.rmtree(staging)
    print("INSTALL COMPLETE:", destination)
    if missing:
        print("Next action: supply SAME-DATE official calibration files:", missing)
    print("Next action: run python -u experiments/isj_supplement/run_cross_date.py "
          f"--sequence {drive} --stage inspect")
    print("No training, inference, optimization or test-metric computation performed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--sequence", default="2011_09_30_drive_0020_sync")
    p.add_argument("--install", action="store_true",
                   help="Explicit opt-in to extract to a new, never-overwritten drive folder")
    args = p.parse_args()
    install(args.archive, args.sequence, KITTI, args.install)


if __name__ == "__main__":
    main()
