from pathlib import Path

import pytest

from experiments.isj_supplement.audit_cross_date_raw import (
    REQUIRED_CALIBRATION, STREAMS, discover, inspect_drive,
)


def make_drive(root: Path, date: str = "2011_09_26",
               name: str = "2011_09_26_drive_0001_sync",
               frames: int = 3):
    date_dir = root / date
    date_dir.mkdir(parents=True)
    for c in REQUIRED_CALIBRATION:
        (date_dir / c).write_text("calibration\n", encoding="utf-8")
    drive = date_dir / name
    for kind, ext in STREAMS.items():
        p = drive / kind / "data"
        p.mkdir(parents=True)
        for i in range(frames):
            (p / f"{i:010d}{ext}").write_bytes(b"fake")
        (drive / kind / "timestamps.txt").write_text(
            "2011-09-26 00:00:00.000000\n" * frames, encoding="utf-8"
        )
    return date_dir, drive


def test_cross_date_full_file_layout_can_be_ready(tmp_path):
    date_dir, drive = make_drive(tmp_path)
    row = inspect_drive(date_dir, drive, min_frames=3)
    assert row["raw_layout_status"] == "READY_RAW"
    assert row["cross_date_candidate"]
    assert len(row["missing"]) == 0
    assert all(v["frames"] == 3 for v in row["counts"].values())


def test_missing_timestamp_is_not_ready(tmp_path):
    date_dir, drive = make_drive(tmp_path)
    (drive / "oxts" / "timestamps.txt").unlink()
    row = inspect_drive(date_dir, drive, min_frames=2)
    assert row["raw_layout_status"] == "NOT_READY_RAW"
    assert any("oxts/timestamps.txt" in p for p in row["missing"])


def test_same_development_date_cannot_qualify_as_cross_date(tmp_path):
    date_dir, drive = make_drive(
        tmp_path, date="2011_10_03",
        name="2011_10_03_drive_0034_sync",
    )
    row = inspect_drive(date_dir, drive, min_frames=2)
    assert row["cross_date_candidate"] is False
    assert row["raw_layout_status"] == "NOT_READY_RAW"


def test_discover_ignores_non_drive_directories(tmp_path):
    date_dir, drive = make_drive(tmp_path)
    (date_dir / "not_a_drive").mkdir()
    other = tmp_path / "other"
    other.mkdir()
    rows = discover(tmp_path, min_frames=2)
    assert [r["drive"] for r in rows] == [drive.name]


def test_wrong_date_dir_rejected(tmp_path):
    date_dir, drive = make_drive(tmp_path)
    with pytest.raises(ValueError):
        inspect_drive(tmp_path / "2011_10_03", drive, min_frames=2)
