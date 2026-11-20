from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from experiments.isj_supplement.install_kitti_sync_zip import (
    inventory, install, safe_parts,
)

DRIVE = "2011_09_30_drive_0020_sync"


def fake_zip(file: Path, *, data_count=100):
    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_STORED) as z:
        for mod, ext in (
            ("oxts", ".txt"), ("velodyne_points", ".bin"),
            ("image_02", ".png"), ("image_03", ".png"),
        ):
            z.writestr(f"{DRIVE}/{mod}/timestamps.txt", "0\n" * data_count)
            for i in range(data_count):
                z.writestr(
                    f"{DRIVE}/{mod}/data/{i:010d}{ext}",
                    b"x" if ext != ".txt" else b"0 0 0\n",
                )


def test_zip_prefixes_are_unwrapped_and_path_traversal_is_rejected():
    assert safe_parts(f"{DRIVE}/oxts/data/0.txt", DRIVE) == (
        "oxts", "data", "0.txt",
    )
    assert safe_parts(f"2011_09_30/{DRIVE}/image_02/a.png", DRIVE) == (
        "image_02", "a.png"
    )
    for unsafe in (
        f"{DRIVE}/../oxts/data/1.txt",
        f"/{DRIVE}/oxts/data/0.txt",
        f"{DRIVE}/oxts\\data\\0.txt",
        "2011_10_03_drive_0027_sync/oxts/data/0.txt",
    ):
        with pytest.raises(ValueError):
            safe_parts(unsafe, DRIVE)


def test_install_is_dry_run_first_atomic_and_existing_dest_is_never_replaced(tmp_path):
    file = tmp_path / "new.zip"
    fake_zip(file)
    records, total, counts = inventory(file, DRIVE)
    assert total > 0
    assert counts == {s: 101 for s in (
        "oxts", "velodyne_points", "image_02", "image_03"
    )}
    data_root = tmp_path / "kitti"
    target = data_root / "2011_09_30" / DRIVE

    install(file, DRIVE, data_root, do_install=False)
    assert not target.exists()

    install(file, DRIVE, data_root, do_install=True)
    assert (target / "oxts" / "timestamps.txt").exists()
    assert len(list((target / "image_02" / "data").glob("*.png"))) == 100

    with pytest.raises(FileExistsError):
        install(file, DRIVE, data_root, do_install=True)


def test_incomplete_zip_is_rejected_even_in_preview(tmp_path):
    file = tmp_path / "bad.zip"
    fake_zip(file, data_count=2)
    with pytest.raises(ValueError, match="Incomplete sync ZIP"):
        install(file, DRIVE, tmp_path / "kitti", do_install=False)


def test_bad_or_duplicate_entry_rejected(tmp_path):
    file = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(file, "w") as z:
        z.writestr(f"{DRIVE}/image_02/data/a.png", b"first")
        z.writestr(f"{DRIVE}/image_02/data/a.png", b"second")
    with pytest.raises(ValueError, match="Duplicate ZIP path"):
        inventory(file, DRIVE)
