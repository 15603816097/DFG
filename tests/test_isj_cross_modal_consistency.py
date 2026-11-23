from __future__ import annotations

import numpy as np
import pytest

from src.reliability.cross_modal_consistency import (
    CrossModalSettings, measurement_consistency,
)


def make_stream(n=12):
    gps = np.stack([
        np.arange(n, dtype=float),
        np.zeros(n),
        np.zeros(n),
    ], axis=1)
    gyro = np.zeros((n, 3), dtype=float)
    ts = np.arange(n, dtype=float) * .1
    between = np.repeat(np.eye(4)[None], n-1, axis=0)
    between[:, 0, 3] = 1.0
    valid = np.ones(n-1, bool)
    quality = np.ones(n-1, float)
    return dict(
        gps_local=gps, imu_gyro=gyro, timestamps_seconds=ts,
        lidar_between=between, lidar_valid=valid, lidar_quality=quality,
    )


def test_perfect_agreement_q_is_one_and_masks_are_target_frame_aligned():
    d = make_stream()
    result = measurement_consistency(**d)
    assert result["gps_valid"][:5].tolist() == [False]*5
    assert np.all(result["gps_valid"][5:])
    assert result["imu_valid"][0] == False
    assert np.all(result["imu_valid"][1:])
    assert np.allclose(result["gps_quality"][5:], 1.0)
    assert np.allclose(result["imu_quality"][1:], 1.0)
    assert np.allclose(result["gps_nis"][5:], 0.0, atol=1e-10)
    assert np.allclose(result["imu_nis"][1:], 0.0, atol=1e-9)


def test_gps_jump_and_imu_rotation_outlier_decrease_consistency():
    d = make_stream()
    d["gps_local"][10, 0] += 50
    d["imu_gyro"][7, 2] += 2.0
    result = measurement_consistency(**d)
    assert result["gps_valid"][10]
    assert result["gps_quality"][10] < .5
    # gyro[7] corresponds to rotation factor 7 -> 8, stored at target frame 8
    assert result["imu_valid"][8]
    assert result["imu_quality"][8] < 1.0
    assert result["imu_quality"][7] == pytest.approx(1.0)


def test_invalid_lidar_pair_masks_feedback_rather_than_fabricating_signal():
    d = make_stream()
    d["lidar_valid"][6] = False
    result = measurement_consistency(**d)
    assert not result["imu_valid"][7]
    assert not result["gps_valid"][7]
    assert result["gps_quality"][7] == 1.0  # PLACEHOLDER only
    assert result["imu_quality"][7] == 1.0
    assert np.isnan(result["gps_nis"][7])
    assert np.isnan(result["imu_nis"][7])


def test_current_frame_feedback_does_not_use_future_measurements():
    d = make_stream(n=15)
    before = measurement_consistency(**d)
    # Only perturb measurements at future frame/pair >=12.
    d["gps_local"][12] += [50., 0., 0.]
    d["imu_gyro"][12] += [0., 0., 5.]
    d["lidar_between"][12, :3, 3] += [10., 0., 0.]
    after = measurement_consistency(**d)
    for key in ("gps_quality", "imu_quality", "gps_valid", "imu_valid"):
        assert np.array_equal(before[key][:12], after[key][:12])


def test_invalid_shapes_and_nan_rejected():
    d = make_stream()
    d["lidar_between"] = np.eye(4)
    with pytest.raises(ValueError, match="shape mismatch"):
        measurement_consistency(**d)
    d = make_stream()
    d["gps_local"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        measurement_consistency(**d)
    with pytest.raises(ValueError, match="gps_window"):
        CrossModalSettings(gps_window=0)
