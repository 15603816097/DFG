from __future__ import annotations

import numpy as np

from src.multisensor_reliability.features import gps_features, imu_features


def test_gps_features_do_not_change_if_only_future_measurements_change():
    rng = np.random.default_rng(20261119)
    gps = rng.normal(size=(120, 3)).cumsum(axis=0)
    original = gps_features(gps, dt=.1, window=10)
    other = gps.copy()
    other[80:] += 100000.0
    changed = gps_features(other, dt=.1, window=10)
    assert np.array_equal(original[:80], changed[:80])


def test_imu_features_do_not_change_if_only_future_measurements_change():
    rng = np.random.default_rng(20261119)
    acc = rng.normal(size=(120, 3))
    gyro = rng.normal(size=(120, 3))
    original = imu_features(acc, gyro, window=20)
    changed_acc = acc.copy()
    changed_gyro = gyro.copy()
    changed_acc[80:] *= 999
    changed_gyro[80:] -= 555
    changed = imu_features(changed_acc, changed_gyro, window=20)
    assert np.array_equal(original[:80], changed[:80])
