import numpy as np
from src.features.factor_oriented_features import (
    build_gps_factor_features,
    build_imu_factor_features,
)

def test_gps_feature_shape():
    n = 20
    base = np.zeros((n, 5), dtype=np.float32)
    position = np.zeros((n, 3), dtype=np.float64)
    timestamps = np.arange(n, dtype=np.float64) * 0.1
    result = build_gps_factor_features(base, position, timestamps)
    assert result.shape[0] == n
    assert result.shape[1] > base.shape[1]

def test_imu_feature_shape():
    n = 20
    base = np.zeros((n, 5), dtype=np.float32)
    acc = np.zeros((n, 3), dtype=np.float64)
    gyro = np.zeros((n, 3), dtype=np.float64)
    timestamps = np.arange(n, dtype=np.float64) * 0.1
    result = build_imu_factor_features(base, acc, gyro, timestamps)
    assert result.shape[0] == n
    assert result.shape[1] > base.shape[1]
