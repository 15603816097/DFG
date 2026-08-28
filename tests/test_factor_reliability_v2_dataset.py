import numpy as np
from src.dataset.factor_reliability_v2_dataset import (
    compute_normalization,
    apply_normalization,
)

def test_normalization():
    features = {
        sensor: np.random.randn(100, 4).astype(np.float32)
        for sensor in ("gps", "imu", "lidar", "camera")
    }

    normalization = compute_normalization(features, 70)
    normalized = apply_normalization(features, normalization)

    for sensor in features:
        assert normalized[sensor].shape == (100, 4)
