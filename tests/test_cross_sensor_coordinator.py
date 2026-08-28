import numpy as np

from src.reliability.cross_sensor_coordinator import (
    CoordinatorConfig,
    coordinate_reliabilities,
)


def _sample(n=100):
    x = np.linspace(0.0, 1.0, n)
    return {
        "gps": np.clip(0.9 - 0.7 * x, 0.0, 1.0),
        "imu": np.clip(0.8 - 0.3 * x, 0.0, 1.0),
        "lidar": np.clip(0.95 - 0.8 * x, 0.0, 1.0),
        "camera": np.clip(0.85 - 0.4 * x, 0.0, 1.0),
    }


def test_output_shape_and_range():
    data = _sample()
    out = coordinate_reliabilities(data)

    for sensor in data:
        assert out[sensor].shape == data[sensor].shape
        assert np.all(np.isfinite(out[sensor]))
        assert np.all(out[sensor] >= 0.0)
        assert np.all(out[sensor] <= 1.0)


def test_bounded_change():
    data = _sample()
    cfg = CoordinatorConfig()
    out = coordinate_reliabilities(data, cfg)

    limits = {
        "gps": cfg.max_delta_gps,
        "imu": cfg.max_delta_imu,
        "lidar": cfg.max_delta_lidar,
        "camera": cfg.max_delta_camera,
    }

    for sensor in data:
        # Floors can raise values farther than max_delta when the
        # original prediction is below the configured safety floor.
        mask = data[sensor] >= (
            getattr(cfg, f"floor_{sensor}")
        )
        if np.any(mask):
            delta = np.abs(
                out[sensor][mask] - data[sensor][mask]
            )
            assert np.max(delta) <= limits[sensor] + 1e-12


def test_lidar_is_protected():
    n = 50
    data = {
        "gps": np.full(n, 0.90),
        "imu": np.full(n, 0.90),
        "lidar": np.full(n, 0.20),
        "camera": np.full(n, 0.90),
    }

    out = coordinate_reliabilities(data)

    assert np.all(out["lidar"] >= 0.72)


def test_length_mismatch_raises():
    data = _sample()
    data["camera"] = data["camera"][:-1]

    try:
        coordinate_reliabilities(data)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for length mismatch.")
