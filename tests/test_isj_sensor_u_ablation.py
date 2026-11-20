import numpy as np
import pytest

from experiments.isj_supplement.run_sensor_uncertainty_ablation import (
    SENSORS,
    MODELS,
    available_specs,
    case_name,
    select_uncertainty,
    target_sensors,
)
from experiments.isj_supplement.run_uniform_convergence import available_cases


def test_drop_one_sensor_retains_other_sensors():
    for sensor in SENSORS:
        active = target_sensors("drop", sensor)
        assert sensor not in active
        assert len(active) == 2


def test_only_one_sensor_retains_just_itself():
    for sensor in SENSORS:
        assert target_sensors("only", sensor) == {sensor}


def test_uncertainty_intervention_does_not_mutate_input():
    u = np.array([0.2, 0.6, 1.0])
    original = u.copy()
    active = target_sensors("drop", "camera")
    np.testing.assert_array_equal(
        select_uncertainty("camera", u, active), np.zeros_like(u)
    )
    np.testing.assert_array_equal(
        select_uncertainty("gps", u, active), original
    )
    np.testing.assert_array_equal(u, original)


def test_sensor_u_case_names_and_convergence_specs():
    assert case_name("GRU", "drop", "camera") == "ISJ_SENSU_GRU_DROP_CAMERA"
    assert case_name("MAMBA", "only", "imu") == "ISJ_SENSU_MAMBA_ONLY_IMU"
    specs = available_specs("all")
    assert len(specs) == len(MODELS) * 2 * len(SENSORS)
    assert len({x["name"] for x in specs}) == 12
    conv = available_cases("sensor_u_drop")
    assert len(conv) == 6
    assert conv == available_specs("drop")


def test_unknown_sensor_strategy_rejected():
    with pytest.raises(ValueError):
        target_sensors("invalid", "gps")
    with pytest.raises(ValueError):
        select_uncertainty("lidar", np.ones(4), frozenset(SENSORS))
    with pytest.raises(ValueError):
        select_uncertainty("gps", np.array([0.1, float("nan")]),
                           frozenset(SENSORS))
