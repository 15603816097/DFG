from types import SimpleNamespace

from src.factor_graph.lidar_event_triggered_covariance import (
    lidar_event_sigma,
)


def make_config():
    return SimpleNamespace(
        lidar_rotation_sigma=0.03,
        lidar_translation_sigma=0.35,

        lidar_rot_sigma_min=0.02,
        lidar_rot_sigma_max=0.20,

        lidar_trans_sigma_min=0.20,
        lidar_trans_sigma_max=2.00,

        lidar_gamma=3.0,
    )


def test_healthy_is_exactly_fixed():
    config = make_config()

    rot, trans = lidar_event_sigma(
        config,
        trigger_active=False,
        event_reliability=0.1,
    )

    assert rot == config.lidar_rotation_sigma
    assert trans == config.lidar_translation_sigma


def test_triggered_inflates_covariance():
    config = make_config()

    rot, trans = lidar_event_sigma(
        config,
        trigger_active=True,
        event_reliability=0.15,
    )

    assert rot > config.lidar_rotation_sigma
    assert trans > config.lidar_translation_sigma
