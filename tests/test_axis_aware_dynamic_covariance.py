from types import SimpleNamespace

import numpy as np

from src.factor_graph.axis_aware_dynamic_covariance import (
    camera_axis_sigmas,
    get_policy,
    gps_axis_sigmas,
    imu_axis_sigmas,
)


def make_config():
    return SimpleNamespace(
        gps_fixed_sigma=5.0,
        gps_sigma_min=3.0,
        gps_sigma_max=30.0,
        gps_gamma=3.0,

        imu_rotation_sigma=0.03,
        imu_rot_sigma_min=0.01,
        imu_rot_sigma_max=0.20,
        imu_gamma=3.0,

        camera_rotation_sigma=0.03,
        camera_translation_sigma=0.45,
        camera_rot_sigma_min=0.02,
        camera_rot_sigma_max=0.20,
        camera_trans_sigma_min=0.20,
        camera_trans_sigma_max=2.00,
        camera_gamma=3.0,
    )


def test_gps_z_fixed_policy():
    config = make_config()

    policy = get_policy(
        "gps_z_fixed"
    )

    sx, sy, sz = gps_axis_sigmas(
        config,
        reliability=0.1,
        policy=policy,
    )

    assert sx != config.gps_fixed_sigma
    assert sy != config.gps_fixed_sigma
    assert sz == config.gps_fixed_sigma


def test_camera_z_fixed_policy():
    config = make_config()

    policy = get_policy(
        "camera_z_fixed"
    )

    values = camera_axis_sigmas(
        config,
        reliability=0.1,
        policy=policy,
    )

    assert values[3] != config.camera_translation_sigma
    assert values[4] != config.camera_translation_sigma
    assert values[5] == config.camera_translation_sigma


def test_camera_rpz_fixed_policy():
    config = make_config()

    policy = get_policy(
        "camera_rpz_fixed"
    )

    values = camera_axis_sigmas(
        config,
        reliability=0.1,
        policy=policy,
    )

    # roll / pitch fixed
    assert values[0] == config.camera_rotation_sigma
    assert values[1] == config.camera_rotation_sigma

    # yaw dynamic
    assert values[2] != config.camera_rotation_sigma

    # x/y dynamic, z fixed
    assert values[3] != config.camera_translation_sigma
    assert values[4] != config.camera_translation_sigma
    assert values[5] == config.camera_translation_sigma


def test_full_axis_aware_imu_only_yaw_dynamic():
    config = make_config()

    policy = get_policy(
        "full_axis_aware"
    )

    roll, pitch, yaw = imu_axis_sigmas(
        config,
        reliability=0.1,
        policy=policy,
    )

    assert roll == config.imu_rotation_sigma
    assert pitch == config.imu_rotation_sigma
    assert yaw != config.imu_rotation_sigma
