from types import SimpleNamespace

from src.factor_graph.gps_z_soft_dynamic_covariance import (
    GPSZSoftPolicy,
    gps_dynamic_sigma,
    gps_z_soft_sigma,
)


def make_config():
    return SimpleNamespace(
        gps_fixed_sigma=5.0,
        gps_sigma_min=3.0,
        gps_sigma_max=30.0,
        gps_gamma=3.0,
    )


def test_alpha_zero_equals_fixed():
    config = make_config()

    sigma = gps_z_soft_sigma(
        config,
        reliability=0.1,
        alpha_z=0.0,
    )

    assert abs(
        sigma - config.gps_fixed_sigma
    ) < 1e-12


def test_alpha_one_equals_dynamic():
    config = make_config()

    expected = gps_dynamic_sigma(
        config,
        reliability=0.1,
    )

    sigma = gps_z_soft_sigma(
        config,
        reliability=0.1,
        alpha_z=1.0,
    )

    assert abs(
        sigma - expected
    ) < 1e-12


def test_half_is_linear_midpoint():
    config = make_config()

    dynamic = gps_dynamic_sigma(
        config,
        reliability=0.1,
    )

    sigma = gps_z_soft_sigma(
        config,
        reliability=0.1,
        alpha_z=0.5,
    )

    expected = (
        0.5 * config.gps_fixed_sigma
        + 0.5 * dynamic
    )

    assert abs(
        sigma - expected
    ) < 1e-12


def test_invalid_alpha_rejected():
    failed = False

    try:
        GPSZSoftPolicy(
            alpha_z=1.5
        )

    except ValueError:
        failed = True

    assert failed
