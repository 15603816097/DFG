import numpy as np

from src.multisensor_reliability.degradation.balanced_plans import (
    create_balanced_multisensor_degradation_plan,
)


SENSORS = (
    "gps",
    "imu",
    "lidar",
    "camera",
)


def test_shapes():
    plan = (
        create_balanced_multisensor_degradation_plan(
            1000
        )
    )

    for sensor in SENSORS:
        assert (
            plan[
                "severity"
            ][
                sensor
            ].shape
            ==
            (
                1000,
            )
        )

        assert (
            plan[
                "mode"
            ][
                sensor
            ].shape
            ==
            (
                1000,
            )
        )


def test_each_split_contains_degradation():
    plan = (
        create_balanced_multisensor_degradation_plan(
            1000
        )
    )

    for split_name, (
        start,
        end,
    ) in plan[
        "split_bounds"
    ].items():

        for sensor in SENSORS:
            severity = plan[
                "severity"
            ][
                sensor
            ][
                start:end
            ]

            assert np.max(
                severity
            ) >= 0.95

            assert np.mean(
                severity
                >
                0.0
            ) > 0.05

            assert np.mean(
                severity
                <
                0.05
            ) > 0.05


def test_reproducible():
    a = (
        create_balanced_multisensor_degradation_plan(
            1000,
            seed=123,
        )
    )

    b = (
        create_balanced_multisensor_degradation_plan(
            1000,
            seed=123,
        )
    )

    for sensor in SENSORS:
        assert np.allclose(
            a[
                "severity"
            ][
                sensor
            ],
            b[
                "severity"
            ][
                sensor
            ],
        )
