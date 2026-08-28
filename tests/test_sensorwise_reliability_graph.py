from src.factor_graph.sensorwise_reliability_graph import (
    ReliabilitySource,
)


def test_valid_modes():
    assert (
        ReliabilitySource(
            "fixed"
        ).mode
        ==
        "fixed"
    )

    assert (
        ReliabilitySource(
            "predictive"
        ).mode
        ==
        "predictive"
    )

    assert (
        ReliabilitySource(
            "oracle"
        ).mode
        ==
        "oracle"
    )


def test_invalid_mode():
    failed = False

    try:
        ReliabilitySource(
            "bad_mode"
        )

    except ValueError:
        failed = True

    assert failed
