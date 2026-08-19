from src.preprocessing.synchronization import (
    SensorSynchronizer,
)


DATASET_ROOT = (
    "dataset/kitti/"
    "2011_10_03/"
    "2011_10_03_drive_0027_sync"
)


def test_synchronizer_length():
    """
    Test synchronization sequence length.
    """

    synchronizer = SensorSynchronizer(
        DATASET_ROOT,
        reference_sensor="imu",
    )

    print(
        "Reference sensor:",
        synchronizer.reference_sensor,
    )

    print(
        "Reference frames:",
        len(synchronizer),
    )

    assert len(synchronizer) > 0


def test_synchronize_first_frame():
    """
    Test synchronization of the first frame.
    """

    synchronizer = SensorSynchronizer(
        DATASET_ROOT,
        reference_sensor="imu",
    )

    result = synchronizer[0]

    print(
        "Reference frame:",
        result["reference_frame_id"],
    )

    print(
        "Reference timestamp:",
        result["reference_timestamp"],
    )

    print(
        "Synchronization result:",
        result,
    )

    assert result["reference_frame_id"] == 0

    assert (
        result["reference_sensor"]
        == "imu"
    )

    assert "sensors" in result

    assert "imu" in result["sensors"]

    assert "gps" in result["sensors"]

    assert "lidar" in result["sensors"]

    assert "camera" in result["sensors"]


def test_sensor_frame_ids():
    """
    Test that nearest sensor frames are valid indices.
    """

    synchronizer = SensorSynchronizer(
        DATASET_ROOT,
        reference_sensor="imu",
    )

    result = synchronizer[0]

    for sensor in (
        "imu",
        "gps",
        "lidar",
        "camera",
    ):

        frame_id = (
            result["sensors"][sensor]["frame_id"]
        )

        assert frame_id >= 0

        assert frame_id < len(
            synchronizer.loader
        )


def test_time_offsets():
    """
    Test timestamp offset calculation.
    """

    synchronizer = SensorSynchronizer(
        DATASET_ROOT,
        reference_sensor="imu",
    )

    offsets = (
        synchronizer.get_time_offsets(0)
    )

    print(
        "Time offsets:",
        offsets,
    )

    assert "imu" in offsets

    assert "gps" in offsets

    assert "lidar" in offsets

    assert "camera" in offsets

    # Reference sensor compared with itself
    # should have zero offset.
    assert offsets["imu"] == 0.0


def test_absolute_time_offsets():
    """
    Test absolute timestamp offset calculation.
    """

    synchronizer = SensorSynchronizer(
        DATASET_ROOT,
        reference_sensor="imu",
    )

    offsets = (
        synchronizer.get_absolute_time_offsets(0)
    )

    for sensor in (
        "imu",
        "gps",
        "lidar",
        "camera",
    ):

        assert offsets[sensor] >= 0.0


def test_sync_quality():
    """
    Test synchronization quality calculation.
    """

    synchronizer = SensorSynchronizer(
        DATASET_ROOT,
        reference_sensor="imu",
    )

    quality = (
        synchronizer.get_sync_quality(0)
    )

    print(
        "Synchronization quality:",
        quality,
    )

    assert "mean_abs_dt" in quality

    assert "max_abs_dt" in quality

    assert "valid" in quality

    assert quality["mean_abs_dt"] >= 0.0

    assert quality["max_abs_dt"] >= 0.0


def test_negative_index():
    """
    Test negative frame indexing.
    """

    synchronizer = SensorSynchronizer(
        DATASET_ROOT,
        reference_sensor="imu",
    )

    result = synchronizer[-1]

    assert (
        result["reference_frame_id"]
        == len(synchronizer) - 1
    )


def test_invalid_reference_sensor():
    """
    Test invalid reference sensor handling.
    """

    try:

        SensorSynchronizer(
            DATASET_ROOT,
            reference_sensor="invalid",
        )

        assert False, (
            "Invalid reference sensor "
            "should raise ValueError"
        )

    except ValueError:

        pass
