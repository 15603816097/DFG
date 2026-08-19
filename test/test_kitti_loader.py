from src.loader.kitti_loader import KITTILoader


DATASET_ROOT = (
    "dataset/kitti/"
    "2011_10_03/"
    "2011_10_03_drive_0027_sync"
)


def test_kitti_length():
    """
    Test unified dataset length.
    """

    loader = KITTILoader(
        DATASET_ROOT,
        camera_id="image_02",
    )

    print(
        "Dataset frames:",
        len(loader)
    )

    assert len(loader) > 0


def test_kitti_first_frame():
    """
    Test loading the first synchronized frame.
    """

    loader = KITTILoader(
        DATASET_ROOT,
        camera_id="image_02",
    )

    frame = loader[0]

    print(
        "Frame ID:",
        frame["frame_id"]
    )

    print(
        "Timestamp:",
        frame["timestamp"]
    )

    print(
        "IMU:",
        frame["imu"]
    )

    print(
        "GPS:",
        frame["gps"]
    )

    print(
        "LiDAR shape:",
        frame["lidar"]["points"].shape
    )

    print(
        "Camera shape:",
        frame["camera"]["image"].shape
    )

    assert frame["frame_id"] == 0

    assert "timestamp" in frame

    assert "imu" in frame

    assert "gps" in frame

    assert "lidar" in frame

    assert "camera" in frame


def test_kitti_frame_consistency():
    """
    Test whether all sensor frame IDs are consistent.
    """

    loader = KITTILoader(
        DATASET_ROOT
    )

    frame = loader[0]

    frame_id = frame["frame_id"]

    assert frame["imu"]["frame_id"] == frame_id

    assert frame["gps"]["frame_id"] == frame_id

    assert frame["lidar"]["frame_id"] == frame_id

    assert frame["camera"]["frame_id"] == frame_id


def test_kitti_sensor_access():
    """
    Test individual sensor access methods.
    """

    loader = KITTILoader(
        DATASET_ROOT
    )

    imu = loader.get_imu(0)

    gps = loader.get_gps(0)

    lidar = loader.get_lidar(0)

    camera = loader.get_camera(0)

    assert imu["frame_id"] == 0

    assert gps["frame_id"] == 0

    assert lidar["frame_id"] == 0

    assert camera["frame_id"] == 0


def test_kitti_negative_index():
    """
    Test negative indexing.
    """

    loader = KITTILoader(
        DATASET_ROOT
    )

    frame = loader[-1]

    assert (
        frame["frame_id"]
        == len(loader) - 1
    )


def test_kitti_timestamp_sequence():
    """
    Test whether timestamps are available.
    """

    loader = KITTILoader(
        DATASET_ROOT
    )

    timestamps = loader.get_timestamps()

    assert len(timestamps) == len(loader)

    assert timestamps[0] is not None
