from src.loader.lidar_loader import LidarLoader


DATASET_ROOT = (
    "dataset/kitti/"
    "2011_10_03/"
    "2011_10_03_drive_0027_sync"
)


def test_lidar_length():
    """
    Test whether LiDAR frames can be counted correctly.
    """

    loader = LidarLoader(
        DATASET_ROOT
    )

    print(
        "LiDAR frames:",
        len(loader)
    )

    assert len(loader) > 0


def test_load_first_lidar():
    """
    Test loading the first LiDAR frame.
    """

    loader = LidarLoader(
        DATASET_ROOT
    )

    lidar = loader[0]

    print(
        "Frame ID:",
        lidar["frame_id"]
    )

    print(
        "Timestamp:",
        lidar["timestamp"]
    )

    print(
        "Point cloud shape:",
        lidar["points"].shape
    )

    print(
        "First point:",
        lidar["points"][0]
    )

    assert "frame_id" in lidar
    assert "timestamp" in lidar
    assert "points" in lidar

    assert lidar["frame_id"] == 0

    assert lidar["points"].ndim == 2

    assert lidar["points"].shape[1] == 4


def test_lidar_point_cloud_values():
    """
    Test whether the LiDAR point cloud contains
    valid floating-point values.
    """

    loader = LidarLoader(
        DATASET_ROOT
    )

    lidar = loader[0]

    points = lidar["points"]

    assert points.dtype == "float32"

    assert points.shape[0] > 0

    assert points.shape[1] == 4
