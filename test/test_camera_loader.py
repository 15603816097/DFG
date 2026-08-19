from src.loader.camera_loader import CameraLoader


DATASET_ROOT = (
    "dataset/kitti/"
    "2011_10_03/"
    "2011_10_03_drive_0027_sync"
)


def test_camera_length():
    """
    Test whether camera frames can be counted correctly.
    """

    loader = CameraLoader(
        DATASET_ROOT,
        camera_id="image_02",
    )

    print(
        "Camera:",
        loader.camera_id
    )

    print(
        "Camera frames:",
        len(loader)
    )

    assert len(loader) > 0


def test_load_first_image():
    """
    Test loading the first camera image.
    """

    loader = CameraLoader(
        DATASET_ROOT,
        camera_id="image_02",
    )

    image_data = loader.load(0)

    print(
        "Frame ID:",
        image_data["frame_id"]
    )

    print(
        "Timestamp:",
        image_data["timestamp"]
    )

    print(
        "Camera ID:",
        image_data["camera_id"]
    )

    print(
        "Image shape:",
        image_data["image"].shape
    )

    assert image_data["frame_id"] == 0

    assert image_data["camera_id"] == "image_02"

    assert image_data["image"] is not None

    assert image_data["image"].ndim == 3

    assert image_data["image"].shape[2] == 3


def test_camera_negative_index():
    """
    Test Python-style negative indexing.
    """

    loader = CameraLoader(
        DATASET_ROOT,
        camera_id="image_02",
    )

    image_data = loader[-1]

    assert image_data["frame_id"] == len(loader) - 1


def test_camera_invalid_id():
    """
    Test invalid camera ID handling.
    """

    try:
        CameraLoader(
            DATASET_ROOT,
            camera_id="image_04",
        )

        assert False, (
            "CameraLoader should reject "
            "an invalid camera_id"
        )

    except ValueError:
        pass


def test_camera_image_exists():
    """
    Test that the first image can actually be read.
    """

    loader = CameraLoader(
        DATASET_ROOT,
        camera_id="image_02",
    )

    image_data = loader[0]

    image = image_data["image"]

    assert image.size > 0

    assert image.shape[0] > 0

    assert image.shape[1] > 0
