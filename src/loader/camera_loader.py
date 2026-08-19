from pathlib import Path

import cv2

from src.common.time_utils import load_kitti_timestamps


class CameraLoader:
    """
    KITTI Camera Loader.

    Dataset structure:
        sequence_path/
        ├── image_00/
        │   ├── data/
        │   │   ├── 0000000000.png
        │   │   ├── 0000000001.png
        │   │   └── ...
        │   └── timestamps.txt
        │
        ├── image_01/
        ├── image_02/
        └── image_03/

    Parameters
    ----------
    sequence_path : str or Path
        Path to the KITTI synchronized drive directory.

    camera_id : str
        Camera directory name.
        Supported:
            image_00
            image_01
            image_02
            image_03

        Default:
            image_02
    """

    VALID_CAMERA_IDS = {
        "image_00",
        "image_01",
        "image_02",
        "image_03",
    }

    def __init__(self, sequence_path, camera_id="image_02"):
        """
        Initialize CameraLoader.
        """

        self.sequence_path = Path(sequence_path)

        self.camera_id = str(camera_id)

        # --------------------------------------------------
        # Validate camera ID
        # --------------------------------------------------

        if self.camera_id not in self.VALID_CAMERA_IDS:
            raise ValueError(
                f"Invalid camera_id: {self.camera_id}. "
                f"Expected one of: "
                f"{sorted(self.VALID_CAMERA_IDS)}"
            )

        # --------------------------------------------------
        # Camera paths
        # --------------------------------------------------

        self.camera_path = (
            self.sequence_path / self.camera_id
        )

        self.data_path = (
            self.camera_path / "data"
        )

        self.timestamp_path = (
            self.camera_path / "timestamps.txt"
        )

        # --------------------------------------------------
        # Check paths
        # --------------------------------------------------

        if not self.sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence path not found: "
                f"{self.sequence_path}"
            )

        if not self.camera_path.exists():
            raise FileNotFoundError(
                f"Camera directory not found: "
                f"{self.camera_path}"
            )

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Camera data directory not found: "
                f"{self.data_path}"
            )

        if not self.timestamp_path.exists():
            raise FileNotFoundError(
                f"Camera timestamp file not found: "
                f"{self.timestamp_path}"
            )

        # --------------------------------------------------
        # Find image files
        # --------------------------------------------------

        self.files = sorted(
            self.data_path.glob("*.png")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No PNG images found in: "
                f"{self.data_path}"
            )

        # --------------------------------------------------
        # Load timestamps
        # --------------------------------------------------

        self.timestamps = load_kitti_timestamps(
            self.timestamp_path
        )

        # --------------------------------------------------
        # Validate frame count
        # --------------------------------------------------

        if len(self.files) != len(self.timestamps):
            raise RuntimeError(
                "Camera file/timestamp count mismatch: "
                f"{len(self.files)} images vs "
                f"{len(self.timestamps)} timestamps"
            )

    def __len__(self):
        """
        Return number of camera frames.
        """

        return len(self.files)

    def __getitem__(self, index):
        """
        Load camera frame by index.
        """

        return self.load(index)

    def load(self, index):
        """
        Load one camera image.

        Returns
        -------
        dict
            {
                "frame_id": int,
                "timestamp": datetime,
                "image": np.ndarray,
                "camera_id": str
            }

        image:
            OpenCV BGR image with shape (H, W, 3)
        """

        # --------------------------------------------------
        # Validate index
        # --------------------------------------------------

        if not isinstance(index, int):
            raise TypeError(
                f"index must be an integer, "
                f"got {type(index)}"
            )

        # Support negative indexing
        if index < 0:
            index += len(self.files)

        if index < 0 or index >= len(self.files):
            raise IndexError(
                f"Camera frame index out of range: {index}"
            )

        # --------------------------------------------------
        # Image path
        # --------------------------------------------------

        image_path = self.files[index]

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to read camera image: "
                f"{image_path}"
            )

        # --------------------------------------------------
        # Return unified frame structure
        # --------------------------------------------------

        return {
            "frame_id": index,
            "timestamp": self.timestamps[index],
            "camera_id": self.camera_id,
            "image": image,
        }
