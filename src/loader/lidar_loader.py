from pathlib import Path

import numpy as np

from src.common.time_utils import load_kitti_timestamps


class LidarLoader:
    """
    KITTI LiDAR Loader.

    Dataset structure:
        sequence_path/
        └── velodyne_points/
            ├── data/
            │   ├── 0000000000.bin
            │   ├── 0000000001.bin
            │   └── ...
            ├── timestamps.txt
            ├── timestamps_start.txt
            └── timestamps_end.txt

    Each KITTI LiDAR point is stored as:
        x, y, z, reflectance

    Returned point cloud shape:
        (N, 4)
    """

    def __init__(self, sequence_path):
        """
        Parameters
        ----------
        sequence_path : str or Path
            Path to the KITTI synchronized drive directory.
        """

        self.sequence_path = Path(sequence_path)

        self.lidar_path = (
            self.sequence_path / "velodyne_points"
        )

        self.data_path = (
            self.lidar_path / "data"
        )

        self.timestamp_path = (
            self.lidar_path / "timestamps.txt"
        )

        # --------------------------------------------------
        # Check directories and files
        # --------------------------------------------------

        if not self.sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence path not found: {self.sequence_path}"
            )

        if not self.lidar_path.exists():
            raise FileNotFoundError(
                f"LiDAR directory not found: {self.lidar_path}"
            )

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"LiDAR data directory not found: {self.data_path}"
            )

        if not self.timestamp_path.exists():
            raise FileNotFoundError(
                f"LiDAR timestamp file not found: "
                f"{self.timestamp_path}"
            )

        # --------------------------------------------------
        # Load LiDAR files
        # --------------------------------------------------

        self.files = sorted(
            self.data_path.glob("*.bin")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No LiDAR .bin files found in: {self.data_path}"
            )

        # --------------------------------------------------
        # Load timestamps
        # --------------------------------------------------

        self.timestamps = load_kitti_timestamps(
            self.timestamp_path
        )

        # --------------------------------------------------
        # Validate number of frames
        # --------------------------------------------------

        if len(self.files) != len(self.timestamps):
            raise RuntimeError(
                "LiDAR file/timestamp count mismatch: "
                f"{len(self.files)} files vs "
                f"{len(self.timestamps)} timestamps"
            )

    def __len__(self):
        """
        Return number of LiDAR frames.
        """
        return len(self.files)

    def __getitem__(self, index):
        """
        Load LiDAR frame by index.
        """
        return self.load(index)

    def load(self, index):
        """
        Load one LiDAR frame.

        Returns
        -------
        dict
            {
                "frame_id": int,
                "timestamp": datetime,
                "points": np.ndarray
            }

        points:
            shape = (N, 4)
            columns = [x, y, z, reflectance]
        """

        # --------------------------------------------------
        # Validate index
        # --------------------------------------------------

        if not isinstance(index, (int, np.integer)):
            raise TypeError(
                f"index must be an integer, got {type(index)}"
            )

        if index < 0:
            index += len(self.files)

        if index < 0 or index >= len(self.files):
            raise IndexError(
                f"LiDAR frame index out of range: {index}"
            )

        # --------------------------------------------------
        # Read binary point cloud
        # --------------------------------------------------

        file_path = self.files[index]

        points = np.fromfile(
            file_path,
            dtype=np.float32
        )

        # KITTI Velodyne format:
        # x, y, z, reflectance

        if points.size % 4 != 0:
            raise RuntimeError(
                f"Invalid LiDAR file format: {file_path}. "
                f"Expected number of float32 values to be "
                f"divisible by 4, got {points.size}."
            )

        points = points.reshape(-1, 4)

        # --------------------------------------------------
        # Return unified frame structure
        # --------------------------------------------------

        return {
            "frame_id": index,
            "timestamp": self.timestamps[index],
            "points": points,
        }
