from pathlib import Path

from src.loader.imu_loader import IMULoader
from src.loader.gps_loader import GPSLoader
from src.loader.lidar_loader import LidarLoader
from src.loader.camera_loader import CameraLoader


class KITTILoader:
    """
    Unified KITTI multi-sensor loader.

    Dataset structure:

        dataset/kitti/
        └── 2011_10_03/
            ├── calib_cam_to_cam.txt
            ├── calib_imu_to_velo.txt
            ├── calib_velo_to_cam.txt
            │
            └── 2011_10_03_drive_0027_sync/
                ├── image_00/
                ├── image_01/
                ├── image_02/
                ├── image_03/
                ├── oxts/
                └── velodyne_points/

    This class provides one unified interface for:

        IMU
        GPS
        LiDAR
        Camera

    Example
    -------
    loader = KITTILoader(
        "dataset/kitti/2011_10_03/"
        "2011_10_03_drive_0027_sync"
    )

    frame = loader[0]
    """

    def __init__(
        self,
        sequence_path,
        camera_id="image_02",
    ):
        """
        Parameters
        ----------
        sequence_path : str or Path
            Path to the KITTI synchronized drive.

        camera_id : str
            Camera used by the unified loader.
            Default: image_02
        """

        self.sequence_path = Path(sequence_path)

        if not self.sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence path not found: "
                f"{self.sequence_path}"
            )

        if not self.sequence_path.is_dir():
            raise NotADirectoryError(
                f"Sequence path is not a directory: "
                f"{self.sequence_path}"
            )

        self.camera_id = camera_id

        # --------------------------------------------------
        # Initialize sensor loaders
        # --------------------------------------------------

        self.imu_loader = IMULoader(
            self.sequence_path
        )

        self.gps_loader = GPSLoader(
            self.sequence_path
        )

        self.lidar_loader = LidarLoader(
            self.sequence_path
        )

        self.camera_loader = CameraLoader(
            self.sequence_path,
            camera_id=self.camera_id,
        )

        # --------------------------------------------------
        # Check sensor frame counts
        # --------------------------------------------------

        self._validate_frame_counts()

    def _validate_frame_counts(self):
        """
        Validate whether all sensor sequences contain
        the same number of frames.

        KITTI synchronized sequences are expected to have
        aligned frame indices for these sensors.
        """

        counts = {
            "imu": len(self.imu_loader),
            "gps": len(self.gps_loader),
            "lidar": len(self.lidar_loader),
            "camera": len(self.camera_loader),
        }

        unique_counts = set(
            counts.values()
        )

        if len(unique_counts) != 1:
            raise RuntimeError(
                "Sensor frame count mismatch: "
                f"{counts}"
            )

        self._length = next(
            iter(unique_counts)
        )

    def __len__(self):
        """
        Return the number of synchronized frames.
        """

        return self._length

    def __getitem__(self, index):
        """
        Load one synchronized multi-sensor frame.

        Returns
        -------
        dict

        {
            "frame_id": int,
            "timestamp": datetime,

            "imu": {
                ...
            },

            "gps": {
                ...
            },

            "lidar": {
                ...
            },

            "camera": {
                ...
            }
        }
        """

        return self.load(index)

    def _normalize_index(self, index):
        """
        Normalize and validate frame index.

        Supports negative indexing.
        """

        if not isinstance(index, int):
            raise TypeError(
                f"index must be an integer, "
                f"got {type(index)}"
            )

        if index < 0:
            index += self._length

        if index < 0 or index >= self._length:
            raise IndexError(
                f"Frame index out of range: {index}"
            )

        return index

    def load(self, index):
        """
        Load one synchronized frame.
        """

        index = self._normalize_index(index)

        # --------------------------------------------------
        # Load each sensor
        # --------------------------------------------------

        imu = self.imu_loader[index]

        gps = self.gps_loader[index]

        lidar = self.lidar_loader[index]

        camera = self.camera_loader[index]

        # --------------------------------------------------
        # Validate frame IDs
        # --------------------------------------------------

        frame_ids = {
            "imu": imu["frame_id"],
            "gps": gps["frame_id"],
            "lidar": lidar["frame_id"],
            "camera": camera["frame_id"],
        }

        if len(set(frame_ids.values())) != 1:
            raise RuntimeError(
                "Sensor frame ID mismatch: "
                f"{frame_ids}"
            )

        # --------------------------------------------------
        # Use IMU timestamp as the reference timestamp
        # --------------------------------------------------

        reference_timestamp = imu["timestamp"]

        # --------------------------------------------------
        # Build unified frame
        # --------------------------------------------------

        frame = {
            "frame_id": index,

            "timestamp": reference_timestamp,

            "imu": imu,

            "gps": gps,

            "lidar": lidar,

            "camera": camera,
        }

        return frame

    def get_imu(self, index):
        """
        Get IMU data for a frame.
        """

        index = self._normalize_index(index)

        return self.imu_loader[index]

    def get_gps(self, index):
        """
        Get GPS data for a frame.
        """

        index = self._normalize_index(index)

        return self.gps_loader[index]

    def get_lidar(self, index):
        """
        Get LiDAR data for a frame.
        """

        index = self._normalize_index(index)

        return self.lidar_loader[index]

    def get_camera(self, index):
        """
        Get camera data for a frame.
        """

        index = self._normalize_index(index)

        return self.camera_loader[index]

    def get_timestamps(self):
        """
        Return the timestamps of the sequence.

        The IMU timestamps are used as the reference
        timestamps.
        """

        return list(
            self.imu_loader.timestamps
        )
