from pathlib import Path
from bisect import bisect_left

import numpy as np

from src.loader import KITTILoader


class SensorSynchronizer:
    """
    Multi-sensor timestamp synchronizer for KITTI.

    The synchronizer uses one sensor as the reference timeline
    and finds the nearest timestamp from other sensors.

    Supported sensors:
        - imu
        - gps
        - lidar
        - camera

    Example
    -------
    synchronizer = SensorSynchronizer(
        "dataset/kitti/2011_10_03/"
        "2011_10_03_drive_0027_sync"
    )

    frame = synchronizer[0]
    """

    SUPPORTED_SENSORS = (
        "imu",
        "gps",
        "lidar",
        "camera",
    )

    def __init__(
        self,
        sequence_path,
        camera_id="image_02",
        reference_sensor="imu",
        max_time_diff=0.05,
    ):
        """
        Parameters
        ----------
        sequence_path : str or Path
            KITTI synchronized drive path.

        camera_id : str
            Camera used for synchronization.

        reference_sensor : str
            Reference sensor timeline.
            Default: "imu".

        max_time_diff : float
            Maximum allowed timestamp difference in seconds.

            For example:
                0.05 = 50 ms

            If the nearest sensor frame is farther than this
            threshold, the synchronization result is marked
            invalid.
        """

        self.sequence_path = Path(sequence_path)

        if not self.sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence path not found: "
                f"{self.sequence_path}"
            )

        # --------------------------------------------------
        # Validate reference sensor
        # --------------------------------------------------

        if reference_sensor not in self.SUPPORTED_SENSORS:
            raise ValueError(
                f"Invalid reference_sensor: "
                f"{reference_sensor}. "
                f"Expected one of: "
                f"{self.SUPPORTED_SENSORS}"
            )

        self.reference_sensor = reference_sensor

        # --------------------------------------------------
        # Validate threshold
        # --------------------------------------------------

        if max_time_diff < 0:
            raise ValueError(
                "max_time_diff must be >= 0"
            )

        self.max_time_diff = float(
            max_time_diff
        )

        # --------------------------------------------------
        # Unified KITTI loader
        # --------------------------------------------------

        self.loader = KITTILoader(
            self.sequence_path,
            camera_id=camera_id,
        )

        # --------------------------------------------------
        # Store timestamps
        # --------------------------------------------------

        self.timestamps = {
            "imu": list(
                self.loader.imu_loader.timestamps
            ),

            "gps": list(
                self.loader.gps_loader.timestamps
            ),

            "lidar": list(
                self.loader.lidar_loader.timestamps
            ),

            "camera": list(
                self.loader.camera_loader.timestamps
            ),
        }

        # --------------------------------------------------
        # Convert timestamps to seconds
        # --------------------------------------------------

        self.timestamp_seconds = {
            sensor: self._timestamps_to_seconds(
                timestamps
            )
            for sensor, timestamps
            in self.timestamps.items()
        }

        # --------------------------------------------------
        # Validate timestamp sequences
        # --------------------------------------------------

        for sensor, values in self.timestamp_seconds.items():

            if len(values) == 0:
                raise RuntimeError(
                    f"No timestamps found for "
                    f"sensor: {sensor}"
                )

            if not self._is_monotonic(values):
                raise RuntimeError(
                    f"Timestamps are not monotonically "
                    f"increasing for sensor: {sensor}"
                )

        # --------------------------------------------------
        # Reference timestamps
        # --------------------------------------------------

        self.reference_timestamps = (
            self.timestamps[self.reference_sensor]
        )

        self.reference_seconds = (
            self.timestamp_seconds[
                self.reference_sensor
            ]
        )

    # ======================================================
    # Basic utilities
    # ======================================================

    @staticmethod
    def _timestamps_to_seconds(timestamps):
        """
        Convert datetime timestamps to relative seconds.

        The first timestamp is used as the zero point.

        This avoids dealing with large Unix timestamps and
        is sufficient for relative synchronization.
        """

        if len(timestamps) == 0:
            return []

        first = timestamps[0]

        return [
            (timestamp - first).total_seconds()
            for timestamp in timestamps
        ]

    @staticmethod
    def _is_monotonic(values):
        """
        Check whether timestamps are monotonically increasing.
        """

        if len(values) <= 1:
            return True

        return all(
            values[i] >= values[i - 1]
            for i in range(1, len(values))
        )

    @staticmethod
    def _nearest_index(
        timestamps,
        target_time,
    ):
        """
        Find the index of the timestamp nearest to target_time.

        Parameters
        ----------
        timestamps : list[float]
            Sorted timestamp sequence.

        target_time : float
            Target timestamp in seconds.

        Returns
        -------
        int
            Index of nearest timestamp.
        """

        if len(timestamps) == 0:
            raise ValueError(
                "Timestamp sequence is empty"
            )

        right = bisect_left(
            timestamps,
            target_time,
        )

        # Target is before the first timestamp.
        if right == 0:
            return 0

        # Target is after the last timestamp.
        if right == len(timestamps):
            return len(timestamps) - 1

        left = right - 1

        left_diff = abs(
            timestamps[left] - target_time
        )

        right_diff = abs(
            timestamps[right] - target_time
        )

        if left_diff <= right_diff:
            return left

        return right

    # ======================================================
    # Sensor matching
    # ======================================================

    def find_nearest(
        self,
        sensor,
        target_timestamp,
    ):
        """
        Find the sensor frame nearest to target_timestamp.

        Returns
        -------
        dict

        {
            "frame_id": int,
            "timestamp": datetime,
            "dt": float,
            "valid": bool
        }
        """

        if sensor not in self.SUPPORTED_SENSORS:
            raise ValueError(
                f"Unsupported sensor: {sensor}"
            )

        target_seconds = (
            target_timestamp
            - self.reference_timestamps[0]
        ).total_seconds()

        sensor_times = (
            self.timestamp_seconds[sensor]
        )

        index = self._nearest_index(
            sensor_times,
            target_seconds,
        )

        matched_timestamp = (
            self.timestamps[sensor][index]
        )

        dt = (
            matched_timestamp
            - target_timestamp
        ).total_seconds()
        
        return {
            "frame_id": index,
            "timestamp": matched_timestamp,
            "dt": dt,
            "abs_dt": abs(dt),
            "valid": abs(dt) <= self.max_time_diff,
        }

    # ======================================================
    # Frame synchronization
    # ======================================================

    def synchronize_frame(self, reference_index):
        """
        Synchronize all sensors for one reference frame.

        Parameters
        ----------
        reference_index : int
            Index on the reference sensor timeline.

        Returns
        -------
        dict
            Synchronization information for all sensors.
        """

        if not isinstance(
            reference_index,
            (int, np.integer),
        ):
            raise TypeError(
                "reference_index must be an integer"
            )

        reference_index = int(
            reference_index
        )

        if reference_index < 0:
            reference_index += len(
                self.reference_timestamps
            )

        if (
            reference_index < 0
            or reference_index
            >= len(self.reference_timestamps)
        ):
            raise IndexError(
                f"Reference frame index out of range: "
                f"{reference_index}"
            )

        reference_timestamp = (
            self.reference_timestamps[
                reference_index
            ]
        )

        result = {
            "reference_sensor":
                self.reference_sensor,

            "reference_frame_id":
                reference_index,

            "reference_timestamp":
                reference_timestamp,

            "sensors": {},
        }

        # --------------------------------------------------
        # Find nearest frame for every sensor
        # --------------------------------------------------

        for sensor in self.SUPPORTED_SENSORS:

            match = self.find_nearest(
                sensor,
                reference_timestamp,
            )

            result["sensors"][sensor] = match

        # --------------------------------------------------
        # Overall validity
        # --------------------------------------------------

        result["valid"] = all(
            sensor_info["valid"]
            for sensor_info
            in result["sensors"].values()
        )

        return result

    def __getitem__(self, index):
        """
        Support:

            synchronizer[0]
        """

        return self.synchronize_frame(index)

    def __len__(self):
        """
        Number of frames on the reference timeline.
        """

        return len(
            self.reference_timestamps
        )

    # ======================================================
    # Synchronization statistics
    # ======================================================

    def get_time_offsets(self, index):
        """
        Return absolute time offsets for one frame.

        Returns
        -------
        dict

        Example:
            {
                "imu": 0.0,
                "gps": 0.001,
                "lidar": 0.004,
                "camera": 0.012
            }
        """

        result = self.synchronize_frame(
            index
        )

        return {
            sensor:
                info["dt"]
            for sensor, info
            in result["sensors"].items()
        }

    def get_absolute_time_offsets(self, index):
        """
        Return absolute synchronization errors.

        Returns
        -------
        dict
        """

        result = self.synchronize_frame(
            index
        )

        return {
            sensor:
                info["abs_dt"]
            for sensor, info
            in result["sensors"].items()
        }

    def get_sync_quality(self, index):
        """
        Calculate a simple synchronization quality score.

        The score is based on the mean absolute timestamp
        difference among all sensors.

        Returns
        -------
        dict

        {
            "mean_abs_dt": float,
            "max_abs_dt": float,
            "valid": bool
        }
        """

        result = self.synchronize_frame(
            index
        )

        offsets = [
            info["abs_dt"]
            for info
            in result["sensors"].values()
        ]

        mean_abs_dt = float(
            np.mean(offsets)
        )

        max_abs_dt = float(
            np.max(offsets)
        )

        return {
            "mean_abs_dt": mean_abs_dt,
            "max_abs_dt": max_abs_dt,
            "valid": result["valid"],
        }
