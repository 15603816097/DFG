import numpy as np
import torch

from torch.utils.data import Dataset


from src.loader.kitti_loader import KITTILoader


from src.feature.imu_feature import IMUFeatureExtractor
from src.feature.gps_feature import GPSFeatureExtractor
from src.feature.lidar_feature import LidarFeatureExtractor
from src.feature.camera_feature import CameraFeatureExtractor
from src.feature.fusion_feature import FusionFeatureExtractor



class KITTIRawDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        window_size=20
    ):

        self.dataset_root = dataset_root

        self.window_size = window_size


        self.kitti = KITTILoader(
            dataset_root
        )


        self.length = len(self.kitti)



        self.imu_extractor = IMUFeatureExtractor()

        self.gps_extractor = GPSFeatureExtractor()

        self.lidar_extractor = LidarFeatureExtractor()

        self.camera_extractor = CameraFeatureExtractor()

        self.fusion_extractor = FusionFeatureExtractor()



    def __len__(self):

        return self.length - self.window_size



    def __getitem__(
        self,
        index
    ):


        if index < 0:

            index = len(self) + index



        if index < 0 or index >= len(self):

            raise IndexError(
                "Index out of range"
            )



        feature_sequence = []

        state_sequence = []



        for i in range(
            index,
            index + self.window_size
        ):


            frame = self.kitti[i]



            ##############################
            # IMU feature
            ##############################

            imu_data = self._get_imu(
                frame
            )


            imu_feature = self.imu_extractor.extract(
                imu_data
            )



            ##############################
            # GPS feature
            ##############################

            gps_data = self._get_gps(
                frame
            )


            gps_feature = self.gps_extractor.extract(
                gps_data
            )



            ##############################
            # LiDAR feature
            ##############################

            lidar_data = self._get_lidar(
                frame
            )


            lidar_feature = self.lidar_extractor.extract(
                lidar_data
            )



            ##############################
            # Camera feature
            ##############################

            image = self._get_camera(
                frame
            )


            camera_feature = self.camera_extractor.extract(
                image
            )



            ##############################
            # Fusion feature
            ##############################

            fusion_feature = self.fusion_extractor.extract(
                imu_feature,
                gps_feature,
                lidar_feature,
                camera_feature
            )



            feature_sequence.append(
                fusion_feature
            )



            ##############################
            # State
            ##############################

            state = self._get_state(
                frame
            )


            state_sequence.append(
                state
            )



        #################################
        # feature
        #################################

        feature = np.asarray(
            feature_sequence,
            dtype=np.float32
        )


        feature = np.nan_to_num(
            feature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        feature = torch.tensor(
            feature,
            dtype=torch.float32
        )



        #################################
        # state
        #################################

        # 当前窗口最后一帧作为预测目标

        state = np.asarray(
            state_sequence[-1],
            dtype=np.float32
        )


        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        state = torch.tensor(
            state,
            dtype=torch.float32
        )



        return {

            "feature": feature,

            "state": state

        }



    ##################################################
    # IMU
    ##################################################

    def _get_imu(
        self,
        frame
    ):


        imu = frame["imu"]



        acceleration = imu["acceleration"]

        angular_velocity = imu["angular_velocity"]



        data = np.concatenate(
            [
                acceleration,
                angular_velocity
            ]
        )



        return data.reshape(
            1,
            6
        )



    ##################################################
    # GPS
    ##################################################

    def _get_gps(
        self,
        frame
    ):


        gps = frame["gps"]



        position = gps["position"]



        return np.asarray(
            position,
            dtype=np.float32
        ).reshape(
            1,
            3
        )



    ##################################################
    # LiDAR
    ##################################################

    def _get_lidar(
        self,
        frame
    ):


        lidar = frame["lidar"]


        return lidar["points"]



    ##################################################
    # Camera
    ##################################################

    def _get_camera(
        self,
        frame
    ):


        camera = frame["camera"]


        return camera["image"]



    ##################################################
    # State
    ##################################################

    def _get_state(
        self,
        frame
    ):


        gps = frame["gps"]

        imu = frame["imu"]



        position = gps["position"]



        velocity = imu.get(
            "velocity",
            np.zeros(
                3,
                dtype=np.float32
            )
        )



        state = np.concatenate(
            [
                position,
                velocity
            ]
        )


        return state
