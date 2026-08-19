import numpy as np


class FusionFeatureExtractor:
    """
    Multi-sensor feature fusion.


    Input:

        IMU      21 dim
        GPS      12 dim
        LiDAR    10 dim
        Camera    8 dim


    Output:

        Fusion feature

        51 dim


    Functions:

        1. feature concatenation
        2. normalization
        3. missing sensor handling

    """


    def __init__(
        self,
        normalize=True,
        eps=1e-8,
    ):

        self.normalize = normalize

        self.eps = eps


        self.mean = None

        self.std = None



    # ==================================
    # concatenate
    # ==================================

    def concatenate(
        self,
        imu_feature,
        gps_feature,
        lidar_feature,
        camera_feature,
    ):

        features=[]


        sensors=[
            imu_feature,
            gps_feature,
            lidar_feature,
            camera_feature,
        ]


        for feature in sensors:


            if feature is None:

                continue


            feature=np.asarray(
                feature,
                dtype=np.float64
            )


            features.append(
                feature
            )


        if len(features)==0:

            raise ValueError(
                "No sensor feature"
            )


        return np.concatenate(
            features
        )



    # ==================================
    # normalize
    # ==================================

    def fit(
        self,
        features,
    ):
        """
        Calculate normalization parameters.


        Input:

            NxD

        """


        features=np.asarray(
            features,
            dtype=np.float64
        )


        self.mean=np.mean(
            features,
            axis=0
        )


        self.std=np.std(
            features,
            axis=0
        )


        self.std[
            self.std < self.eps
        ] = 1.0



    def transform(
        self,
        features,
    ):


        if self.mean is None:

            raise RuntimeError(
                "Call fit first"
            )


        return (
            features-self.mean
        ) / self.std



    # ==================================
    # main interface
    # ==================================

    def extract(
        self,
        imu_feature,
        gps_feature,
        lidar_feature,
        camera_feature,
    ):


        feature=self.concatenate(
            imu_feature,
            gps_feature,
            lidar_feature,
            camera_feature,
        )


        if self.normalize:

            return feature


        return feature



    # ==================================
    # sequence feature
    # ==================================

    def extract_sequence(
        self,
        imu_features,
        gps_features,
        lidar_features,
        camera_features,
    ):

        """

        Input:

            T x dim


        Output:

            T x 51


        """


        sequence=[]


        length=len(
            imu_features
        )


        for i in range(length):


            feature=self.extract(
                imu_features[i],
                gps_features[i],
                lidar_features[i],
                camera_features[i],
            )


            sequence.append(
                feature
            )


        return np.asarray(
            sequence,
            dtype=np.float64
        )
