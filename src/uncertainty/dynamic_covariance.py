import numpy as np


class DynamicCovarianceEstimator:
    """
    Residual adaptive GPS covariance estimator

    根据GPS残差动态调整因子权重

    小残差:
        高可信
        sigma小

    大残差:
        GPS退化
        sigma大

    """


    def __init__(
        self,
        sigma_min=2.0,
        sigma_max=60.0,
        k=10.0
    ):

        self.sigma_min = sigma_min

        self.sigma_max = sigma_max

        self.k = k



    def compute(
        self,
        gps_position,
        predicted_position
    ):


        gps_position=np.asarray(
            gps_position,
            dtype=float
        )


        predicted_position=np.asarray(
            predicted_position,
            dtype=float
        )


        residual=np.linalg.norm(
            gps_position -
            predicted_position
        )


        # residual normalization

        weight = (
            residual /
            (residual+self.k)
        )


        sigma = (
            self.sigma_min
            +
            (
                self.sigma_max
                -
                self.sigma_min
            )
            *
            weight
        )


        sigma=np.clip(
            sigma,
            self.sigma_min,
            self.sigma_max
        )


        return float(sigma)



    def covariance_matrix(
        self,
        sigma
    ):


        return (
            np.eye(3)
            *
            sigma**2
        )