import numpy as np



class DynamicCovariance:
    """
    Dynamic covariance estimator


    Input:

        GPS residual


    Output:

        measurement sigma


    Formula:

        sigma =
            sigma_min +
            (1-reliability)
            *
            (sigma_max-sigma_min)

    """


    def __init__(
        self,
        sigma_min=5.0,
        sigma_max=50.0
    ):


        self.sigma_min=sigma_min

        self.sigma_max=sigma_max





    def compute_reliability(
        self,
        residual
    ):


        """
        Residual based reliability


        small residual:
            reliability -> 1


        large residual:
            reliability -> 0

        """


        error=np.linalg.norm(
            residual
        )


        reliability=np.exp(
            -error/20.0
        )


        reliability=np.clip(
            reliability,
            0.0,
            1.0
        )


        return reliability






    def compute_sigma(
        self,
        residual
    ):


        reliability=self.compute_reliability(
            residual
        )


        sigma=(

            self.sigma_min

            +

            (1-reliability)

            *
            (
                self.sigma_max
                -
                self.sigma_min
            )

        )


        return sigma






    def covariance_matrix(
        self,
        residual
    ):


        sigma=self.compute_sigma(
            residual
        )


        return (

            np.eye(3)

            *
            sigma**2

        )
