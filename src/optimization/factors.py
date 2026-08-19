import numpy as np



class BaseFactor:
    """
    Base factor class
    """


    def __init__(
        self,
        measurement,
        covariance=1.0,
    ):

        self.measurement=np.asarray(
            measurement,
            dtype=np.float64
        )


        self.covariance=covariance



    def weight(self):

        return 1.0 / (
            self.covariance
            +
            1e-6
        )



    def residual(
        self,
        state,
    ):

        raise NotImplementedError




class IMUFactor(BaseFactor):
    """
    IMU motion constraint
    """


    def residual(
        self,
        state,
    ):

        return (
            state[:3]
            -
            self.measurement[:3]
        )




class GPSFactor(BaseFactor):
    """
    GPS position constraint
    """


    def residual(
        self,
        state,
    ):

        return (
            state[:3]
            -
            self.measurement[:3]
        )





class LiDARFactor(BaseFactor):
    """
    LiDAR pose constraint
    """


    def residual(
        self,
        state,
    ):

        return (
            state[:3]
            -
            self.measurement[:3]
        )





class CameraFactor(BaseFactor):
    """
    Camera pose constraint
    """


    def residual(
        self,
        state,
    ):

        return (
            state[:3]
            -
            self.measurement[:3]
        )
