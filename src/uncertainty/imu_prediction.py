import numpy as np


class IMUPredictor:
    """
    Simple IMU position prediction

    Used for dynamic covariance estimation

    """

    def __init__(
        self,
        dt=0.1
    ):

        self.dt=dt


        self.position=np.zeros(3)

        self.velocity=np.zeros(3)



    def reset(
        self,
        position
    ):

        self.position=np.asarray(
            position,
            dtype=float
        )

        self.velocity=np.zeros(3)



    def update(
        self,
        acceleration
    ):

        acceleration=np.asarray(
            acceleration,
            dtype=float
        )


        dt=self.dt


        # velocity update

        self.velocity += acceleration*dt


        # position update

        self.position += (
            self.velocity*dt
            +
            0.5*acceleration*dt*dt
        )


        return self.position.copy()
