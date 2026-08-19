import numpy as np


class ESKF:
    """
    Simplified Error State Kalman Filter

    State:

        position(3)
        velocity(3)

    x =
    [px,py,pz,vx,vy,vz]

    Measurement:

        GPS ENU position
    """


    def __init__(self):

        self.x = np.zeros(
            6,
            dtype=np.float64
        )


        self.P = np.eye(
            6,
            dtype=np.float64
        )


        # process noise

        self.Q = (
            np.eye(6)
            *
            0.01
        )


        # GPS measurement noise

        self.R = (
            np.eye(3)
            *
            5.0
        )



    def predict(
        self,
        acceleration,
        dt
    ):

        """
        IMU prediction
        """

        a = np.asarray(
            acceleration,
            dtype=np.float64
        )


        p = self.x[:3]

        v = self.x[3:6]


        # motion model

        self.x[:3] = (
            p
            +
            v * dt
            +
            0.5*a*dt*dt
        )


        self.x[3:6] = (
            v
            +
            a*dt
        )


        # Jacobian

        F = np.eye(6)


        F[:3,3:6] = (
            np.eye(3)
            *
            dt
        )


        self.P = (
            F
            @
            self.P
            @
            F.T
            +
            self.Q
        )



    def update(
        self,
        gps_position
    ):

        """
        GPS correction
        """


        H = np.zeros(
            (3,6)
        )


        H[:,:3] = np.eye(3)


        z = np.asarray(
            gps_position,
            dtype=np.float64
        )


        y = (
            z
            -
            H @ self.x
        )


        S = (
            H
            @
            self.P
            @
            H.T
            +
            self.R
        )


        K = (
            self.P
            @
            H.T
            @
            np.linalg.inv(S)
        )


        self.x += (
            K @ y
        )


        I = np.eye(6)


        self.P = (
            (I-K@H)
            @
            self.P
        )



    def get_position(self):

        return self.x[:3].copy()
