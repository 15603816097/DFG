import numpy as np


class IMUFeatureExtractor:
    """
    IMU degradation feature extractor.

    Input:

        Nx6

        [
            ax,
            ay,
            az,
            gx,
            gy,
            gz
        ]


    Output:

        21 dimensional feature

    Feature:

        acceleration statistics
        gyro statistics
        motion energy
        acceleration magnitude
        gyro magnitude

    """


    def __init__(
        self,
        window_size=10,
        dt=0.01,
    ):

        self.window_size = window_size

        self.dt = dt



    @staticmethod
    def mean_value(data):

        return np.mean(
            data,
            axis=0
        )



    @staticmethod
    def std_value(data):

        return np.std(
            data,
            axis=0
        )



    @staticmethod
    def change_rate(data):

        if len(data) < 2:

            return np.zeros(
                data.shape[1]
            )


        diff = np.diff(
            data,
            axis=0
        )


        return np.mean(
            np.abs(diff),
            axis=0
        )



    @staticmethod
    def motion_energy(data):

        """
        Overall motion energy.
        """

        return np.array(
            [
                np.mean(
                    data ** 2
                )
            ]
        )



    @staticmethod
    def acceleration_magnitude(acc):

        """
        Mean acceleration magnitude.
        """

        magnitude = np.linalg.norm(
            acc,
            axis=1
        )


        return np.array(
            [
                np.mean(magnitude)
            ]
        )



    @staticmethod
    def gyro_magnitude(gyro):

        """
        Mean angular velocity magnitude.
        """

        magnitude = np.linalg.norm(
            gyro,
            axis=1
        )


        return np.array(
            [
                np.mean(magnitude)
            ]
        )



    def extract(
        self,
        imu_data,
    ):

        imu_data = np.asarray(
            imu_data,
            dtype=np.float64,
        )


        if imu_data.ndim != 2:

            raise ValueError(
                "IMU data must be Nx6"
            )


        if imu_data.shape[1] != 6:

            raise ValueError(
                "IMU data must contain 6 channels"
            )


        acc = imu_data[:,0:3]


        gyro = imu_data[:,3:6]



        acc_mean = self.mean_value(
            acc
        )


        acc_std = self.std_value(
            acc
        )


        acc_change = self.change_rate(
            acc
        )



        gyro_mean = self.mean_value(
            gyro
        )


        gyro_std = self.std_value(
            gyro
        )


        gyro_change = self.change_rate(
            gyro
        )



        energy = self.motion_energy(
            imu_data
        )


        acc_mag = self.acceleration_magnitude(
            acc
        )


        gyro_mag = self.gyro_magnitude(
            gyro
        )


        feature = np.concatenate(
            [
                acc_mean,
                acc_std,
                acc_change,

                gyro_mean,
                gyro_std,
                gyro_change,

                energy,

                acc_mag,

                gyro_mag,
            ]
        )


        return feature



    def extract_sequence(
        self,
        imu_sequence,
    ):

        imu_sequence = np.asarray(
            imu_sequence,
            dtype=np.float64,
        )


        features=[]


        for i in range(
            self.window_size,
            len(imu_sequence)+1
        ):

            window = imu_sequence[
                i-self.window_size:i
            ]


            feature = self.extract(
                window
            )


            features.append(
                feature
            )


        return np.asarray(
            features,
            dtype=np.float64
        )
