import numpy as np


class GPSFeatureExtractor:
    """
    GPS feature extractor


    Input:

        Nx3

        latitude longitude altitude


    Output:

        12 dimensional feature

    """


    def __init__(self):

        pass


    def extract(
        self,
        gps_data,
    ):


        #################################
        # 输入转换
        #################################

        gps_data = np.asarray(
            gps_data,
            dtype=np.float64
        )


        #################################
        # 单帧GPS兼容
        #################################

        if gps_data.ndim == 1:

            if gps_data.shape[0] != 3:

                raise ValueError(
                    "GPS data must be Nx3"
                )


            gps_data = gps_data.reshape(
                1,
                3
            )


        if gps_data.ndim != 2:

            raise ValueError(
                "GPS data must be Nx3"
            )


        if gps_data.shape[1] != 3:

            raise ValueError(
                "GPS dimension error"
            )


        #################################
        # 删除异常值
        #################################

        gps_data = gps_data[
            np.isfinite(gps_data).all(axis=1)
        ]


        #################################
        # 空数据保护
        #################################

        if len(gps_data) == 0:

            return np.zeros(
                12,
                dtype=np.float64
            )


        #################################
        # 基础坐标
        #################################

        latitude = gps_data[:, 0]

        longitude = gps_data[:, 1]

        altitude = gps_data[:, 2]



        #################################
        # 位置变化
        #################################

        if len(gps_data) > 1:


            position_diff = np.diff(
                gps_data,
                axis=0
            )


            position_change = np.mean(
                position_diff,
                axis=0
            )


            position_std = np.std(
                position_diff,
                axis=0
            )


        else:


            # 单帧没有变化量

            position_change = np.zeros(
                3
            )


            position_std = np.zeros(
                3
            )



        #################################
        # 12维GPS特征
        #################################

        feature = np.array(
            [

                # 1-3
                np.mean(latitude),

                np.mean(longitude),

                np.mean(altitude),



                # 4-6
                np.std(latitude),

                np.std(longitude),

                np.std(altitude),



                # 7
                np.linalg.norm(
                    position_change
                ),



                # 8
                np.linalg.norm(
                    position_std
                ),



                # 9
                len(gps_data),



                # 10
                np.mean(
                    np.linalg.norm(
                        gps_data - gps_data.mean(axis=0),
                        axis=1
                    )
                ),



                # 11
                np.min(
                    altitude
                ),



                # 12
                np.max(
                    altitude
                )

            ],
            dtype=np.float64
        )



        #################################
        # 最终NaN保护
        #################################

        feature = np.nan_to_num(
            feature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        return feature
