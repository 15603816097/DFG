import numpy as np


class LidarFeatureExtractor:
    """
    LiDAR point cloud quality feature extractor.


    Input:

        Nx4

        x y z intensity


    Output:

        10 dimensional feature

    """


    def __init__(self):

        pass



    def extract(
        self,
        points,
    ):


        #################################
        # 输入转换
        #################################

        if isinstance(points, dict):

            if "points" in points:

                points = points["points"]

            else:

                raise ValueError(
                    "LiDAR dictionary has no points key"
                )


        points = np.asarray(
            points,
            dtype=np.float64
        )


        #################################
        # 空点云处理
        #################################

        if points.size == 0:

            return np.zeros(
                10,
                dtype=np.float64
            )


        #################################
        # 维度检查
        #################################

        if points.ndim != 2:

            raise ValueError(
                "Point cloud must be Nx3 or Nx4"
            )


        if points.shape[1] not in [3,4]:

            raise ValueError(
                "Point cloud dimension error"
            )


        #################################
        # 去除异常值
        #################################

        points = points[
            np.isfinite(points).all(axis=1)
        ]


        if len(points)==0:

            return np.zeros(
                10,
                dtype=np.float64
            )


        #################################
        # xyz
        #################################

        xyz = points[:,:3]


        distance = np.linalg.norm(
            xyz,
            axis=1
        )


        #################################
        # 特征计算
        #################################

        feature=np.array(
            [
                len(points),

                np.mean(distance),

                np.std(distance),

                np.mean(xyz[:,0]),

                np.mean(xyz[:,1]),

                np.mean(xyz[:,2]),

                np.std(xyz[:,0]),

                np.std(xyz[:,1]),

                np.std(xyz[:,2]),

                np.mean(
                    distance < 20
                )
            ],
            dtype=np.float64
        )


        #################################
        # 最终保险
        #################################

        feature=np.nan_to_num(
            feature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


        return feature
