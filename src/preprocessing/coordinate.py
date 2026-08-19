import numpy as np


class GPSCoordinateConverter:
    """
    KITTI GPS coordinate converter.

    Convert:

        Latitude
        Longitude
        Altitude

    to:

        Local ENU coordinate

    """

    def __init__(
        self,
        latitude0=None,
        longitude0=None,
        altitude0=None,
    ):
        """
        Reference origin.

        If None,
        the first GPS point will be used.
        """

        self.latitude0 = latitude0
        self.longitude0 = longitude0
        self.altitude0 = altitude0

        self.initialized = (
            latitude0 is not None
            and longitude0 is not None
            and altitude0 is not None
        )


        # Earth radius
        self.R = 6378137.0


    # ==================================================
    # Initialize origin
    # ==================================================

    def set_origin(
        self,
        latitude,
        longitude,
        altitude,
    ):
        """
        Set ENU reference point.
        """

        self.latitude0 = latitude
        self.longitude0 = longitude
        self.altitude0 = altitude

        self.initialized = True



    # ==================================================
    # GPS -> ENU
    # ==================================================

    def gps_to_enu(
        self,
        latitude,
        longitude,
        altitude,
    ):
        """
        Convert GPS WGS84 to local ENU.

        Output:

            [east,north,up]

        """

        if not self.initialized:

            self.set_origin(
                latitude,
                longitude,
                altitude,
            )


        lat = np.deg2rad(
            latitude
        )

        lon = np.deg2rad(
            longitude
        )


        lat0 = np.deg2rad(
            self.latitude0
        )

        lon0 = np.deg2rad(
            self.longitude0
        )


        d_lat = lat - lat0

        d_lon = lon - lon0


        east = (
            self.R
            *
            np.cos(lat0)
            *
            d_lon
        )


        north = (
            self.R
            *
            d_lat
        )


        up = (
            altitude
            -
            self.altitude0
        )


        return np.array(
            [
                east,
                north,
                up
            ],
            dtype=np.float64
        )



    # ==================================================
    # Batch conversion
    # ==================================================

    def gps_array_to_enu(
        self,
        gps_array,
    ):
        """
        Convert multiple GPS points.

        Input:

            Nx3

        [
          lat,
          lon,
          alt
        ]

        Output:

            Nx3

        [
          east,
          north,
          up
        ]

        """

        gps_array = np.asarray(
            gps_array,
            dtype=np.float64,
        )


        if gps_array.ndim != 2:

            raise ValueError(
                "GPS array must be Nx3"
            )


        if gps_array.shape[1] != 3:

            raise ValueError(
                "GPS array must contain "
                "latitude longitude altitude"
            )


        result=[]


        for gps in gps_array:

            enu = self.gps_to_enu(
                gps[0],
                gps[1],
                gps[2],
            )

            result.append(
                enu
            )


        return np.asarray(
            result,
            dtype=np.float64
        )



# ======================================================
# Coordinate transformation utilities
# ======================================================


class CoordinateTransform:
    """
    General coordinate transformation.

    Used for:

        LiDAR
        Camera
        IMU

    """

    @staticmethod
    def homogeneous(
        points
    ):
        """
        Convert XYZ to homogeneous.

        Nx3

        ->
        
        Nx4
        """

        points=np.asarray(
            points,
            dtype=np.float64
        )


        ones=np.ones(
            (
                points.shape[0],
                1
            ),
            dtype=np.float64
        )


        return np.concatenate(
            [
                points,
                ones
            ],
            axis=1
        )


    @staticmethod
    def transform(
        points,
        T,
    ):
        """
        Apply 4x4 transformation.

        """

        points=np.asarray(
            points,
            dtype=np.float64
        )


        T=np.asarray(
            T,
            dtype=np.float64
        )


        if T.shape != (4,4):

            raise ValueError(
                "Transformation must be 4x4"
            )


        homo = (
            CoordinateTransform
            .homogeneous(points)
        )


        result = (
            T
            @
            homo.T
        ).T


        return result[:,:3]



    @staticmethod
    def distance(
        p1,
        p2,
    ):
        """
        Euclidean distance.
        """

        p1=np.asarray(
            p1
        )

        p2=np.asarray(
            p2
        )


        return np.linalg.norm(
            p1-p2
        )
