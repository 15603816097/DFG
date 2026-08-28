from __future__ import annotations

import numpy as np


def estimate_rigid_transform_svd(
    source,
    target,
):
    """
    Estimate T such that:

        target ~= R @ source + t

    Parameters
    ----------
    source : (N, 3)
    target : (N, 3)

    Returns
    -------
    T : (4, 4)
        Homogeneous rigid transform.
    """
    source = np.asarray(
        source,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape != target.shape
        or source.shape[1] != 3
    ):
        raise ValueError(
            "source and target must have shape (N, 3)"
        )

    if len(source) < 3:
        raise ValueError(
            "At least 3 correspondences are required"
        )

    source_centroid = source.mean(
        axis=0
    )

    target_centroid = target.mean(
        axis=0
    )

    source_centered = (
        source
        -
        source_centroid
    )

    target_centered = (
        target
        -
        target_centroid
    )

    H = (
        source_centered.T
        @
        target_centered
    )

    U, _, Vt = np.linalg.svd(
        H
    )

    R = (
        Vt.T
        @
        U.T
    )

    if np.linalg.det(
        R
    ) < 0.0:
        Vt[-1, :] *= -1.0

        R = (
            Vt.T
            @
            U.T
        )

    t = (
        target_centroid
        -
        R
        @
        source_centroid
    )

    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[
        :3,
        :3
    ] = R

    T[
        :3,
        3
    ] = t

    return T


def transform_points(
    points,
    T,
):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    T = np.asarray(
        T,
        dtype=np.float64,
    )

    return (
        points
        @
        T[
            :3,
            :3
        ].T
        +
        T[
            :3,
            3
        ]
    )
