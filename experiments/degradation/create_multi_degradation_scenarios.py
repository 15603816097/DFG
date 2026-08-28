"""
Multi-Degradation Scenario Generator
====================================

All algorithm parameters are frozen.

This script only creates multiple GPS degradation scenarios for robustness
testing. Ground truth is used here ONLY because these are controlled
synthetic degradation experiments.

Scenarios
---------
1. mild_progressive
2. medium_progressive
3. severe_progressive
4. sudden
5. bias_drift
6. intermittent_outlier

Outputs
-------
results/multi_degradation/<scenario>/
    gps_corrupted.txt
    gps_error.txt
    severity.txt
    metadata.txt
"""

from __future__ import annotations

import os
import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

GT_PATH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "results",
    "multi_degradation",
)

SEED = 20260826


def smoothstep(x):
    x = np.clip(
        np.asarray(x, dtype=np.float64),
        0.0,
        1.0,
    )
    return x * x * (3.0 - 2.0 * x)


def base_progressive_severity(n):
    """
    Three gradual degradation episodes with recovery.
    """
    severity = np.zeros(
        n,
        dtype=np.float64,
    )

    frame = np.arange(
        n,
        dtype=np.int64,
    )

    episodes = [
        (700, 1000, 1450, 1700, 0.65),
        (1900, 2250, 2900, 3250, 1.00),
        (3500, 3750, 4050, 4300, 0.80),
    ]

    for (
        start,
        rise_end,
        hold_end,
        recovery_end,
        peak,
    ) in episodes:

        rise = (
            (frame >= start)
            &
            (frame < rise_end)
        )

        if np.any(rise):
            x = (
                frame[rise]
                -
                start
            ) / max(
                rise_end
                -
                start,
                1,
            )

            severity[rise] = np.maximum(
                severity[rise],
                peak
                *
                smoothstep(x),
            )

        hold = (
            (frame >= rise_end)
            &
            (frame < hold_end)
        )

        severity[hold] = np.maximum(
            severity[hold],
            peak,
        )

        recovery = (
            (frame >= hold_end)
            &
            (frame < recovery_end)
        )

        if np.any(recovery):
            x = (
                frame[recovery]
                -
                hold_end
            ) / max(
                recovery_end
                -
                hold_end,
                1,
            )

            severity[recovery] = np.maximum(
                severity[recovery],
                peak
                *
                (
                    1.0
                    -
                    smoothstep(x)
                ),
            )

    return np.clip(
        severity,
        0.0,
        1.0,
    )


def generate_progressive(
    gt,
    strength,
    rng,
):
    n = len(gt)

    severity = np.clip(
        base_progressive_severity(n)
        *
        strength,
        0.0,
        1.0,
    )

    noise_sigma = (
        0.35
        +
        5.0
        *
        severity
    )

    noise = rng.normal(
        size=(n, 3)
    ) * noise_sigma[:, None]

    noise[:, 2] *= 0.60

    bias = np.zeros(
        (n, 3),
        dtype=np.float64,
    )

    for i in range(
        1,
        n,
    ):
        walk_sigma = (
            0.015
            +
            0.10
            *
            severity[i]
        )

        step = rng.normal(
            0.0,
            walk_sigma,
            3,
        )

        step[2] *= 0.40

        bias[i] = (
            0.997
            *
            bias[i - 1]
            +
            severity[i]
            *
            step
        )

    t = np.arange(
        n,
        dtype=np.float64,
    )

    drift = np.column_stack(
        [
            7.0
            *
            severity
            *
            np.sin(
                t / 110.0
            ),

            5.0
            *
            severity
            *
            np.cos(
                t / 145.0
            ),

            1.2
            *
            severity
            *
            np.sin(
                t / 180.0
            ),
        ]
    )

    gps = (
        gt
        +
        noise
        +
        bias
        +
        drift
    )

    return gps, severity


def generate_sudden(
    gt,
    rng,
):
    n = len(gt)

    gps = (
        gt
        +
        rng.normal(
            0.0,
            0.35,
            size=gt.shape,
        )
    )

    gps[:, 2] = (
        gt[:, 2]
        +
        rng.normal(
            0.0,
            0.20,
            size=n,
        )
    )

    severity = np.zeros(
        n,
        dtype=np.float64,
    )

    segments = [
        (1000, 1250, np.array([12.0, -8.0, 1.5])),
        (2400, 2650, np.array([-18.0, 10.0, -1.0])),
        (3700, 3900, np.array([14.0, 14.0, 0.8])),
    ]

    for (
        start,
        end,
        offset,
    ) in segments:
        if start >= n:
            continue

        end = min(
            end,
            n,
        )

        gps[
            start:end
        ] += offset

        extra_noise = rng.normal(
            0.0,
            2.0,
            size=(
                end - start,
                3,
            ),
        )

        extra_noise[:, 2] *= 0.4

        gps[
            start:end
        ] += extra_noise

        severity[
            start:end
        ] = 1.0

    return gps, severity


def generate_bias_drift(
    gt,
    rng,
):
    n = len(gt)

    gps = (
        gt
        +
        rng.normal(
            0.0,
            0.35,
            size=gt.shape,
        )
    )

    gps[:, 2] = (
        gt[:, 2]
        +
        rng.normal(
            0.0,
            0.20,
            size=n,
        )
    )

    severity = np.zeros(
        n,
        dtype=np.float64,
    )

    drift = np.zeros(
        (n, 3),
        dtype=np.float64,
    )

    episodes = [
        (900, 1550, 1900, np.array([14.0, -10.0, 1.5])),
        (2350, 3050, 3400, np.array([-20.0, 13.0, -1.2])),
    ]

    frame = np.arange(
        n,
        dtype=np.int64,
    )

    for (
        start,
        peak_end,
        recovery_end,
        max_bias,
    ) in episodes:
        if start >= n:
            continue

        rise = (
            (frame >= start)
            &
            (frame < peak_end)
        )

        if np.any(rise):
            x = (
                frame[rise]
                -
                start
            ) / max(
                peak_end
                -
                start,
                1,
            )

            scale = smoothstep(x)

            drift[rise] += (
                scale[:, None]
                *
                max_bias[None, :]
            )

            severity[rise] = np.maximum(
                severity[rise],
                scale,
            )

        recovery = (
            (frame >= peak_end)
            &
            (frame < recovery_end)
        )

        if np.any(recovery):
            x = (
                frame[recovery]
                -
                peak_end
            ) / max(
                recovery_end
                -
                peak_end,
                1,
            )

            scale = (
                1.0
                -
                smoothstep(x)
            )

            drift[recovery] += (
                scale[:, None]
                *
                max_bias[None, :]
            )

            severity[recovery] = np.maximum(
                severity[recovery],
                scale,
            )

    gps += drift

    return gps, severity


def generate_intermittent_outlier(
    gt,
    rng,
):
    n = len(gt)

    gps = (
        gt
        +
        rng.normal(
            0.0,
            0.35,
            size=gt.shape,
        )
    )

    gps[:, 2] = (
        gt[:, 2]
        +
        rng.normal(
            0.0,
            0.20,
            size=n,
        )
    )

    severity = np.zeros(
        n,
        dtype=np.float64,
    )

    bursts = [
        (800, 20),
        (1200, 12),
        (1780, 35),
        (2300, 18),
        (2820, 30),
        (3300, 15),
        (3950, 28),
    ]

    for start, length in bursts:
        if start >= n:
            continue

        end = min(
            start + length,
            n,
        )

        count = (
            end - start
        )

        directions = rng.normal(
            size=(
                count,
                3,
            )
        )

        norms = np.linalg.norm(
            directions,
            axis=1,
            keepdims=True,
        )

        norms[
            norms < 1e-12
        ] = 1.0

        directions /= norms

        amplitudes = rng.uniform(
            12.0,
            28.0,
            size=(
                count,
                1,
            ),
        )

        directions[:, 2] *= 0.25

        gps[
            start:end
        ] += (
            directions
            *
            amplitudes
        )

        severity[
            start:end
        ] = 1.0

    return gps, severity


def save_scenario(
    name,
    gps,
    severity,
    gt,
    description,
):
    out = os.path.join(
        OUTPUT_ROOT,
        name,
    )

    os.makedirs(
        out,
        exist_ok=True,
    )

    error = np.linalg.norm(
        gps
        -
        gt,
        axis=1,
    )

    np.savetxt(
        os.path.join(
            out,
            "gps_corrupted.txt",
        ),
        gps,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            out,
            "gps_error.txt",
        ),
        error,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            out,
            "severity.txt",
        ),
        severity,
        fmt="%.8f",
    )

    with open(
        os.path.join(
            out,
            "metadata.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"name={name}\n"
        )

        file.write(
            f"description={description}\n"
        )

        file.write(
            f"frames={len(gps)}\n"
        )

        file.write(
            f"error_mean={error.mean():.8f}\n"
        )

        file.write(
            f"error_rmse={np.sqrt(np.mean(error**2)):.8f}\n"
        )

        file.write(
            f"error_max={error.max():.8f}\n"
        )

    print(
        f"{name:24s} "
        f"mean={error.mean():8.3f} m "
        f"rmse={np.sqrt(np.mean(error**2)):8.3f} m "
        f"max={error.max():8.3f} m"
    )


def main():
    if not os.path.exists(
        GT_PATH
    ):
        raise FileNotFoundError(
            GT_PATH
        )

    gt = np.loadtxt(
        GT_PATH,
        dtype=np.float64,
    )

    if (
        gt.ndim != 2
        or
        gt.shape[1] != 3
    ):
        raise ValueError(
            "Ground truth must be N x 3"
        )

    print("=" * 88)
    print(
        "MULTI-DEGRADATION GPS "
        "SCENARIO GENERATION"
    )
    print("=" * 88)

    scenario_specs = [
        (
            "mild_progressive",
            "progressive",
            0.50,
            "Gradual low-intensity noise/bias degradation",
        ),

        (
            "medium_progressive",
            "progressive",
            0.75,
            "Gradual medium-intensity noise/bias degradation",
        ),

        (
            "severe_progressive",
            "progressive",
            1.00,
            "Gradual severe noise/bias degradation",
        ),

        (
            "sudden",
            "sudden",
            None,
            "Abrupt GPS position jumps with short persistent faults",
        ),

        (
            "bias_drift",
            "bias_drift",
            None,
            "Slowly growing and recovering GPS bias drift",
        ),

        (
            "intermittent_outlier",
            "intermittent_outlier",
            None,
            "Short intermittent large GPS outlier bursts",
        ),
    ]

    for index, (
        name,
        kind,
        strength,
        description,
    ) in enumerate(
        scenario_specs
    ):
        rng = np.random.default_rng(
            SEED
            +
            1000
            *
            index
        )

        if kind == "progressive":
            gps, severity = (
                generate_progressive(
                    gt,
                    strength,
                    rng,
                )
            )

        elif kind == "sudden":
            gps, severity = (
                generate_sudden(
                    gt,
                    rng,
                )
            )

        elif kind == "bias_drift":
            gps, severity = (
                generate_bias_drift(
                    gt,
                    rng,
                )
            )

        elif (
            kind
            ==
            "intermittent_outlier"
        ):
            gps, severity = (
                generate_intermittent_outlier(
                    gt,
                    rng,
                )
            )

        else:
            raise RuntimeError(
                kind
            )

        save_scenario(
            name,
            gps,
            severity,
            gt,
            description,
        )

    print()
    print(
        "Saved root:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
