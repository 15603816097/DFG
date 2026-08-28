from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from src.factor_graph.gps_z_soft_dynamic_covariance import (
    ALPHA_VALUES,
    alpha_name,
)


GROUND_TRUTH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)

RESULT_ROOT = os.path.join(
    ROOT,
    "results",
    "gps_z_soft_dynamic_sweep",
)

OUTPUT_CSV = os.path.join(
    ROOT,
    "results",
    "gps_z_soft_dynamic_sweep_summary.csv",
)

OUTPUT_TXT = os.path.join(
    ROOT,
    "results",
    "gps_z_soft_dynamic_sweep_summary.txt",
)


def metrics(reference, trajectory):
    n = min(
        len(reference),
        len(trajectory),
    )

    reference = np.asarray(
        reference[:n, :3],
        dtype=np.float64,
    )

    trajectory = np.asarray(
        trajectory[:n, :3],
        dtype=np.float64,
    )

    error = trajectory - reference
    e3 = np.linalg.norm(error, axis=1)
    e2 = np.linalg.norm(error[:, :2], axis=1)

    return {
        "ATE3D": float(
            np.sqrt(
                np.mean(e3 ** 2)
            )
        ),
        "ATE2D": float(
            np.sqrt(
                np.mean(e2 ** 2)
            )
        ),
        "Mean3D": float(
            np.mean(e3)
        ),
        "Max3D": float(
            np.max(e3)
        ),
        "ZRMSE": float(
            np.sqrt(
                np.mean(
                    error[:, 2] ** 2
                )
            )
        ),
    }


def main():
    if not Path(GROUND_TRUTH).exists():
        raise FileNotFoundError(
            GROUND_TRUTH
        )

    reference = np.loadtxt(
        GROUND_TRUTH,
        dtype=np.float64,
    )

    rows = []

    print("=" * 116)
    print(
        "GPS Z SOFT-DYNAMIC COVARIANCE EVALUATION"
    )
    print("=" * 116)

    print(
        f"{'AlphaZ':>8s} "
        f"{'ATE3D':>12s} "
        f"{'ATE2D':>12s} "
        f"{'ZRMSE':>12s} "
        f"{'Mean3D':>12s} "
        f"{'Max3D':>12s}"
    )

    print("-" * 116)

    for alpha in ALPHA_VALUES:
        path = os.path.join(
            RESULT_ROOT,
            alpha_name(alpha),
            "trajectory.txt",
        )

        if not Path(path).exists():
            print(
                f"{alpha:8.2f} NOT RUN"
            )
            continue

        trajectory = np.loadtxt(
            path,
            dtype=np.float64,
        )

        result = metrics(
            reference,
            trajectory,
        )

        row = {
            "AlphaZ": float(alpha),
            **result,
        }

        rows.append(row)

        print(
            f"{alpha:8.2f} "
            f"{result['ATE3D']:12.6f} "
            f"{result['ATE2D']:12.6f} "
            f"{result['ZRMSE']:12.6f} "
            f"{result['Mean3D']:12.6f} "
            f"{result['Max3D']:12.6f}"
        )

    if not rows:
        raise RuntimeError(
            "No alpha sweep result found."
        )

    best_ate = min(
        rows,
        key=lambda row: row["ATE3D"],
    )

    ate_limit = (
        best_ate["ATE3D"] * 1.03
    )

    near_best = [
        row
        for row in rows
        if row["ATE3D"] <= ate_limit
    ]

    best_balanced = min(
        near_best,
        key=lambda row: row["ZRMSE"],
    )

    print()
    print("=" * 116)
    print("BEST ATE3D")
    print(
        f"alpha_z={best_ate['AlphaZ']:.2f} "
        f"ATE3D={best_ate['ATE3D']:.6f} "
        f"ATE2D={best_ate['ATE2D']:.6f} "
        f"ZRMSE={best_ate['ZRMSE']:.6f}"
    )

    print()
    print(
        "BEST BALANCED "
        "(ATE3D within 3% of best, minimum ZRMSE)"
    )

    print(
        f"alpha_z={best_balanced['AlphaZ']:.2f} "
        f"ATE3D={best_balanced['ATE3D']:.6f} "
        f"ATE2D={best_balanced['ATE2D']:.6f} "
        f"ZRMSE={best_balanced['ZRMSE']:.6f}"
    )
    print("=" * 116)

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "AlphaZ",
                "ATE3D",
                "ATE2D",
                "ZRMSE",
                "Mean3D",
                "Max3D",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "GPS Z SOFT-DYNAMIC COVARIANCE SWEEP",
        "",
    ]

    for row in rows:
        lines.append(
            f"alpha_z={row['AlphaZ']:.2f} "
            f"ATE3D={row['ATE3D']:.6f} "
            f"ATE2D={row['ATE2D']:.6f} "
            f"ZRMSE={row['ZRMSE']:.6f} "
            f"Mean3D={row['Mean3D']:.6f} "
            f"Max3D={row['Max3D']:.6f}"
        )

    lines.extend(
        [
            "",
            (
                "BEST ATE3D: "
                f"alpha_z={best_ate['AlphaZ']:.2f}, "
                f"ATE3D={best_ate['ATE3D']:.6f}, "
                f"ZRMSE={best_ate['ZRMSE']:.6f}"
            ),
            (
                "BEST BALANCED: "
                f"alpha_z={best_balanced['AlphaZ']:.2f}, "
                f"ATE3D={best_balanced['ATE3D']:.6f}, "
                f"ZRMSE={best_balanced['ZRMSE']:.6f}"
            ),
        ]
    )

    Path(OUTPUT_TXT).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("Saved:", OUTPUT_CSV)
    print("Saved:", OUTPUT_TXT)


if __name__ == "__main__":
    main()
