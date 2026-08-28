from __future__ import annotations

import csv
import json
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
    sys.path.insert(
        0,
        ROOT,
    )


from src.reliability.lidar_gt_factor_oracle import (
    build_oracle,
)


OUT = (
    Path(ROOT)
    / "results"
    / "lidar_physical_oracle_benchmark"
)

FACTOR_PATH = (
    OUT
    / "physical_lidar_factor_data.npz"
)

ORACLE_PATH = (
    OUT
    / "lidar_gt_oracle_reliability.npz"
)

SUMMARY_PATH = (
    OUT
    / "degradation_factor_error_summary.csv"
)


def main():

    print("=" * 120)
    print(
        "RECOVER LIDAR PHYSICAL BENCHMARK "
        "WITHOUT RE-RUNNING ICP"
    )
    print("=" * 120)

    # ============================================================
    # 1. Check existing ICP result
    # ============================================================

    if not FACTOR_PATH.exists():
        raise FileNotFoundError(
            "\nphysical_lidar_factor_data.npz 不存在。\n"
            "说明之前 ICP 结果没有成功保存，才需要重新跑。\n"
            f"Expected:\n{FACTOR_PATH}"
        )

    print()
    print(
        "Loading existing factor data:"
    )
    print(
        FACTOR_PATH
    )

    data = np.load(
        FACTOR_PATH,
        allow_pickle=False,
    )

    print()
    print(
        "Available arrays:"
    )

    for key in data.files:
        print(
            f"  {key:20s}",
            data[key].shape,
        )

    # ============================================================
    # 2. Load arrays
    # ============================================================

    te = np.asarray(
        data["te"],
        dtype=np.float64,
    )

    re = np.asarray(
        data["re"],
        dtype=np.float64,
    )

    type_id = np.asarray(
        data["type"],
        dtype=np.int32,
    )

    level_id = np.asarray(
        data["level"],
        dtype=np.int32,
    )

    fitness = np.asarray(
        data["fitness"],
        dtype=np.float64,
    )

    rmse = np.asarray(
        data["rmse"],
        dtype=np.float64,
    )

    converged = np.asarray(
        data["converged"],
        dtype=bool,
    )

    pair_count = len(
        te
    )

    frame_count = (
        pair_count
        +
        1
    )

    print()
    print(
        "Pair count:",
        pair_count,
    )

    print(
        "Frame count:",
        frame_count,
    )

    # ============================================================
    # 3. Build GT oracle
    # ============================================================

    clean_mask = (
        level_id
        ==
        0
    )

    print()
    print(
        "Clean pairs:",
        int(
            np.sum(
                clean_mask
            )
        ),
    )

    oracle = build_oracle(
        te,
        re,
        clean_mask,
    )

    pair_reliability = np.asarray(
        oracle[
            "lidar_reliability"
        ],
        dtype=np.float64,
    )

    # Factor i represents frame i -> i+1.
    # Graph reliability uses target-frame indexing.
    frame_reliability = np.ones(
        frame_count,
        dtype=np.float64,
    )

    frame_reliability[
        1:
    ] = pair_reliability

    print()
    print(
        "Pair reliability:"
    )

    print(
        "  min :",
        float(
            np.min(
                pair_reliability
            )
        ),
    )

    print(
        "  max :",
        float(
            np.max(
                pair_reliability
            )
        ),
    )

    print(
        "  mean:",
        float(
            np.mean(
                pair_reliability
            )
        ),
    )

    # ============================================================
    # 4. Save oracle
    #
    # IMPORTANT:
    # Do NOT use **oracle together with
    # lidar_reliability=frame_reliability because oracle already
    # contains lidar_reliability.
    # ============================================================

    oracle_to_save = dict(
        oracle
    )

    # Rename pair-level reliability explicitly.
    oracle_to_save[
        "lidar_reliability_pair"
    ] = oracle_to_save.pop(
        "lidar_reliability"
    )

    # Graph-facing frame-level reliability.
    oracle_to_save[
        "lidar_reliability"
    ] = frame_reliability

    oracle_to_save[
        "translation_error"
    ] = te

    oracle_to_save[
        "rotation_error_deg"
    ] = re

    np.savez_compressed(
        ORACLE_PATH,
        **oracle_to_save,
    )

    print()
    print(
        "Oracle saved:"
    )
    print(
        ORACLE_PATH
    )

    # ============================================================
    # 5. Build degradation summary
    # ============================================================

    type_names = {
        0: "clean",
        1: "sparse",
        2: "range_noise",
        3: "occlusion",
        4: "ghost_outlier",
    }

    level_names = {
        0: "clean",
        1: "mild",
        2: "moderate",
        3: "severe",
    }

    rows = []

    for tid in range(
        5
    ):

        for lid in range(
            4
        ):

            # clean only corresponds to (0, 0)
            if tid == 0:
                if lid != 0:
                    continue

            else:
                if lid == 0:
                    continue

            mask = (
                (type_id == tid)
                &
                (level_id == lid)
            )

            if not np.any(
                mask
            ):
                continue

            row = {
                "type":
                    type_names[
                        tid
                    ],

                "level":
                    level_names[
                        lid
                    ],

                "count":
                    int(
                        np.sum(
                            mask
                        )
                    ),

                "translation_error_mean":
                    float(
                        np.mean(
                            te[
                                mask
                            ]
                        )
                    ),

                "translation_error_p90":
                    float(
                        np.percentile(
                            te[
                                mask
                            ],
                            90,
                        )
                    ),

                "rotation_error_deg_mean":
                    float(
                        np.mean(
                            re[
                                mask
                            ]
                        )
                    ),

                "rotation_error_deg_p90":
                    float(
                        np.percentile(
                            re[
                                mask
                            ],
                            90,
                        )
                    ),

                "fitness_mean":
                    float(
                        np.mean(
                            fitness[
                                mask
                            ]
                        )
                    ),

                "rmse_mean":
                    float(
                        np.mean(
                            rmse[
                                mask
                            ]
                        )
                    ),

                "converged_ratio":
                    float(
                        np.mean(
                            converged[
                                mask
                            ]
                        )
                    ),

                "oracle_reliability_mean":
                    float(
                        np.mean(
                            pair_reliability[
                                mask
                            ]
                        )
                    ),
            }

            rows.append(
                row
            )

    # ============================================================
    # 6. Save CSV
    # ============================================================

    with SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        "Summary saved:"
    )
    print(
        SUMMARY_PATH
    )

    # ============================================================
    # 7. Print summary immediately
    # ============================================================

    print()
    print("=" * 145)

    print(
        "PHYSICAL DEGRADATION -> TRUE LIDAR FACTOR ERROR"
    )

    print("=" * 145)

    print(
        f"{'Type':16s}"
        f"{'Level':12s}"
        f"{'N':>8s}"
        f"{'tMean':>12s}"
        f"{'tP90':>12s}"
        f"{'rMean':>12s}"
        f"{'rP90':>12s}"
        f"{'Fitness':>12s}"
        f"{'ICP RMSE':>12s}"
        f"{'Conv':>10s}"
        f"{'Oracle R':>12s}"
    )

    print(
        "-" * 145
    )

    for row in rows:

        print(
            f"{row['type']:16s}"
            f"{row['level']:12s}"
            f"{row['count']:8d}"
            f"{row['translation_error_mean']:12.4f}"
            f"{row['translation_error_p90']:12.4f}"
            f"{row['rotation_error_deg_mean']:12.4f}"
            f"{row['rotation_error_deg_p90']:12.4f}"
            f"{row['fitness_mean']:12.4f}"
            f"{row['rmse_mean']:12.4f}"
            f"{row['converged_ratio']:10.4f}"
            f"{row['oracle_reliability_mean']:12.4f}"
        )

    # ============================================================
    # 8. Metadata
    # ============================================================

    metadata = {
        "frames":
            frame_count,

        "pairs":
            pair_count,

        "clean_pairs":
            int(
                np.sum(
                    clean_mask
                )
            ),

        "oracle_reliability_min":
            float(
                np.min(
                    pair_reliability
                )
            ),

        "oracle_reliability_max":
            float(
                np.max(
                    pair_reliability
                )
            ),

        "oracle_reliability_mean":
            float(
                np.mean(
                    pair_reliability
                )
            ),

        "translation_error_mean":
            float(
                np.mean(
                    te
                )
            ),

        "rotation_error_deg_mean":
            float(
                np.mean(
                    re
                )
            ),
    }

    metadata_path = (
        OUT
        /
        "recovered_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)

    print(
        "RECOVERY FINISHED"
    )

    print("=" * 120)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "You DO NOT need to run the 4543-pair ICP again."
    )

    print()
    print(
        "Next:"
    )

    print(
        "python "
        "experiments/lidar_benchmark/"
        "inspect_lidar_physical_benchmark.py"
    )

    print(
        "python "
        "experiments/factor_graph/"
        "run_lidar_physical_oracle_ablation.py"
    )

    print(
        "python "
        "experiments/evaluation/"
        "evaluate_lidar_physical_oracle.py"
    )


if __name__ == "__main__":
    main()
