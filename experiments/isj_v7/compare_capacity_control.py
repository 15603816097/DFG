from __future__ import annotations

"""V7 GPS-only, development-only capacity control: GRU(106) vs Mamba(96).

Read-only. Load saved per-seed predictions and metrics from frozen
development paths. Verify provenance, time alignment, configuration and
capacity difference BEFORE reporting any model result.

This is not an independent-drive experiment and not a final FGO claim.
"""

import argparse
import json

import numpy as np

from experiments.isj_v7.compare_dev_gps import read_candidate, read_run
from experiments.isj_v7.train_gps_dev import DEFAULT_OUTPUT, CAPACITY_OUTPUT

HEADS = ("h0", "h1", "h2")
EXPECTED = {
    "gru_96": {"backend": "gru", "hidden": 96, "parameters": 114630},
    "gru_106": {"backend": "gru", "hidden": 106, "parameters": 139290},
    "mamba_96": {"backend": "mamba", "hidden": 96, "parameters": 139590},
}


def check_triplet(triplet: dict, seed: int) -> None:
    if set(triplet) != set(EXPECTED):
        raise ValueError("missing GRU96, GRU106 or Mamba96")
    configurations = []
    for label, expected in EXPECTED.items():
        record = triplet[label]
        metrics = record["metrics"]
        config = metrics["config"]
        if (metrics["backend"] != expected["backend"]
                or config["hidden"] != expected["hidden"]
                or metrics["parameter_count"] != expected["parameters"]
                or config["seed"] != seed
                or config["layers"] != 2):
            raise ValueError(f"{label}: invalid architecture or seed")
        if label == "gru_106":
            if metrics.get("run_family") != "capacity_gru_h106":
                raise ValueError("GRU106 is not from capacity-control run")
        elif metrics.get("run_family", "baseline") != "baseline":
            raise ValueError(f"{label}: not a baseline run")
        configs = dict(config)
        configs.pop("hidden")
        configurations.append(configs)
    if not all(config == configurations[0] for config in configurations[1:]):
        raise ValueError("capacity comparison training settings differ")
    baseline = triplet["gru_96"]["arrays"]
    for label in ("gru_106", "mamba_96"):
        contender = triplet[label]["arrays"]
        if not np.array_equal(baseline["anchors"], contender["anchors"]):
            raise ValueError(f"{label}: incomparable validation anchors")
        if not np.array_equal(baseline["synthetic_targets"],
                              contender["synthetic_targets"]):
            raise ValueError(f"{label}: incomparable validation targets")
    hashes = [
        record["metrics"]["dataset_sha256"]
        for record in triplet.values()
    ]
    if len(set(hashes)) != 1:
        raise ValueError("capacity comparison datasets differ")


def summarize_one_seed(triplet: dict, seed: int) -> dict:
    check_triplet(triplet, seed)
    result = {
        "seed": int(seed),
        "development_only": True,
        "independent_test_run": False,
        "heads": {},
    }
    for hidx, h in enumerate(HEADS):
        target = triplet["gru_96"]["arrays"]["synthetic_targets"][:, hidx]
        point = {}
        for key, record in triplet.items():
            prediction = record["arrays"]["reliability"][:, hidx]
            delta = prediction - target
            point[key] = {
                "MAE": float(np.mean(np.abs(delta))),
                "RMSE": float(np.sqrt(np.mean(delta ** 2))),
            }
        for metric in ("MAE", "RMSE"):
            base = point["gru_106"][metric]
            point[f"mamba_improvement_over_capacity_gru_{metric}_pct"] = (
                100.0 * (base - point["mamba_96"][metric]) / base
                if base > 0 else None
            )
        result["heads"][h] = point
    return result


def aggregate_seeds(records: list[dict]) -> dict:
    if not records:
        raise ValueError("no seed records")
    if len({row["seed"] for row in records}) != len(records):
        raise ValueError("duplicate random seeds")
    result = {
        "schema": "isj_v7_dev_capacity_control_v1",
        "development_only": True,
        "independent_drives": False,
        "seeds": [row["seed"] for row in records],
        "model_parameters": {
            name: cfg["parameters"] for name, cfg in EXPECTED.items()
        },
        "horizons": {},
    }
    for h in HEADS:
        result["horizons"][h] = {}
        for model in EXPECTED:
            result["horizons"][h][model] = {}
            for metric in ("MAE", "RMSE"):
                values = np.array(
                    [row["heads"][h][model][metric] for row in records],
                    dtype=float,
                )
                result["horizons"][h][model][metric] = {
                    "mean": float(np.mean(values)),
                    "sample_std": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1 else None
                    ),
                }
        for metric in ("MAE", "RMSE"):
            baseline = result["horizons"][h]["gru_106"][metric]["mean"]
            mamba = result["horizons"][h]["mamba_96"][metric]["mean"]
            result["horizons"][h][
                f"mamba_gain_over_capacity_gru_{metric}_pct"
            ] = 100.0 * (baseline - mamba) / baseline
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    args = parser.parse_args(argv)
    if (len(args.seeds) == 0 or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 for seed in args.seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    candidate = read_candidate()
    per_seed = []
    for seed in args.seeds:
        records = {
            "gru_96": read_run(DEFAULT_OUTPUT, "gru", seed, candidate),
            "gru_106": read_run(CAPACITY_OUTPUT, "gru", seed, candidate),
            "mamba_96": read_run(DEFAULT_OUTPUT, "mamba", seed, candidate),
        }
        per_seed.append(summarize_one_seed(records, seed))
    summary = aggregate_seeds(per_seed)
    for h, result in summary["horizons"].items():
        print(
            f"{h}: GRU96 MAE="
            f"{result['gru_96']['MAE']['mean']:.6f}, "
            f"GRU106 MAE={result['gru_106']['MAE']['mean']:.6f}, "
            f"Mamba96 MAE={result['mamba_96']['MAE']['mean']:.6f}, "
            f"Mamba vs GRU106 MAE improvement="
            f"{result['mamba_gain_over_capacity_gru_MAE_pct']:+.2f}%"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("CAPACITY_COMPARISON_PASS; DEV_ONLY; NO_HELDOUT_ACCESSED; NO_WRITES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
