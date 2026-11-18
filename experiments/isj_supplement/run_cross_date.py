from __future__ import annotations

"""Isolated, frozen KITTI cross-date Stage-7 evaluation.

The historical 2011_10_03 Stage-7 code and results are NEVER modified.
This script calls the same Stage-7 functions with temporary in-memory path
bindings, always writing to results/isj_cross_date/<drive>/.

Stages are deliberately separate because raw LiDAR/stereo preprocessing is
expensive and Mamba may need a different Python environment:

  inspect     validate cross-date raw files (counts/timestamps only)
  preprocess  run the frozen 0027 degradation recipe + sensor factor extraction
  gru         run frozen V5 GRU on saved feature arrays (CPU)
  mamba       run frozen V6 Mamba on the saved features (mamba_ssm required)
  fgo         Fixed, full Dynamic-U and Reliability Only with common fixed start
  converge    continue each saved graph under the previously used uniform rule

No checkpoint training; NO target-sequence hyperparameter search or GT
initialization. The OXTS/GPS reference is NOT independent of GPS input.
This script evaluates the existing synthetic-degradation protocol, not actual
naturally occurring sensor faults. Date-aware paths are verified before work.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.isj_supplement.audit_cross_date_raw import inspect_drive

RAW_ROOT = ROOT / "dataset" / "kitti"
RESULT_ROOT = ROOT / "results" / "isj_cross_date"
DEV_DATE = "2011_10_03"
SEQUENCE_PATTERN = re.compile(r"^(\d{4}_\d{2}_\d{2})_drive_\d+_sync$")

MODELS = {
    "gru": {
        "source": "v5",
        "full": "FINAL_S1_v5_gru",
        "rel": "ABL_V5_GRU_RELIABILITY_ONLY",
    },
    "mamba": {
        "source": "v6",
        "full": "FINAL_S2_v6_all_mamba",
        "rel": "ABL_V6_MAMBA_RELIABILITY_ONLY",
    },
}
SENSORS = ("gps", "imu", "camera")


def date_for(sequence: str) -> str:
    match = SEQUENCE_PATTERN.fullmatch(sequence)
    if match is None:
        raise ValueError(
            f"Invalid KITTI Raw synchronized drive name: {sequence!r}"
        )
    if match.group(1) == DEV_DATE:
        raise ValueError(
            f"{sequence}: historical {DEV_DATE} development/test sequence "
            "is not allowed by the independent cross-date entrypoint"
        )
    return match.group(1)


def validate_destination(sequence: str, result_root: Path) -> Path:
    """Prevents outputs from escaping the dedicated cross-date result root."""
    date_for(sequence)
    result_root = Path(result_root).resolve()
    original = (ROOT / "results" / "cross_sequence").resolve()
    if result_root == original or original in result_root.parents:
        raise ValueError(f"Refusing to write historical Stage-7 output: {result_root}")
    if result_root == (ROOT / "results").resolve():
        raise ValueError("Output root must be a dedicated new result subdirectory")
    result_root_parent = (ROOT / "results").resolve()
    if result_root_parent not in result_root.parents:
        raise ValueError(
            f"Output root must live under repository results/: {result_root}"
        )
    out = (result_root / sequence).resolve()
    if out.parent != result_root:
        raise ValueError("Invalid sequence path")
    return out


def raw_preflight(sequence: str, min_frames: int) -> dict:
    date = date_for(sequence)
    date_dir = RAW_ROOT / date
    drive = date_dir / sequence
    if not drive.is_dir():
        raise FileNotFoundError(drive)
    report = inspect_drive(date_dir, drive, min_frames=min_frames)
    if report["raw_layout_status"] != "READY_RAW":
        raise ValueError(
            f"Raw sequence not ready: {sequence}: "
            + "; ".join(report["issues"])
        )
    return report


@contextmanager
def crossdate_bindings(sequence: str, output_root: Path, stage: str):
    """Patch only module-local bindings during one isolated process stage."""
    date = date_for(sequence)
    out_dir = validate_destination(sequence, output_root)
    data_root = RAW_ROOT / date

    def sequence_path(name: str) -> Path:
        if name != sequence:
            raise ValueError(f"Cross-date loader cannot access other drive: {name}")
        return data_root / name

    def sequence_out(name: str) -> Path:
        if name != sequence:
            raise ValueError(f"Cross-date writer cannot access other drive: {name}")
        return out_dir

    with ExitStack() as stack:
        import experiments.cross_sequence.stage7_config as cfg
        stack.enter_context(patch.object(cfg, "TEST_SEQUENCES", (sequence,)))
        stack.enter_context(patch.object(cfg, "DATE_DIR", data_root))
        stack.enter_context(patch.object(cfg, "OUT_ROOT", output_root))
        stack.enter_context(patch.object(cfg, "sequence_path", sequence_path))
        stack.enter_context(patch.object(cfg, "sequence_out", sequence_out))

        if stage == "preprocess":
            from experiments.cross_sequence import build_sequence_data as bld
            for key, value in (
                ("TEST_SEQUENCES", (sequence,)),
                ("DATE_DIR", data_root),
                ("OUT_ROOT", output_root),
                ("sequence_path", sequence_path),
                ("sequence_out", sequence_out),
            ):
                stack.enter_context(patch.object(bld, key, value))
        elif stage == "gru":
            from experiments.cross_sequence import infer_v5_gru as v5
            stack.enter_context(patch.object(v5, "TEST_SEQUENCES", (sequence,)))
            stack.enter_context(patch.object(v5, "sequence_out", sequence_out))
        elif stage == "mamba":
            from experiments.cross_sequence import infer_v6_mamba as v6
            stack.enter_context(patch.object(v6, "TEST_SEQUENCES", (sequence,)))
            stack.enter_context(patch.object(v6, "sequence_out", sequence_out))
        elif stage in ("fgo", "converge"):
            import experiments.cross_sequence.run_stage7_factor_graph as s7
            stack.enter_context(patch.object(s7, "sequence_out", sequence_out))
            if stage == "converge":
                import experiments.isj_supplement.run_uniform_convergence as conv
                stack.enter_context(patch.object(conv, "sequence_out", sequence_out))
        else:
            raise ValueError(f"Unsupported binding stage: {stage}")
        yield out_dir


def prediction_preflight(out: Path, selected_models: tuple[str, ...]) -> None:
    for model in selected_models:
        source = MODELS[model]["source"]
        for sensor in SENSORS:
            for kind in ("reliability", "uncertainty"):
                path = out / "predictions" / source / f"{sensor}_{kind}.txt"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Frozen {model} prediction missing: {path}. "
                        f"Run --stage {model} first."
                    )


def result_is_complete(out: Path, name: str, n: int) -> bool:
    folder = out / "factor_graph" / name
    pose = folder / "poses.npy"
    metric_file = folder / "metrics.json"
    if not pose.exists() and not metric_file.exists():
        return False
    if not (pose.is_file() and metric_file.is_file()):
        raise ValueError(f"Incomplete cached FGO result {folder}; inspect manually")
    saved = np.load(pose, allow_pickle=False)
    if saved.shape != (n, 4, 4) or not np.isfinite(saved).all():
        raise ValueError(f"Cached FGO poses are not valid: {pose}")
    info = json.loads(metric_file.read_text(encoding="utf-8"))
    if not math.isfinite(float(info["ATE3D"])):
        raise ValueError(f"Invalid cached FGO metric: {metric_file}")
    return True


def write_protocol(out: Path, sequence: str, state: dict) -> None:
    manifest = out / "isj_cross_date_protocol.json"
    data = {
        "sequence": sequence,
        "date": date_for(sequence),
        "status": "frozen_cross_date_synthetic_degradation_evaluation",
        "historical_stage7_output_untouched": True,
        "checkpoint_training": False,
        "checkpoint_normalization": "frozen",
        "degradation": "original 0027 plan resampled to new drive",
        "calibrations": f"dataset/kitti/{date_for(sequence)}/calib_*.txt",
        "GPS_ground_truth_caveat": (
            "Clean OXTS/GPS is the metric reference, degraded GPS comes "
            "from the same OXTS/GPS records; reference is not independent."
        ),
        "solver": "original GTSAM LM 100 iterations + separate uniform continuation",
        "warning": "This is a short cross-date pipeline pilot, not broad held-out statistical validation.",
    }
    if manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("sequence") != sequence or old.get("date") != date_for(sequence):
            raise ValueError("Existing cross-date manifest belongs to another drive")
    data.update(state)
    manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def require_processed(out: Path) -> int:
    required = (
        "features.npz", "sensor_data.npz", "lidar_factor_data.npz",
        "camera_factor_data.npz", "ground_truth.txt",
    )
    for fname in required:
        if not (out / fname).is_file():
            raise FileNotFoundError(
                f"{out / fname} missing; first run --stage preprocess"
            )
    with np.load(out / "sensor_data.npz", allow_pickle=False) as arrays:
        n = len(arrays["degraded_gps_local"])
    with np.load(out / "features.npz", allow_pickle=False) as arrays:
        for sensor in SENSORS:
            a = arrays[f"{sensor}_features"]
            if len(a) != n or not np.isfinite(a).all():
                raise ValueError(f"Invalid {sensor} features in {out}")
    return n


def stage_preprocess(sequence: str, out: Path, force: bool) -> None:
    processed = out / "sensor_data.npz"
    if processed.is_file() and not force:
        require_processed(out)
        print(f"PREPROCESS CACHED: {out}")
        return
    if out.exists() and any(
        (out / p).exists() for p in (
            "predictions", "factor_graph", "isj_cross_date_fgo_results.csv"
        )
    ):
        raise RuntimeError(
            f"Cannot rebuild {out}: predictions/FGO already exist. "
            "Do not silently change frozen inputs after inference."
        )
    from experiments.cross_sequence import build_sequence_data as bld
    bld.build_one(sequence)
    n = require_processed(out)
    write_protocol(out, sequence, {"preprocessing": "complete", "frames": n})
    print(f"PREPROCESS DONE: {sequence}, {n} frames, out={out}")


def stage_infer(sequence: str, out: Path, model: str, force: bool) -> None:
    n = require_processed(out)
    source = MODELS[model]["source"]
    paths = [
        out / "predictions" / source / f"{sensor}_{kind}.txt"
        for sensor in SENSORS for kind in ("reliability", "uncertainty")
    ]
    if all(p.is_file() for p in paths) and not force:
        for p in paths:
            a = np.loadtxt(p, dtype=float).reshape(-1)
            if len(a) != n or not np.isfinite(a).all():
                raise ValueError(f"Corrupt cached prediction: {p}")
        print(f"FROZEN {model.upper()} PREDICTIONS CACHED: {out}")
        return
    if any(p.is_file() for p in paths) and not force:
        raise ValueError(
            f"Partial {model} predictions: inspect {out / 'predictions' / source}; "
            "use --force only after checking they are safe to recompute."
        )
    if (out / "factor_graph").exists():
        raise RuntimeError("Refuse to change predictions after FGO results exist")
    if model == "gru":
        from experiments.cross_sequence import infer_v5_gru as v5
        for sensor in SENSORS:
            v5.infer(sequence, sensor)
    elif model == "mamba":
        try:
            from mamba_ssm import Mamba  # noqa:F401
        except ImportError as exc:
            raise RuntimeError(
                "Official Mamba inference requires mamba_ssm in this Python "
                "environment. Use a separate supported environment for "
                "--stage mamba; do not substitute a new checkpoint/model."
            ) from exc
        import torch
        from experiments.cross_sequence import infer_v6_mamba as v6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for sensor in SENSORS:
            v6.infer(sequence, sensor, device=device, batch_size=256)
    else:
        raise ValueError(model)
    prediction_preflight(out, (model,))
    write_protocol(out, sequence, {f"{model}_inference": "complete"})
    print(f"FROZEN {model.upper()} INFERENCE DONE: {out}")


def stage_fgo(
    sequence: str, out: Path, selected_models: tuple[str, ...],
    force: bool,
) -> None:
    n = require_processed(out)
    prediction_preflight(out, selected_models)
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    fixed = "FINAL_S0_fixed"
    if not result_is_complete(out, fixed, n) or force:
        # For safety never overwrite old results even when "--force" is set.
        if (out / "factor_graph" / fixed).exists():
            raise RuntimeError(
                f"Fixed stage exists already: {out / 'factor_graph' / fixed}; "
                "refusing to overwrite warm start of existing method runs."
            )
        s7.run_case(seq=sequence, name=fixed)
    fixed_path = out / "factor_graph" / fixed / "poses.npy"
    poses = np.load(fixed_path, allow_pickle=False)

    original_initial = s7.initial_trajectory_from_pose
    original_confidence = s7.confidence

    def fixed_start(gtsam, measurements, initial):
        if len(measurements.gps_local) != len(poses):
            raise ValueError("Fixed warm-start does not match current sequence")
        return [s7._pose3_from_matrix(gtsam, pose) for pose in poses]

    cases = []
    for model in selected_models:
        source = MODELS[model]["source"]
        cases.extend([
            (MODELS[model]["full"], source, False),
            (MODELS[model]["rel"], source, True),
        ])
    rows = []
    try:
        s7.initial_trajectory_from_pose = fixed_start
        for name, source, reliability_only in cases:
            if result_is_complete(out, name, n):
                if force:
                    raise RuntimeError(
                        f"FGO case exists already: {name}. Never overwrite "
                        "raw optimizer outputs; use convergence continuation."
                    )
                status = "cached"
            else:
                s7.confidence = (lambda u: 1.0) if reliability_only else original_confidence
                s7.run_case(
                    seq=sequence, name=name,
                    gps_src=source, imu_src=source, cam_src=source
                )
                status = "computed"
            metric = json.loads(
                (out / "factor_graph" / name / "metrics.json").read_text(encoding="utf-8")
            )
            rows.append({
                "sequence": sequence,
                "case": name,
                "model": "GRU" if source == "v5" else "Mamba",
                "configuration": "reliability_only" if reliability_only else "full_dynamic_U",
                "ATE3D_m": float(metric["ATE3D"]),
                "status": status,
            })
    finally:
        s7.confidence = original_confidence
        s7.initial_trajectory_from_pose = original_initial
    fixed_ate = json.loads(
        (out / "factor_graph" / fixed / "metrics.json").read_text(encoding="utf-8")
    )["ATE3D"]
    print(f"\nFixed stage: ATE3D={fixed_ate:.6f} m")
    for row in rows:
        print(f"  {row['case']:<38} ATE3D={row['ATE3D_m']:.6f} m ({row['status']})")
    path = out / f"isj_cross_date_fgo_{'_'.join(selected_models)}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_protocol(
        out, sequence, {
            "fgo": "complete",
            "FGO_models_requested": list(selected_models),
            "common_warm_start": fixed,
            "stage1_fixed_ATE3D": float(fixed_ate),
        }
    )
    print("Saved:", path)


def stage_converge(
    sequence: str, out: Path, selected_models: tuple[str, ...],
    max_extra_passes: int,
) -> None:
    import gtsam
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    from experiments.isj_supplement.run_uniform_convergence import refine_case
    n = require_processed(out)
    m = s7.load_measurements(sequence)
    cases = [{"name": "FINAL_S0_fixed", "type": "fixed"}]
    for model in selected_models:
        spec = MODELS[model]
        cases.append({
            "name": spec["full"], "type": "dynamic", "source": spec["source"]
        })
        cases.append({
            "name": spec["rel"], "type": "reliability", "source": spec["source"]
        })
    for case in cases:
        if not result_is_complete(out, case["name"], n):
            raise FileNotFoundError(
                f"{case['name']} not yet solved; run --stage fgo first"
            )
    rows = []
    for case in cases:
        print(f"\n{sequence} | {case['name']}", flush=True)
        rows.append(refine_case(
            gtsam, sequence, m, case, max_extra_passes, 1e-7, 1e-3
        ))
    filename = out / f"isj_cross_date_convergence_{'_'.join(selected_models)}.csv"
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['case']:<38} "
            f"{row['source_ATE3D']:.6f} -> {row['continued_ATE3D']:.6f}m "
            f"{'STABLE' if row['stationary_under_rule'] else 'NOT_STABLE'}"
        )
    write_protocol(
        out, sequence, {
            "convergence_models": list(selected_models),
            "convergence_rule": {
                "max_extra_passes": max_extra_passes,
                "outer_relative_objective_tol": 1e-7,
                "outer_translation_shift_RMSE_m": 0.001,
            }
        }
    )
    print("Saved:", filename)
    if any(not row["stationary_under_rule"] for row in rows):
        print("WARNING: one or more methods did NOT meet the strict stopping rule.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    p.add_argument(
        "--stage", required=True,
        choices=("inspect", "preprocess", "gru", "mamba", "fgo", "converge")
    )
    p.add_argument("--model", choices=("gru", "mamba", "both"), default="gru",
                   help="For fgo/converge; use gru first if Mamba unavailable")
    p.add_argument("--max-extra-passes", type=int, default=4)
    p.add_argument("--force", action="store_true",
                   help="Only permits rerunning PREPROCESS before inference, or "
                        "inference before FGO. NEVER overwrites FGO outputs.")
    args = p.parse_args()
    if args.max_extra_passes < 1:
        p.error("--max-extra-passes must be >=1")
    date_for(args.sequence)
    output_root = RESULT_ROOT
    out = validate_destination(args.sequence, output_root)
    if args.stage == "inspect":
        print(json.dumps(
            raw_preflight(args.sequence, min_frames=100),
            indent=2, ensure_ascii=False
        ))
        print("Isolated output:", out)
        return
    selected_models = ("gru", "mamba") if args.model == "both" else (args.model,)

    if args.stage == "preprocess":
        raw_preflight(args.sequence, min_frames=100)

    with crossdate_bindings(args.sequence, output_root, args.stage):
        if args.stage == "preprocess":
            stage_preprocess(args.sequence, out, args.force)
        elif args.stage == "gru":
            stage_infer(args.sequence, out, "gru", args.force)
        elif args.stage == "mamba":
            stage_infer(args.sequence, out, "mamba", args.force)
        elif args.stage == "fgo":
            stage_fgo(args.sequence, out, selected_models, args.force)
        elif args.stage == "converge":
            stage_converge(
                args.sequence, out, selected_models, args.max_extra_passes
            )


if __name__ == "__main__":
    main()
