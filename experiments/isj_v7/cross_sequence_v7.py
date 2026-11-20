from __future__ import annotations

"""Frozen V7 across-drive synthetic GPS corruption benchmark.

Default invocation is metadata-only preflight and NEVER reads reserved
0016/0020 sensor contents. --release-all is the ONE-TIME evaluation release:
build both drives, solve every method without reference, then evaluate.
This is NOT independent physical ground truth or causal iSAM2.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.cross_sequence.build_sequence_data import normalized_plan
from experiments.degradation.build_degraded_four_sensor_measurements import (
    corrupt_local_gps, lidar_quality, camera_quality, extract_lidar,
    ts_seconds,
)
from experiments.isj_v7.run_gps_covariance_pilot import (
    CROSS_SEQUENCE_CASES, frozen_v7_sequence_gps_sigmas,
    load_train_axis_noise, rebase_frozen_v7_measurements,
)
from experiments.isj_v7.supervision_gate import read_policy, validate_policy
from experiments.isj_v7.train_gps_dev import (
    load_frozen_v7_gps_checkpoint,
)
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import (
    gps_geodetic_to_local, run_four_sensor_factor_graph, _pose3_to_matrix,
)
from src.loader.gps_loader import GPSLoader
from src.loader.imu_loader import IMULoader
from src.loader.lidar_loader import LidarLoader
from src.common.time_utils import load_kitti_timestamps
from src.multisensor_reliability.degradation.balanced_plans import (
    load_balanced_plan,
)
from src.multisensor_reliability.degradation.apply import (
    corrupt_imu, corrupt_lidar, corrupt_camera,
)
from src.preprocessing.calibration import KITTICalibration
from src.odometry.body_frame_converter import BodyFrameConverter
from src.odometry.lidar_odometry import LidarOdometry
from src.odometry.stereo_visual_odometry import StereoVisualOdometry
from src.odometry.relative_factor_data import make_factor_data

ROOT = Path(__file__).resolve().parents[2]
DATE_DIR = ROOT / "dataset" / "kitti" / "2011_09_30"
DEV_PLAN_DIR = ROOT / "results" / "multisensor_reliability_v2"
DEST = ROOT / "results" / "isj_v7" / "heldout_cap4_v1"
EXPECTED_FRAMES = {"0016": 279, "0020": 1104}
CASES = (
    "axis_fixed",
    "axis_gru96_r_cap4",
    "axis_gru106_r_cap4",
    "axis_mamba_r_cap4",
)
MAX_PASSES = 12
REL_OBJ_TOL = 1e-6
POSITION_SHIFT_TOL_M = .01
CALIBRATION_NAMES = (
    "calib_cam_to_cam.txt", "calib_imu_to_velo.txt",
    "calib_velo_to_cam.txt",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sequence_path(sequence: str) -> Path:
    if sequence not in EXPECTED_FRAMES:
        raise ValueError("unapproved reserved V7 sequence")
    return DATE_DIR / f"2011_09_30_drive_{sequence}_sync"


def metadata_only_preflight() -> dict:
    """Never open OXTS, LiDAR, image, or reserved timestamp contents."""
    status = {}
    for filename in CALIBRATION_NAMES:
        p = DATE_DIR / filename
        status[filename] = {
            "exists": p.is_file(),
            "size_bytes": p.stat().st_size if p.is_file() else 0,
        }
    for sid, n in EXPECTED_FRAMES.items():
        seq = sequence_path(sid)
        timestamp_status = {}
        for folder in ("oxts", "velodyne_points", "image_02", "image_03"):
            timestamp_file = seq / folder / "timestamps.txt"
            timestamp_status[folder] = (
                timestamp_file.is_file()
                and timestamp_file.stat().st_size > 0
            )
        counts = {}
        for relative, pattern in (
            ("oxts/data", "*.txt"), ("velodyne_points/data", "*.bin"),
            ("image_02/data", "*.png"), ("image_03/data", "*.png"),
        ):
            folder = seq / relative
            counts[relative] = (
                sum(1 for _ in folder.glob(pattern)) if folder.is_dir()
                else 0
            )
        status[sid] = {
            "expected_frames": n, "counts": counts,
            "timestamps_ready": timestamp_status,
            "ready": (all(count == n for count in counts.values())
                      and all(timestamp_status.values())),
        }
    status["all_ready"] = (
        all(status[k]["exists"] and status[k]["size_bytes"] > 0
            for k in CALIBRATION_NAMES)
        and all(status[sid]["ready"] for sid in EXPECTED_FRAMES)
        and DEV_PLAN_DIR.joinpath("degradation_plan_v2.npz").is_file()
    )
    status["reserved_content_read"] = False
    status["evaluation_released"] = False
    status["output_exists"] = DEST.exists()
    return status


def _check_numbered_paths(paths, n, label):
    if len(paths) != n:
        raise ValueError(f"{label}: unexpected file count {len(paths)}")
    expected = [f"{i:010d}" for i in range(n)]
    observed = [Path(p).stem for p in paths]
    if observed != expected:
        raise ValueError(f"{label}: nonconsecutive or misaligned frame IDs")



def validate_frozen_time_alignment(
    sequence: str, oxts_seconds: np.ndarray,
    other_seconds: dict[str, np.ndarray],
) -> dict:
    """Post-release timestamp engineering audit, no frame removal/reindex.

    The first three 0020 synchronized frames have a joint early timestamp
    anomaly. It was observed AFTER heldout release, so this exception is
    transparently post-hoc and is NOT a clean preregistered test gate.
    Original frame indices, timestamps and factor data are untouched.
    For 0020, reject any deviation from the precise observed pattern.
    """
    if sequence not in EXPECTED_FRAMES:
        raise ValueError("unapproved reserved V7 sequence")
    ref = np.asarray(oxts_seconds, dtype=np.float64)
    n = EXPECTED_FRAMES[sequence]
    if (ref.shape != (n,) or not np.isfinite(ref).all()
            or np.any(np.diff(ref) <= 0)):
        raise ValueError("OXTS timestamps invalid/nonmonotone")
    required = {"velodyne", "left-camera", "right-camera"}
    if set(other_seconds) != required:
        raise ValueError("required modality timestamps missing")
    details = {}
    early_deltas = []
    all_failing = []
    for modality in sorted(required):
        times = np.asarray(other_seconds[modality], dtype=np.float64)
        if (times.shape != (n,) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError(f"{sequence}/{modality}: invalid timestamps")
        delta = times - ref
        failed = np.flatnonzero(np.abs(delta) > .10)
        if failed.size:
            if (sequence != "0020"
                    or not np.array_equal(
                        failed, np.array([0, 1, 2], dtype=np.int64))
                    or np.any(delta[:3] < -.15)
                    or np.any(delta[:3] >= -.10)
                    or np.max(np.abs(delta[3:])) > .10):
                raise ValueError(
                    f"{sequence}/{modality}: timestamp misalignment"
                )
        details[modality] = {
            "exceed_100ms_indices": [int(x) for x in failed],
            "median_abs_delta_s": float(np.median(np.abs(delta))),
            "max_abs_delta_s": float(np.max(np.abs(delta))),
        }
        early_deltas.append(delta[:3])
        all_failing.append(bool(failed.size))
    applied = any(all_failing)
    if applied and (
            sequence != "0020"
            or not all(all_failing)
            or np.max(np.ptp(np.stack(early_deltas), axis=0)) > .010):
        raise ValueError("0020 startup offsets not shared by all sensors")
    return {
        "policy": (
            "post_release_0020_first_three_joint_early_offsets"
            if applied else "original_all_frames_max_100ms"
        ),
        "startup_exception_applied": bool(applied),
        "frame_ids_unchanged": True,
        "timestamps_unchanged": True,
        "truncated_frames": 0,
        "modalities": details,
    }


def _build_one(sequence: str, plan: dict, out: Path) -> None:
    """Read source measurements only AFTER explicit release gate."""
    seq = sequence_path(sequence)
    n = EXPECTED_FRAMES[sequence]
    gl, il, ll = GPSLoader(seq), IMULoader(seq), LidarLoader(seq)
    left = sorted((seq / "image_02" / "data").glob("*.png"))
    right = sorted((seq / "image_03" / "data").glob("*.png"))
    if (len(gl), len(il), len(ll), len(left), len(right)) != (n,)*5:
        raise ValueError(f"{sequence}: sensor frame count mismatch")
    _check_numbered_paths(gl.files, n, "OXTS")
    _check_numbered_paths(ll.files, n, "LiDAR")
    _check_numbered_paths(left, n, "left stereo")
    _check_numbered_paths(right, n, "right stereo")
    # All checks precede loading any OXTS position or any sensor frame.
    # Only the documented joint 0020 startup pattern is exceptioned.
    time_audit = validate_frozen_time_alignment(
        sequence,
        ts_seconds(gl.timestamps),
        {
            "velodyne": ts_seconds(ll.timestamps),
            "left-camera": ts_seconds(load_kitti_timestamps(
                seq / "image_02" / "timestamps.txt")),
            "right-camera": ts_seconds(load_kitti_timestamps(
                seq / "image_03" / "timestamps.txt")),
        },
    )
    print(
        "TIMESTAMP_POLICY", sequence, time_audit["policy"],
        "start_exception=", time_audit["startup_exception_applied"],
        flush=True,
    )
    pp = normalized_plan(plan, n)
    seed = int(pp["seed"])
    lla = np.asarray([gl[i]["position"] for i in range(n)], np.float64)
    clean = gps_geodetic_to_local(lla)
    degraded = np.empty_like(clean)
    dgyro = np.empty((n, 3), np.float64)
    timestamps = []
    for i in range(n):
        degraded[i] = corrupt_local_gps(
            clean[i], pp["severity"]["gps"][i],
            pp["mode"]["gps"][i], i, seed)
        _, g = corrupt_imu(
            il[i]["acceleration"], il[i]["angular_velocity"],
            pp["severity"]["imu"][i], pp["mode"]["imu"][i],
            i, seed=seed)
        dgyro[i] = g
        timestamps.append(il[i]["timestamp"])
    stamp_sec = ts_seconds(timestamps)
    if (stamp_sec.shape != (n,) or not np.isfinite(stamp_sec).all()
            or np.any(np.diff(stamp_sec) <= 0)):
        raise ValueError(f"{sequence}: nonmonotone/invalid timestamps")

    calib = KITTICalibration(DATE_DIR)
    converter = BodyFrameConverter(calib)
    lidar = LidarOdometry(
        max_iterations=20, max_correspondence_distance=1.5,
        min_correspondences=80, max_points=6000,
    )
    cam = StereoVisualOdometry(
        str(DATE_DIR / "calib_cam_to_cam.txt"),
        num_features=2500, min_matches=30, min_inliers=20,
        ratio_test=.75, disparity_num=128,
        disparity_block_size=5, max_depth=80,
    )
    lc, lb, lv, lq = [], [], [], []
    previous = np.eye(4)
    print("BUILD_LIDAR_START", sequence, "pairs=", n-1, flush=True)
    for i in range(n-1):
        p = corrupt_lidar(
            extract_lidar(ll[i]), pp["severity"]["lidar"][i],
            pp["mode"]["lidar"][i], i, seed=seed)
        q = corrupt_lidar(
            extract_lidar(ll[i+1]), pp["severity"]["lidar"][i+1],
            pp["mode"]["lidar"][i+1], i+1, seed=seed)
        result = lidar.estimate(
            p, q, initial_transform=previous if i else np.eye(4),
            source_frame_id=i, target_frame_id=i+1,
        )
        valid, quality = lidar_quality(result)
        f = converter.lidar_to_body(
            result.transform, valid=valid, quality=quality)
        lc.append(f.coordinate_transform)
        lb.append(f.between_measurement)
        lv.append(valid)
        lq.append(quality)
        previous = result.transform if valid else np.eye(4)
        if i % 200 == 0:
            print("BUILD_LIDAR", sequence, i, "/", n-1, flush=True)

    import cv2
    cc, cb, cv, cq = [], [], [], []
    print("BUILD_STEREO_START", sequence, "pairs=", n-1, flush=True)
    for i in range(n-1):
        a = cv2.imread(str(left[i]), cv2.IMREAD_COLOR)
        b = cv2.imread(str(right[i]), cv2.IMREAD_COLOR)
        c = cv2.imread(str(left[i+1]), cv2.IMREAD_COLOR)
        if a is None or b is None or c is None:
            f = converter.camera_to_body(
                np.eye(4), valid=False, quality=0.)
            cc.append(f.coordinate_transform)
            cb.append(f.between_measurement)
            cv.append(False)
            cq.append(0.)
            continue
        a = corrupt_camera(
            a, pp["severity"]["camera"][i],
            pp["mode"]["camera"][i], i, seed=seed)
        b = corrupt_camera(
            b, pp["severity"]["camera"][i],
            pp["mode"]["camera"][i], i+100000, seed=seed)
        c = corrupt_camera(
            c, pp["severity"]["camera"][i+1],
            pp["mode"]["camera"][i+1], i+1, seed=seed)
        result = cam.estimate(a, b, c)
        valid, quality = camera_quality(result)
        f = converter.camera_to_body(
            result.transform, valid=valid, quality=quality)
        cc.append(f.coordinate_transform)
        cb.append(f.between_measurement)
        cv.append(valid)
        cq.append(quality)
        if i % 200 == 0:
            print("BUILD_STEREO", sequence, i, "/", n-1, flush=True)

    # Write only after sensor checks/factor generation. A partial run
    # remains explicitly marked INCOMPLETE; output is never overwritten.
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "degraded_sensor_data.npz",
        degraded_gps_local=degraded,
        degraded_imu_gyro=dgyro,
        timestamps_seconds=stamp_sec,
    )
    np.savez_compressed(
        out / "evaluation_reference.npz",
        original_oxts_gps_local=clean,
    )
    make_factor_data(
        np.asarray(lc), np.asarray(lb),
        np.asarray(lv, bool), np.asarray(lq, float),
    ).save(out / "degraded_lidar_factor_data.npz")
    make_factor_data(
        np.asarray(cc), np.asarray(cb),
        np.asarray(cv, bool), np.asarray(cq, float),
    ).save(out / "degraded_camera_factor_data.npz")
    metadata = {
        "schema": "v7_reserved_synthetic_four_sensor_v1",
        "sequence": sequence, "frames": n,
        "date_calibration": "2011_09_30",
        "timestamp_alignment_audit": time_audit,
        "synthetic_gps_reference": "ORIGINAL_OXTS_GPS_NONINDEPENDENT",
        "plan": "frozen dev 0027 schedule; normalized-time resampling",
        "degradation_seed": seed,
        "corrupt_lidar": "legacy 0027 four-sensor corrupt_lidar",
        "corrupt_camera": "legacy 0027 four-sensor corrupt_camera",
        "cam_lidar_estimator": "legacy 0027 four-sensor settings",
        "gps_feature_same_corruption_stream": True,
        "used_test_reference_for_model_selection": False,
        "status": "BUILT_NOT_EVALUATED",
    }
    (out / "build_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("BUILD_COMPLETE", sequence, "frames=", n, flush=True)


def _positions(gtsam, values, n):
    return np.stack(
        [_pose3_to_matrix(values.atPose3(i)) for i in range(n)],
        axis=0,
    )


def _solve_graph(graph, initial, n):
    import gtsam
    state = initial
    previous = _positions(gtsam, state, n)
    steps = []
    stable = False
    began = time.monotonic()
    for idx in range(MAX_PASSES):
        before = float(graph.error(state))
        cfg = gtsam.LevenbergMarquardtParams()
        cfg.setMaxIterations(100)
        cfg.setRelativeErrorTol(1e-7)
        new = gtsam.LevenbergMarquardtOptimizer(
            graph, state, cfg).optimize()
        after = float(graph.error(new))
        current = _positions(gtsam, new, n)
        if (not np.isfinite([before, after]).all()
                or not np.isfinite(current).all()
                or after > before + 1e-8 * max(abs(before), 1.0)):
            raise RuntimeError("invalid/nondecreasing V7 heldout FGO")
        shift = float(np.sqrt(np.mean(np.sum(
            (current[:, :3, 3] - previous[:, :3, 3])**2, axis=1))))
        ratio = max(0.0, before-after) / max(abs(before), 1.)
        steps.append({
            "pass": idx+1, "objective_after": after,
            "relative_objective_reduction": ratio,
            "position_shift_rmse_m": shift,
            "stabilized": bool(ratio <= REL_OBJ_TOL
                               and shift <= POSITION_SHIFT_TOL_M),
        })
        state, previous = new, current
        if steps[-1]["stabilized"]:
            stable = True
            break
    return {
        "poses": previous,
        "objective": float(graph.error(state)),
        "stabilized": stable, "passes": steps,
        "elapsed_seconds": round(time.monotonic()-began, 3),
    }


def origin_relative_diagnostics(position, reference):
    """Synthetic OXTS-relative position/displacement, NOT physical ATE."""
    p = np.asarray(position, np.float64)
    q = np.asarray(reference, np.float64)
    if (p.ndim != 2 or p.shape[1] != 3 or q.shape != p.shape
            or len(p) <= 10 or not np.isfinite(p).all()
            or not np.isfinite(q).all()):
        raise ValueError("invalid reserved synthetic reference dimensions")
    q0 = q - q[0]
    def rmse(delta):
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    return {
        "origin_aligned_oxTS_relative_3d_rmse_m": rmse(p-q0),
        "origin_aligned_oxTS_relative_2d_rmse_m": rmse((p-q0)[:, :2]),
        "relative_displacement_1_frame_rmse_m": rmse(
            np.diff(p, axis=0)-np.diff(q0, axis=0)),
        "relative_displacement_10_frame_rmse_m": rmse(
            (p[10:]-p[:-10])-(q0[10:]-q0[:-10])),
        "independent_position_ground_truth": False,
    }


def _evaluate_without_reference(sequence, train_axis_fit):
    """Run all frozen cases; do not open saved OXTS reference."""
    src = DEST / sequence
    m = rebase_frozen_v7_measurements(
        load_degraded_four_sensor_measurements(src))
    n = len(m.gps_local)
    if n != EXPECTED_FRAMES[sequence]:
        raise ValueError("built measurement length differs from preflight")
    cases = {}
    common_initial_positions = None
    for case in CASES:
        backend = CROSS_SEQUENCE_CASES[case]
        if backend is None:
            sigma, metadata = frozen_v7_sequence_gps_sigmas(
                m.gps_local, case=case, train_axis_fit=train_axis_fit)
            provenance = None
        else:
            model, center, iqr, provenance = (
                load_frozen_v7_gps_checkpoint(
                    backend, seed=43, device="auto"))
            sigma, metadata = frozen_v7_sequence_gps_sigmas(
                m.gps_local, case=case, train_axis_fit=train_axis_fit,
                model=model, train_median=center, train_iqr=iqr,
                model_provenance=provenance,
            )
        graph, initial, counts = run_four_sensor_factor_graph(
            m, src / "not_written_graph", mode="fixed",
            gps_sigmas_override=sigma, build_graph_only=True,
        )
        import gtsam
        init_pose = _positions(gtsam, initial, n)
        if common_initial_positions is None:
            common_initial_positions = init_pose
        elif not np.array_equal(common_initial_positions, init_pose):
            raise ValueError("V7 cases do not share the identical initialization")
        solved = _solve_graph(graph, initial, n)
        print(
            "SEALED_SOLVE", sequence, case,
            "passes=", len(solved["passes"]),
            "stable=", solved["stabilized"],
            "objective=", solved["objective"],
            flush=True,
        )
        cases[case] = {
            "solve": {k:v for k,v in solved.items() if k != "poses"},
            "poses": solved["poses"],
            "sigma_metadata": metadata,
            "factor_counts": counts,
            "checkpoint_provenance": provenance,
        }
    return cases


def release_all():
    """Single release of both drives. No per-drive tuning/retry switches."""
    status = metadata_only_preflight()
    if not status["all_ready"]:
        raise RuntimeError("reserved V7 input metadata preflight failed")
    if status["output_exists"]:
        raise FileExistsError(
            f"V7 heldout output exists; refuse repeat/overwrite: {DEST}")
    policy = read_policy()
    validate_policy(policy)
    train_fit = load_train_axis_noise(policy)
    plan = load_balanced_plan(DEV_PLAN_DIR)
    # Reserve the output root once, before the sealed full release.
    # A failure leaves a visible incomplete archive (never auto-overwrite).
    DEST.mkdir(parents=True, exist_ok=False)
    (DEST / "RUN_INCOMPLETE").write_text(
        "Heldout released; do not rerun without audit.\n",
        encoding="utf-8")
    for seq in EXPECTED_FRAMES:
        _build_one(seq, plan, DEST / seq)
    return _finish_sealed_evaluation(train_fit)


def _finish_sealed_evaluation(train_fit: dict, *, recovered: bool = False):
    """Evaluate only after all four methods on both drives are solved."""
    # No reference file opened until *both* drives and all frozen
    # methods have finished optimizing.
    solved = {
        seq: _evaluate_without_reference(seq, train_fit)
        for seq in EXPECTED_FRAMES
    }
    report = {
        "schema": "v7_one_time_synthetic_reserved_evaluation_v1",
        "development_only_checkpoint": True,
        "independent_physical_gt": False,
        "same_oxTS_gps_used_to_synthesize_corruption": True,
        "locked_methods": list(CASES),
        "max_optimizer_passes": MAX_PASSES,
        "common_initialization": "LiDAR-integrated, zero pose prior",
        "no_development_saved_pose_initialization": True,
        "post_release_timestamp_engineering_exception": (
            "0020 synchronized streams first 3 frames: [-0.150,-0.100)s, "
            "all other frames <=0.100s; original indices and timestamps "
            "unchanged; adopted after first attempted release"
        ),
        "recovered_from_interrupted_initial_release": bool(recovered),
        "sequences": {},
    }
    for seq, candidates in solved.items():
        with np.load(
            DEST / seq / "evaluation_reference.npz",
            allow_pickle=False,
        ) as z:
            clean = np.asarray(z["original_oxts_gps_local"], np.float64)
        report["sequences"][seq] = {}
        for case, res in candidates.items():
            poses = res["poses"]
            scores = origin_relative_diagnostics(
                poses[:, :3, 3], clean)
            report["sequences"][seq][case] = {
                "evaluation": scores,
                "solve": res["solve"],
                "factor_counts": res["factor_counts"],
                "sigma_metadata": res["sigma_metadata"],
                "checkpoint_provenance": res["checkpoint_provenance"],
            }
    (DEST / "heldout_scores.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (DEST / "RUN_INCOMPLETE").unlink()
    print(json.dumps(report["sequences"], indent=2), flush=True)
    print(
        "V7_HELDOUT_SYNTHETIC_EVALUATION_COMPLETE;"
        " BOTH_DRIVES; NO_INDEPENDENT_PHYSICAL_GT; NO_HYPERPARAM_TUNING",
        flush=True)
    return report



def resume_interrupted_0020_startup():
    """One authorized recovery, preserve ALL original 0016 built bytes.

    This is a post-release engineering fix. It does not restore the
    benchmark to a completely preregistered/untouched status, and it
    must be disclosed as an interrupted run with a timestamp exception.
    """
    status = metadata_only_preflight()
    if not status["all_ready"] or not status["output_exists"]:
        raise RuntimeError("incomplete V7 run or input metadata absent")
    marker = DEST / "RUN_INCOMPLETE"
    if (not marker.is_file()
            or marker.read_text(encoding="utf-8") !=
            "Heldout released; do not rerun without audit.\n"):
        raise RuntimeError("original sealed release marker mismatch")
    if ((DEST / "heldout_scores.json").exists()
            or (DEST / "0020").exists()):
        raise RuntimeError(
            "0020 output or heldout scores already exists: no retry allowed")
    first = DEST / "0016"
    manifest_path = first / "build_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("original 0016 build manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "v7_reserved_synthetic_four_sensor_v1"
            or manifest.get("sequence") != "0016"
            or manifest.get("frames") != 279
            or manifest.get("date_calibration") != "2011_09_30"
            or manifest.get("degradation_seed") != 20260826
            or manifest.get("status") != "BUILT_NOT_EVALUATED"):
        raise RuntimeError("0016 original built-provenance mismatch")
    files = [
        first / file for file in (
            "degraded_sensor_data.npz",
            "evaluation_reference.npz",
            "degraded_lidar_factor_data.npz",
            "degraded_camera_factor_data.npz",
            "build_manifest.json",
        )
    ]
    # Never deserialize/read the saved clean reference before BOTH
    # sequences and all graph methods have solved.
    if any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise RuntimeError("original 0016 build artifacts incomplete")
    existing = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in files
    }
    policy = read_policy()
    validate_policy(policy)
    fit = load_train_axis_noise(policy)
    plan = load_balanced_plan(DEV_PLAN_DIR)
    print(
        "AUDITED_RESUME_0020_STARTUP3",
        "preserve_existing_0016=True",
        "no_prior_scores=True",
        flush=True,
    )
    # Resume starts from the missing SECOND drive only, no touching 0016.
    _build_one("0020", plan, DEST / "0020")
    for p in files:
        if (p.stat().st_size, p.stat().st_mtime_ns) != existing[p.name]:
            raise RuntimeError("original 0016 artifact changed during recovery")
    return _finish_sealed_evaluation(fit, recovered=True)



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true",
                        help="metadata-only, never open heldout content")
    action.add_argument("--release-all", action="store_true",
                        help="ONE-TIME frozen release of BOTH drives")
    action.add_argument(
        "--resume-0020-startup3", action="store_true",
        help="POST-RELEASE audited recovery; preserve built 0016, no scores",
    )
    p.add_argument(
        "--release-token", default=None,
        help="must equal SEALED_V7_0016_0020 to release both",
    )
    a = p.parse_args(argv)
    if a.preflight:
        if a.release_token is not None:
            p.error("preflight cannot include release token")
        status = metadata_only_preflight()
        print(json.dumps(status, indent=2), flush=True)
        if not status["all_ready"]:
            raise SystemExit(1)
        print(
            "V7_HELDOUT_METADATA_ONLY_READY; NO_RESERVED_CONTENT_READ",
            flush=True)
        return
    if a.resume_0020_startup3:
        if a.release_token != "AUDITED_V7_0020_STARTUP3":
            p.error("explicit post-release recovery token required")
        resume_interrupted_0020_startup()
        return
    if a.release_token != "SEALED_V7_0016_0020":
        p.error("explicit sealed release token required")
    release_all()


if __name__ == "__main__":
    main()
