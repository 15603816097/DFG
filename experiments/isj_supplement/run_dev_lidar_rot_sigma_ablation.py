from __future__ import annotations

"""0027 DEVELOPMENT ONLY: isolate LiDAR rotational-factor sigma sensitivity.

Frozen four sensor factors, frozen LiDAR and Camera BetweenPose3
measurements and ALL OTHER fixed factor sigmas. In memory ONLY patch
Stage-6 LiDAR_R0 to predeclared [original, camera scale, 0.1, 0.4] rad.
All cases use identical original Stage-6 LiDAR-based initial poses,
GTSAM 100-iteration LM solver and graph topology. This is a STATIC
SENSITIVITY EXPERIMENT, NOT proposed learning, gating or new algorithm.

NO raw data or OXTS IMU attitude loaded; only original 0027 cached
measurements and existing OXTS/GPS-derived evaluation trajectory.
Never read or optimize on 0016/0020. Results written ONLY under the
fresh results/isj_dev_lidar_rot_sigma_v1 directory; historical saved
results and checkpoints untouched. The original 100-iteration solve
does NOT guarantee equal convergence. Any apparent ATE benefit
needs uniform per-case convergence and test on prospective unseen
sequences before a manuscript claim.

Use --stage audit (no optimization, no files written), then --stage fgo.
Pre-existing incomplete or mismatched cache causes a hard error.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6

OUT=ROOT/"results"/"isj_dev_lidar_rot_sigma_v1"
# rad. 0.012857142857 is the unchanged Stage-6 fixed LiDAR rotation
# noise. 0.04 equals frozen Camera rotation noise. Larger values are
# exploratory sensitivity tests, not data-fitted optimal hyperparameters.
SPECS=(
    ("LIDAR_ROT_ORIGINAL",s6.LIDAR_R0),
    ("LIDAR_ROT_CAMERA_SCALE",s6.CAM_R0),
    ("LIDAR_ROT_0100",0.1),
    ("LIDAR_ROT_0400",0.4),
)
PROTOCOL="dev0027_fixed_fgo_lidar_rot_sigma_sensitivity_v1"
N_EXPECTED=4544


def spec_for(name,sigma_r):
    return {
        "protocol":PROTOCOL,
        "name":name,
        "dataset":"2011_10_03_drive_0027_sync",
        "n_frames":N_EXPECTED,
        "lidar_rot_sigma_rad":float(sigma_r),
        "lidar_trans_sigma_m":float(s6.LIDAR_T0),
        "gps_sigma_m":float(s6.GPS_FIXED),
        "imu_rot_sigma_rad":float(s6.IMU_R0),
        "camera_rot_sigma_rad":float(s6.CAM_R0),
        "camera_trans_sigma_m":float(s6.CAM_T0),
        "sensor_modes":{
            "gps":"fixed","imu":"fixed",
            "lidar":"fixed","camera":"fixed",
        },
        "initialization":"same Stage-6 LiDAR first initial poses",
        "optimizer":"original GTSAM Stage-6 100-iteration LM (pilot)",
        "reference":"original OXTS/GPS-derived local position",
    }


def check_cache(path:Path,expected:dict,n:int):
    """Never silently overwrite unrelated results or partial runs."""
    if not path.exists():
        return None
    pose=path/"poses.npy"
    manifest=path/"isj_rot_sigma_manifest.json"
    if not pose.is_file() or not manifest.is_file():
        raise RuntimeError(
            f"Partial pre-existing result; refusing to overwrite: {path}"
        )
    saved=np.load(pose,allow_pickle=False)
    if saved.shape!=(n,4,4) or not np.isfinite(saved).all():
        raise ValueError(f"Invalid cached poses: {pose}")
    obj=json.loads(manifest.read_text(encoding="utf-8"))
    if obj.get("spec")!=expected:
        raise ValueError(f"Cached result spec mismatch: {path}")
    scores=obj.get("metrics",{})
    if not math.isfinite(float(scores.get("ATE3D",float("nan")))):
        raise ValueError(f"Invalid cached ATE3D: {path}")
    return obj


def compare_measurements(m):
    n=len(m.gps_local)
    if n!=N_EXPECTED:
        raise ValueError(f"Expected reserved 0027 dev n=4544, got {n}")
    lidar=np.asarray(m.lidar_between,float)
    camera=np.asarray(m.camera_between,float)
    if lidar.shape!=(n-1,4,4) or camera.shape!=(n-1,4,4):
        raise ValueError("Bad fixed between measurements")
    if not np.isfinite(lidar).all() or not np.isfinite(camera).all():
        raise ValueError("Nonfinite fixed between measurements")
    v=np.asarray(m.lidar_valid,bool)
    cv=np.asarray(m.camera_valid,bool)
    print(f"Frozen measurements 0027: frames={n} "
          f"LiDAR factors={v.sum()} Camera factors={cv.sum()} "
          f"GPS={n} IMU={n-1}")
    print("All factors, initialization and fixed GPS/IMU/Camera/LiDAR "
          "translation sigmas unchanged")
    print("Output root:",OUT)
    print("Current fixed LiDAR rotational sigma=",s6.LIDAR_R0,"rad")
    print("Original LiDAR vs Camera inverse mismatch is NOT an "
          "independent OXTS convention check")
    print("No data or checkpoint writes; 0016/0020 never used")


def execute_case(m,name,sigma):
    spec=spec_for(name,sigma)
    target=OUT/name
    cached=check_cache(target,spec,len(m.gps_local))
    if cached is not None:
        print(f"{name}: CACHED ATE3D={cached['metrics']['ATE3D']:.6f}m")
        return cached["metrics"]

    # This is a conservative guard: if OUT already has an unrelated
    # directory for a case, check_cache above will refuse overwrite.
    print(f"\nFGO 0027 fixed: {name}, LiDAR rotation sigma={sigma:.6f} rad",
          flush=True)
    with patch.object(s6,"OUT",OUT),patch.object(s6,"LIDAR_R0",float(sigma)):
        metrics=s6.run_case(
            name=name,measurements=m,gps_mode="fixed",
            imu_mode="fixed",camera_mode="fixed",lidar_mode="fixed",
        )
    pose_file=target/"poses.npy"
    if not pose_file.is_file():
        raise RuntimeError(f"Stage-6 did not save poses: {pose_file}")
    poses=np.load(pose_file,allow_pickle=False)
    if poses.shape!=(len(m.gps_local),4,4) or not np.isfinite(poses).all():
        raise ValueError(f"Corrupt newly produced poses: {pose_file}")
    if not math.isfinite(float(metrics["ATE3D"])):
        raise ValueError("Nonfinite ATE3D")
    obj={
        "spec":spec,
        "metrics":metrics,
        "status":"DEV ONLY initial LM, NOT convergence-controlled",
        "notes":[
            "For 0027 only; the learning models trained on this drive.",
            "OXTS/GPS reference is not independent of GPS input.",
            "Rotation weight sensitivity does not establish an innovation.",
            "No OXTS/clean reference used to choose factor sigma.",
            "Do not use prospective 0020 for method or sigma selection.",
        ]
    }
    (target/"isj_rot_sigma_manifest.json").write_text(
        json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8",
    )
    print(f"{name}: ATE3D={metrics['ATE3D']:.6f}m "
          "(not convergence controlled)",flush=True)
    return metrics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("audit","fgo"),required=True)
    args=parser.parse_args(argv)

    m=s6.load_measurements()
    compare_measurements(m)
    print("\nPREDECLARED CASES:")
    for name,sigma in SPECS:
        if not np.isfinite(sigma) or sigma<=0:
            raise ValueError("Invalid predefined LiDAR sigma")
        info=spec_for(name,sigma)
        cached=check_cache(OUT/name,info,len(m.gps_local))
        print(f"  {name:<28s} σ_r={sigma:.6f}rad "
              f"status={'CACHED' if cached else 'PENDING'}")
    if args.stage=="audit":
        print("AUDIT ONLY: no FGO or files written")
        return

    scores={}
    for name,sigma in SPECS:
        scores[name]=execute_case(m,name,sigma)
    baseline=float(scores[SPECS[0][0]]["ATE3D"])
    print("\n0027 DEVELOPMENT SENSITIVITY, INITIAL LM ONLY")
    for name,sigma in SPECS:
        x=float(scores[name]["ATE3D"])
        print(f"{name:<28s} LiDAR_rot_sigma={sigma:.6f} "
              f"ATE3D={x:.6f}m "
              f"delta_vs_original={x-baseline:+.6f}m")
    print("STOP: these results are NOT manuscript-level comparisons "
          "until uniform solver convergence is checked.")


if __name__=="__main__":
    main()
