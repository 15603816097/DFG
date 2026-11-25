"""V9.4.1 data construction only. No network inference, FGO or score evaluation.

New data release explicitly authorised by --build --token BUILD_V9_FROZEN_DATA.
Historical V7 artifacts are referenced in place, not regenerated or altered.
Reference arrays are created separately from graph measurement files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from experiments.isj_v8.backend_v9 import load_lock, LOCK_SHA
from experiments.isj_v7.supervision_gate import read_policy, validate_policy
from experiments.isj_v7.cross_sequence_v7 import (
    DEV_PLAN_DIR, validate_frozen_time_alignment,
)
from experiments.degradation.build_degraded_four_sensor_measurements import (
    corrupt_local_gps, lidar_quality, camera_quality, extract_lidar, ts_seconds,
)
from src.factor_graph.four_sensor_graph import gps_geodetic_to_local
from src.loader.gps_loader import GPSLoader
from src.loader.imu_loader import IMULoader
from src.loader.lidar_loader import LidarLoader
from src.common.time_utils import load_kitti_timestamps
from src.multisensor_reliability.degradation.apply import corrupt_imu, corrupt_lidar, corrupt_camera
from src.preprocessing.calibration import KITTICalibration
from src.odometry.body_frame_converter import BodyFrameConverter
from src.odometry.lidar_odometry import LidarOdometry
from src.odometry.stereo_visual_odometry import StereoVisualOdometry
from src.odometry.relative_factor_data import make_factor_data

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset/kitti"
DOWNLOADS = ROOT / "dataset/kitti_downloads"
PROTOCOL = ROOT / "results/isj_v9/final_protocol_v1/protocol.json"
OUT = ROOT / "results/isj_v9/prepared_sequences_v1"
HIST = ROOT / "results/isj_v7/heldout_cap4_v1"
PROTOCOL_SHA = "c34961d584b12175d71f08085ddca6346f9501b01c2703eedfcf2b91de8390bf"
PLAN_SHA = "6e371773810bf792c1fca91862545ce2824c1cf1f48520adcabb282e24b4cc84"
ARCHIVES = {
    "2011_09_26_calib.zip":"e0108cfd000cf802c14ed94fba38601185c792e5640abba015a34b7a85b812e0",
    "2011_09_26_drive_0009_sync.zip":"909639de40540f1d9bd8e3c181d79556ee4852781aa4684999f4cda3f3ef2611",
    "2011_09_26_drive_0023_sync.zip":"a9436a3da5199821712b0341dbabc9b88d3f6fa62bde38935734577c9f3a9d97",
    "2011_09_29_calib.zip":"260fb836916ac3cbbab7104e253531239a0191b87c411fa60b99178abbb8c40d",
    "2011_09_29_drive_0004_sync.zip":"f7c1226b2402416dfe066d86b375fc690cddf7ff24f7d6c80b144223ed63565f",
}
NEW = {
    "2011_09_26_drive_0009_sync": (447, (177,178,179,180)),
    "2011_09_26_drive_0023_sync": (474, ()),
    "2011_09_29_drive_0004_sync": (339, ()),
}
HISTORY = {"0016":279,"0020":1104}
SEEDS = (20261126,20261127)
FILES = ("degraded_sensor_data.npz","degraded_lidar_factor_data.npz",
         "degraded_camera_factor_data.npz","evaluation_reference.npz","build_manifest.json")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()


def frame_index(paths,n,missing=()):
    ids = [int(p.stem) for p in paths]
    if any(p.stem != f"{k:010d}" for p,k in zip(paths,ids)):
        raise ValueError("invalid frame filename")
    expected = sorted(set(range(n))-set(missing))
    if ids != expected:
        raise ValueError("frame inventory differs from frozen IDs")
    return {k:j for j,k in enumerate(ids)}


def pair_disabled(n,missing):
    missing=set(missing)
    return [i for i in range(n-1) if i in missing or i+1 in missing]


def load_protocol():
    load_lock()
    validate_policy(read_policy())
    if sha256(PROTOCOL)!=PROTOCOL_SHA:
        raise ValueError("protocol hash mismatch")
    p=json.loads(PROTOCOL.read_text())
    if (p["method_lock_sha256"]!=LOCK_SHA or p["test_release"] is not False
            or p["planned_total_solves"]!=160 or len(p["methods"])!=20
            or p["new_corruption_seeds"]!=list(SEEDS)
            or p["historical_corruption_seed"]!=20260826
            or set(p["new_sequences"])!=set(NEW)
            or p["source_degradation_plan_sha256"]!=PLAN_SHA
            or sha256(DEV_PLAN_DIR/"degradation_plan_v2.npz")!=PLAN_SHA):
        raise ValueError("frozen protocol mismatch")
    return p


def source_inventory(name,config):
    n,missing=NEW[name]
    date=name[:10]
    seq=RAW/date/name
    if config["frames"]!=n or config["missing_lidar"]!=list(missing):
        raise ValueError("protocol inventory drift")
    for sensor,suffix in (("oxts","txt"),("velodyne_points","bin"),
                          ("image_02","png"),("image_03","png")):
        files=sorted((seq/sensor/"data").glob("*."+suffix))
        frame_index(files,n,missing if sensor=="velodyne_points" else ())
        if any(x.stat().st_size<=0 for x in files):
            raise ValueError("empty sensor file")
        ts=seq/sensor/"timestamps.txt"
        if sha256(ts)!=config["timestamps_sha256"][sensor]:
            raise ValueError("timestamp SHA drift")
    if (config["lidar_disabled_pair_start_ids"]!=pair_disabled(n,missing)
            or config["camera_disabled_pair_start_ids"]!=[]):
        raise ValueError("unexpected availability mask")
    return seq


def verify_raw_archives(*,crc):
    """CRC verifies extracted bytes, without parsing sensor values."""
    for name,digest in ARCHIVES.items():
        path=DOWNLOADS/name
        if sha256(path)!=digest:
            raise ValueError("archive SHA drift: "+name)
        with ZipFile(path) as z:
            for item in z.infolist():
                if item.is_dir(): continue
                rel=PurePosixPath(item.filename)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("unsafe archive member")
                dst=RAW/str(rel)
                if not dst.is_file() or dst.stat().st_size!=item.file_size:
                    raise ValueError("extracted size drift: "+str(dst))
                if crc:
                    checksum=0
                    with dst.open("rb") as f:
                        for block in iter(lambda:f.read(1<<20),b""):
                            checksum=zlib.crc32(block,checksum)
                    if checksum!=item.CRC:
                        raise ValueError("extracted CRC drift: "+str(dst))
        print("RAW_ARCHIVE_OK",name,"extracted_crc_checked=",crc,flush=True)


def verify_arrays(path,n,disabled_lidar=(),disabled_camera=()):
    # Reference is intentionally not deserialized here.
    with np.load(path/FILES[0],allow_pickle=False) as z:
        if set(z.files)!={"degraded_gps_local","degraded_imu_gyro","timestamps_seconds"}:
            raise ValueError("measurement archive schema drift")
        for k,shape in (("degraded_gps_local",(n,3)),("degraded_imu_gyro",(n,3)),("timestamps_seconds",(n,))):
            if z[k].shape!=shape or not np.isfinite(z[k]).all():
                raise ValueError("invalid sensor array: "+k)
        if np.any(np.diff(z["timestamps_seconds"])<=0):
            raise ValueError("invalid timestamp order")
    counts={}
    for modality,disabled in (("lidar",disabled_lidar),("camera",disabled_camera)):
        with np.load(path/f"degraded_{modality}_factor_data.npz",allow_pickle=False) as z:
            for k,shape in (("coordinate_transforms",(n-1,4,4)),("between_measurements",(n-1,4,4)),
                            ("valid",(n-1,)),("quality",(n-1,))):
                if z[k].shape!=shape or not np.isfinite(z[k]).all():
                    raise ValueError("invalid relative factor array: "+k)
            valid=z["valid"]
            q=z["quality"]
            if valid.dtype!=np.bool_ or np.any((q<0)|(q>1)):
                raise ValueError("invalid factor mask/quality")
            if disabled and (np.any(valid[list(disabled)]) or np.any(q[list(disabled)]!=0)):
                raise ValueError("disabled factor became valid")
            counts[modality+"_valid"]=int(valid.sum())
    return counts


def hashes(path):
    return {name:sha256(path/name) for name in FILES}


def reconcile_historical_audit(sid, manifest, recomputed):
    """Preserve old 0016 bytes; only its pre-recovery schema may lack audit."""
    if "timestamp_alignment_audit" not in manifest:
        if sid != "0016" or recomputed["startup_exception_applied"]:
            raise ValueError("unexpected missing historical timestamp audit: "+sid)
        return "recomputed_from_raw_timestamps_legacy_0016"
    if manifest["timestamp_alignment_audit"] != recomputed:
        raise ValueError("historical timestamp audit differs: "+sid)
    return "archived_audit_verified_against_raw_timestamps"


def historical_time_audit(sid,n,path):
    seq=RAW/"2011_09_30"/f"2011_09_30_drive_{sid}_sync"
    times={}
    stamp_hashes={}
    for sensor,suffix,key in (
            ("oxts","txt","oxts"),
            ("velodyne_points","bin","velodyne"),
            ("image_02","png","left-camera"),
            ("image_03","png","right-camera")):
        frame_index(sorted((seq/sensor/"data").glob("*."+suffix)),n)
        stamp=seq/sensor/"timestamps.txt"
        times[key]=ts_seconds(load_kitti_timestamps(stamp))
        stamp_hashes[sensor]=sha256(stamp)
    ref=times.pop("oxts")
    audit=validate_frozen_time_alignment(sid,ref,times)
    # Check that the archived graph clock is still the original OXTS clock.
    # Only timestamp array opened, no positions or reference deserialization.
    with np.load(path/FILES[0],allow_pickle=False) as z:
        saved=np.asarray(z["timestamps_seconds"],float)
    if saved.shape!=(n,) or not np.allclose(saved,ref,rtol=0.,atol=1e-6):
        raise ValueError("historical archived clock drift: "+sid)
    return audit,stamp_hashes


def historical():
    if (HIST/"RUN_INCOMPLETE").exists() or not (HIST/"heldout_scores.json").is_file():
        raise ValueError("historical V7 build not complete")
    entries={}
    for sid,n in HISTORY.items():
        path=HIST/sid
        m=json.loads((path/"build_manifest.json").read_text())
        if (m["schema"]!="v7_reserved_synthetic_four_sensor_v1" or m["sequence"]!=sid
                or m["frames"]!=n or m["degradation_seed"]!=20260826
                or m["status"]!="BUILT_NOT_EVALUATED"):
            raise ValueError("historical manifest drift")
        audit,stamp_hashes=historical_time_audit(sid,n,path)
        audit_origin=reconcile_historical_audit(sid,m,audit)
        entries[sid]=dict(path=str(path.relative_to(ROOT)),frames=n,
                          source="previously_evaluated_v7_recheck",
                          timestamp_alignment_audit=audit,
                          timestamp_audit_origin=audit_origin,
                          timestamps_sha256=stamp_hashes,
                          hashes=hashes(path))
        print("HISTORICAL_ARTIFACTS_OK",sid,audit_origin,
              "policy=",audit["policy"],flush=True)
    return entries


def _build_one(sequence,config,plan,seed,out):
    import cv2
    cv2.setNumThreads(1)
    cv2.setRNGSeed(seed)
    seq=source_inventory(sequence,config)
    date_dir=seq.parent
    n=config["frames"]
    gl,il,ll=GPSLoader(seq),IMULoader(seq),LidarLoader(seq)
    frame_index(gl.files,n)
    frame_index(il.files,n)
    frame_to_index=frame_index(ll.files,n,config["missing_lidar"])
    left=sorted((seq/"image_02/data").glob("*.png"))
    right=sorted((seq/"image_03/data").glob("*.png"))
    frame_index(left,n)
    frame_index(right,n)
    lidar_disabled=set(config["lidar_disabled_pair_start_ids"])
    camera_disabled=set(config["camera_disabled_pair_start_ids"])
    pp={"severity":{k:np.asarray(v,dtype=np.float32) for k,v in plan["severity"].items()},
        "mode":{k:np.asarray(v,dtype=np.int64) for k,v in plan["mode"].items()}}
    for sensor in ("gps","imu","lidar","camera"):
        if (pp["severity"][sensor].shape!=(n,) or pp["mode"][sensor].shape!=(n,)
                or not np.isfinite(pp["severity"][sensor]).all()
                or np.any((pp["severity"][sensor]<0)|(pp["severity"][sensor]>1))):
            raise ValueError("invalid frozen degradation plan")
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

    calib = KITTICalibration(date_dir)
    converter = BodyFrameConverter(calib)
    lidar = LidarOdometry(
        max_iterations=20, max_correspondence_distance=1.5,
        min_correspondences=80, max_points=6000,
    )
    cam = StereoVisualOdometry(
        str(date_dir / "calib_cam_to_cam.txt"),
        num_features=2500, min_matches=30, min_inliers=20,
        ratio_test=.75, disparity_num=128,
        disparity_block_size=5, max_depth=80,
    )
    lc, lb, lv, lq = [], [], [], []
    previous = np.eye(4)
    print("BUILD_LIDAR_START", sequence, "pairs=", n-1, flush=True)
    for i in range(n-1):
        if i in lidar_disabled:
            lc.append(np.eye(4)); lb.append(np.eye(4)); lv.append(False); lq.append(0.)
            previous = np.eye(4)
            continue
        p = corrupt_lidar(
            extract_lidar(ll[frame_to_index[i]]), pp["severity"]["lidar"][i],
            pp["mode"]["lidar"][i], i, seed=seed)
        q = corrupt_lidar(
            extract_lidar(ll[frame_to_index[i+1]]), pp["severity"]["lidar"][i+1],
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
        if i in camera_disabled:
            cc.append(np.eye(4)); cb.append(np.eye(4)); cv.append(False); cq.append(0.)
            continue
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
    # The caller creates a unique staging directory before work starts.
    if not out.is_dir():
        raise ValueError('missing staging directory')
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
        "schema": "isj_v9_prepared_synthetic_v1",
        "sequence": sequence, "frames": n,
        "date_calibration": date_dir.name,
        "timestamp_alignment_audit": config["alignment"],
        "lidar_disabled_pair_start_ids": sorted(lidar_disabled),
        "camera_disabled_pair_start_ids": sorted(camera_disabled),
        "protocol_sha256": PROTOCOL_SHA,
        "builder_sha256": sha256(Path(__file__)),
        "opencv_rng_seed": seed,
        "opencv_threads": 1,
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



def receipt(path,name,seed,config):
    p=path/"receipt.json"
    manifest=json.loads((path/"build_manifest.json").read_text())
    if (manifest["protocol_sha256"]!=PROTOCOL_SHA or manifest["sequence"]!=name
            or manifest["degradation_seed"]!=seed or manifest["frames"]!=config["frames"]
            or manifest["builder_sha256"]!=sha256(Path(__file__))):
        raise ValueError("prepared manifest drift")
    record=dict(sequence=name,seed=seed,frames=config["frames"],
                protocol_sha256=PROTOCOL_SHA,hashes=hashes(path),
                valid_counts=verify_arrays(path,config["frames"],
                    config["lidar_disabled_pair_start_ids"],config["camera_disabled_pair_start_ids"]))
    if p.is_file():
        if json.loads(p.read_text())!=record:
            raise ValueError("prepared receipt drift")
    else:
        p.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight",action="store_true")
    group.add_argument("--build",action="store_true")
    parser.add_argument("--token")
    args=parser.parse_args()
    if args.build and args.token!="BUILD_V9_FROZEN_DATA":
        parser.error("explicit data construction token required")
    if args.preflight and args.token is not None:
        parser.error("preflight cannot use token")
    p=load_protocol()
    for name,config in p["new_sequences"].items():
        source_inventory(name,config)
    verify_raw_archives(crc=args.build)
    hist=historical()
    if args.preflight:
        print("V9_4_1_PREFLIGHT_OK NO_SENSOR_VALUE_ANALYSIS NO_INFERENCE NO_SOLVES")
        return
    OUT.mkdir(parents=True,exist_ok=True)
    records={}
    for name,config in p["new_sequences"].items():
        for seed in SEEDS:
            tag=f"{name}__seed{seed}"
            dst=OUT/tag
            if dst.exists():
                if not (dst/"receipt.json").is_file():
                    raise ValueError("incomplete published build: "+tag)
                records[tag]=receipt(dst,name,seed,config)
                print("REUSE_VERIFIED",tag,flush=True)
                continue
            stage=Path(tempfile.mkdtemp(prefix=".incomplete_"+tag+"_",dir=OUT))
            print("BUILD_START",tag,"stage=",stage,flush=True)
            _build_one(name,config,p["degradation_plans"][name],seed,stage)
            records[tag]=receipt(stage,name,seed,config)
            if dst.exists():
                raise FileExistsError(dst)
            stage.rename(dst)
            print("PREPARED_OK",tag,records[tag]["valid_counts"],flush=True)
    if historical()!=hist:
        raise ValueError("historical assets changed")
    if len(records)!=6:
        raise ValueError("build count drift")
    report=dict(schema="isj_v9_prepared_data_manifest_v1",protocol_sha256=PROTOCOL_SHA,
                builder_sha256=sha256(Path(__file__)),new_realizations=records,
                historical_rechecks=hist,reference_used_to_synthesize_gps=True,
                evaluation_reference_deserialized_for_scoring=False,
                independent_physical_gt=False,model_inference=False,solver_runs=0,
                test_evaluation_released=False)
    payload=json.dumps(report,indent=2,allow_nan=False)+"\n"
    dest=OUT/"prepared_manifest.json"
    if dest.exists():
        if dest.read_text()!=payload:
            raise ValueError("refuse overwrite prepared manifest")
    else:
        with dest.open("x") as f:
            f.write(payload)
    print("PREPARED_MANIFEST_SHA256",sha256(dest),flush=True)
    print("V9_4_1_BUILD_COMPLETE 6_NEW_REALIZATIONS 2_HISTORICAL_RECHECKS NO_INFERENCE NO_SOLVES NO_SCORES")


if __name__=="__main__":
    main()
