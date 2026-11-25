"""V9.4.2 sealed cross-drive synthetic benchmark.

--preflight verifies locked inputs/models; --solve attempts all 160 graphs
without loading evaluation reference arrays; --score requires a sealed
complete attempt ledger. V7 policy and all prepared files stay unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import gtsam
import torch

from experiments.isj_v8.backend_v9 import (
    load_lock, LOCK_SHA, map_h0, build_graphs, solve_uniform,
)
from experiments.isj_v8.backend_c8_numeric import yaw_aligned_seed, poses_to_values
from experiments.isj_v8.prepare_v9_sequences import (
    ROOT, OUT as PREPARED, PROTOCOL_SHA, load_protocol, hashes, verify_arrays,
)
from experiments.isj_v8.lock_v9_train_methods import source_check
from experiments.isj_v7.cross_sequence_v7 import _positions
from experiments.isj_v7.run_gps_covariance_pilot import rebase_frozen_v7_measurements
from experiments.isj_v7.train_gps_dev import (
    load_frozen_v7_gps_checkpoint, predict_frozen_gps_features_causally,
)
from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph
from src.multisensor_reliability.config import MultiSensorReliabilityConfig
from src.multisensor_reliability.features import gps_features

PREPARED_SHA="4a5232d4c3ea4f3374cdb70032f984014a3c37795d6cb389c8d30421bd96502b"
BACKEND_SHA="1a8dc96f3fb63507d29d7c9f168b4893b1a56c13e537c48bd61b9f670a5b4397"
BUILDER_SHA="cd7afe6770c1f426be298f3205be2923ce55b918d1245617cfeaa3d1e780ec49"
OUT=ROOT/"results/isj_v9/final_evaluation_v1"
PRIMARY="origin_aligned_oxTS_relative_3d_rmse_m"
MODELS=("gru96_r","gru106_r","mamba_r")
SEEDS=(43,44,45)
METHODS=["axis_fixed","axis_fixed_huber"]+[
    f"{m}__seed{s}__{fit}" for m in MODELS for s in SEEDS
    for fit in ("original","calibrated")]
METRICS=(PRIMARY,"origin_aligned_oxTS_relative_2d_rmse_m",
         "relative_displacement_1_frame_rmse_m","relative_displacement_10_frame_rmse_m",
         "mean_3d_m","median_3d_m","p95_3d_m","max_3d_m","vertical_rmse_m")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path,obj):
    data=json.dumps(obj,indent=2,allow_nan=False)+"\n"
    with Path(path).open("x") as f:
        f.write(data)


def identity_array(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def input_manifest():
    protocol=load_protocol()
    lock=load_lock()
    if protocol["methods"]!=METHODS:
        raise ValueError("method matrix drift")
    if sha(Path(__file__).with_name("backend_v9.py"))!=BACKEND_SHA:
        raise ValueError("backend changed")
    if sha(Path(__file__).with_name("prepare_v9_sequences.py"))!=BUILDER_SHA:
        raise ValueError("builder changed")
    p=PREPARED/"prepared_manifest.json"
    if sha(p)!=PREPARED_SHA:
        raise ValueError("prepared manifest drift")
    manifest=read_json(p)
    if (manifest["protocol_sha256"]!=PROTOCOL_SHA or manifest["builder_sha256"]!=BUILDER_SHA
            or manifest["test_evaluation_released"] is not False):
        raise ValueError("invalid prepared manifest")
    entries={}
    for name,config in protocol["new_sequences"].items():
        for seed in protocol["new_corruption_seeds"]:
            tag=f"{name}__seed{seed}"
            rec=manifest["new_realizations"][tag]
            path=PREPARED/tag
            if rec!=read_json(path/"receipt.json") or hashes(path)!=rec["hashes"]:
                raise ValueError("prepared receipt/hash drift: "+tag)
            n=config["frames"]
            if rec["frames"]!=n or rec["seed"]!=seed or rec["sequence"]!=name:
                raise ValueError("prepared metadata mismatch")
            counts=verify_arrays(path,n,config["lidar_disabled_pair_start_ids"],
                                 config["camera_disabled_pair_start_ids"])
            if counts!=rec["valid_counts"]:
                raise ValueError("prepared valid-count drift")
            entries[tag]=dict(path=path,n=n,seed=seed,sequence=name,cohort="new",
                              hashes=rec["hashes"])
    for sid,n in (("0016",279),("0020",1104)):
        rec=manifest["historical_rechecks"][sid]
        path=ROOT/rec["path"]
        if path!=ROOT/"results/isj_v7/heldout_cap4_v1"/sid or rec["frames"]!=n:
            raise ValueError("historical path or frame count drift")
        if hashes(path)!=rec["hashes"]:
            raise ValueError("historical bytes changed")
        verify_arrays(path,n)
        name=f"2011_09_30_drive_{sid}_sync"
        tag=f"{name}__seed20260826"
        entries[tag]=dict(path=path,n=n,seed=20260826,sequence=name,cohort="historical",
                          hashes=rec["hashes"])
    if len(entries)!=8 or len(METHODS)*len(entries)!=160:
        raise ValueError("matrix cardinality drift")
    return lock,entries


def load_models(lock):
    source_check()
    if md.version("gtsam")!="4.2" or not torch.cuda.is_available():
        raise RuntimeError("frozen GTSAM 4.2 / CUDA environment required")
    models={}
    for m in MODELS:
        for seed in SEEDS:
            tag=f"{m}__seed{seed}"
            record=lock["checkpoint_records"][tag]
            model,med,iqr,info=load_frozen_v7_gps_checkpoint(m,seed=seed,device="cuda")
            if (info!=record["checkpoint_provenance"]
                    or info["checkpoint_sha256"]!=record["checkpoint_sha256"]
                    or info["dataset_sha256"]!=lock["train_archive_sha256"]
                    or info["parameters"]!=record["parameters"]):
                raise ValueError("frozen checkpoint mismatch: "+tag)
            model.eval()
            models[tag]=(model,med,iqr,info)
            print("CHECKPOINT_OK",tag,info["checkpoint_sha256"],flush=True)
    return models


def sigma_matrix(gps,lock,models):
    n=len(gps)
    original=lock["original_oracle_mapping"]
    fixed=np.asarray(original["sigma_fixed_xyz"],float)
    result={"axis_fixed":np.tile(fixed,(n,1)),"axis_fixed_huber":np.tile(fixed,(n,1))}
    cfg=MultiSensorReliabilityConfig()
    features=gps_features(gps,dt=cfg.dt,window=cfg.gps_window)
    if features.shape!=(n,21) or not np.isfinite(features).all():
        raise ValueError("invalid GPS features")
    preds={}
    for tag,(model,med,iqr,info) in models.items():
        a,p,_=predict_frozen_gps_features_causally(
            model,features,med,iqr,window=64,clip=info["input_clip"],batch_size=64)
        h0=p[:,0].copy()
        preds[tag]=h0
        result[tag+"__original"]=map_h0(a,h0,n,original,fixed)
        result[tag+"__calibrated"]=map_h0(a,h0,n,lock["calibrated_h0_mappings"][tag],fixed)
    if set(result)!=set(METHODS):
        raise ValueError("sigma method mismatch")
    return result,preds


def graph_inputs(entry,lock):
    m=rebase_frozen_v7_measurements(load_degraded_four_sensor_measurements(entry["path"]))
    n=entry["n"]
    if len(m.gps_local)!=n:
        raise ValueError("frame count drift")
    fixed=np.asarray(lock["original_oracle_mapping"]["sigma_fixed_xyz"],float)
    legacy,initial,counts=run_four_sensor_factor_graph(
        m,"NO_GRAPH_WRITES",mode="fixed",gps_sigmas_override=np.tile(fixed,(n,1)),
        build_graph_only=True)
    counts={k:int(v) for k,v in counts.items()}
    if counts["gps"]!=n or counts["imu"]!=n-1 or legacy.size()!=1+sum(counts.values()):
        raise ValueError("factor count drift")
    seed,alignment=yaw_aligned_seed(gtsam,_positions(gtsam,initial,n),m.gps_local)
    poses=_positions(gtsam,seed,n)
    return m,legacy,counts,poses,alignment


def fingerprint():
    # Runtime fingerprint protects interrupted-run reuse (not a claim of
    # bitwise repeatability on different GPU/library versions).
    paths=("experiments/isj_v8/backend_v9.py","experiments/isj_v8/backend_c8_numeric.py",
           "experiments/isj_v8/solver_health_v8.py","experiments/isj_v8/run_c8_weight_controls.py",
           "experiments/isj_v7/train_gps_dev.py","experiments/isj_v7/run_gps_covariance_pilot.py",
           "src/factor_graph/four_sensor_graph.py","src/factor_graph/degraded_measurements.py",
           "src/multisensor_reliability/features.py","src/multisensor_reliability/config.py")
    return dict(runner_sha256=sha(Path(__file__)),source_hashes={p:sha(ROOT/p) for p in paths},
                method_lock_sha256=LOCK_SHA,protocol_sha256=PROTOCOL_SHA,prepared_sha256=PREPARED_SHA,
                packages={k:md.version(k) for k in ("gtsam","numpy","scipy","torch")},
                cuda=str(torch.version.cuda),gpu=torch.cuda.get_device_name(0),
                methods=METHODS,device="cuda",batch_size=64,
                reference_policy="score_only_after_all_160_attempts",
                common_seed="full_window_degraded_GPS_yaw_LiDAR_with_camera_fallback")


def ensure_identical(path,obj):
    if path.exists():
        if read_json(path)!=obj:
            raise ValueError("locked execution metadata drift: "+str(path))
    else:
        write_new(path,obj)


def checked_case(path,expected):
    rec=read_json(path/"case.json")
    if rec["input"]!=expected:
        raise ValueError("case input drift: "+str(path))
    if rec["status"] not in ("STABLE","NOT_STABLE","SOLVER_ERROR"):
        raise ValueError("unrecognized case status")
    if rec["status"]=="SOLVER_ERROR":
        if (path/"poses.npy").exists() or rec.get("poses_sha256") is not None:
            raise ValueError("failed case contains poses")
    else:
        if sha(path/"poses.npy")!=rec["poses_sha256"]:
            raise ValueError("pose hash drift")
        a=np.load(path/"poses.npy",allow_pickle=False)
        if a.shape!=(expected["frames"],4,4) or not np.isfinite(a).all():
            raise ValueError("invalid saved poses")
        health=rec["health"]
        if (not health["derivative_audit"]["passed"]
                or bool(health["last_step_stable"])!=(rec["status"]=="STABLE")):
            raise ValueError("invalid saved health")
    return rec


def case_input(tag,entry,method,counts,seed_pose,sigma):
    return dict(realization=tag,method=method,frames=entry["n"],factor_counts=counts,
                data_hashes=entry["hashes"],seed_sha256=identity_array(seed_pose),
                sigma_sha256=identity_array(sigma),huber=method=="axis_fixed_huber")


def solve_case(dst,expected,legacy,gps,sigma,seed_pose):
    if dst.exists():
        rec=checked_case(dst,expected)
        print("REUSE_CASE",expected["realization"],expected["method"],rec["status"],flush=True)
        return rec
    native,corrected,keepalive,nb=build_graphs(
        gtsam,legacy,gps,sigma,huber=expected["huber"])
    n=expected["frames"]
    if nb!=sum(expected["factor_counts"][k] for k in ("imu","lidar","camera")):
        raise ValueError("relative factor drift")
    seed=poses_to_values(gtsam,seed_pose)
    if not np.array_equal(_positions(gtsam,seed,n),seed_pose):
        if not np.allclose(_positions(gtsam,seed,n),seed_pose,rtol=0,atol=1e-12):
            raise ValueError("seed reconstruction drift")
    stage=Path(tempfile.mkdtemp(prefix=".incomplete_",dir=dst.parent))
    start=time.monotonic()
    try:
        health=solve_uniform(gtsam,native,corrected,seed,n)
    except RuntimeError as exc:
        # A recorded numerical failure, never retried with alternate
        # parameters/initialization. Other exceptions stop the run.
        rec=dict(input=expected,status="SOLVER_ERROR",error=str(exc),
                 poses_sha256=None,elapsed_s=time.monotonic()-start)
    else:
        poses=health.pop("poses")
        if poses.shape!=(n,4,4) or not np.isfinite(poses).all():
            raise ValueError("invalid optimizer output")
        np.save(stage/"poses.npy",poses,allow_pickle=False)
        rec=dict(input=expected,status="STABLE" if health["last_step_stable"] else "NOT_STABLE",
                 health=health,poses_sha256=sha(stage/"poses.npy"),
                 elapsed_s=time.monotonic()-start)
    write_new(stage/"case.json",rec)
    checked_case(stage,expected)
    if dst.exists():
        raise FileExistsError(dst)
    stage.rename(dst)
    print("CASE_COMPLETE",expected["realization"],expected["method"],rec["status"],
          "seconds=",round(rec["elapsed_s"],3),flush=True)
    return rec


def seal_solves(lock,entries,models,fp):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"score_report.json").exists():
        raise ValueError("scores already released; no further solves")
    ensure_identical(OUT/"execution_lock.json",fp)
    ledger={}
    for tag,entry in entries.items():
        dest=OUT/tag
        dest.mkdir(exist_ok=True)
        m,legacy,counts,seed_pose,alignment=graph_inputs(entry,lock)
        sigmas,preds=sigma_matrix(m.gps_local,lock,models)
        context=dict(factor_counts=counts,alignment=alignment,seed_sha256=identity_array(seed_pose),
                     sigma_hashes={k:identity_array(v) for k,v in sigmas.items()},
                     h0_hashes={k:identity_array(v) for k,v in preds.items()})
        ensure_identical(dest/"context.json",context)
        # Save shared inputs once for reproducibility; no reference.
        p=dest/"inputs.npz"
        arrays={"common_seed":seed_pose,"degraded_gps":m.gps_local}
        arrays.update({"sigma__"+k:v for k,v in sigmas.items()})
        arrays.update({"h0__"+k:v for k,v in preds.items()})
        if p.exists():
            with np.load(p,allow_pickle=False) as z:
                if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],v) for k,v in arrays.items()):
                    raise ValueError("saved inputs changed")
        else:
            with p.open("xb") as f:
                np.savez_compressed(f,**arrays)
        print("REALIZATION_START",tag,counts,flush=True)
        ledger[tag]={"inputs_sha256":sha(p),"context_sha256":sha(dest/"context.json"),"cases":{}}
        for method in METHODS:
            expected=case_input(tag,entry,method,counts,seed_pose,sigmas[method])
            rec=solve_case(dest/method,expected,legacy,m.gps_local,sigmas[method],seed_pose)
            ledger[tag]["cases"][method]=dict(status=rec["status"],
                    case_sha256=sha(dest/method/"case.json"),poses_sha256=rec["poses_sha256"])
        if hashes(entry["path"])!=entry["hashes"]:
            raise ValueError("source data changed")
    totals={k:sum(r["status"]==k for v in ledger.values() for r in v["cases"].values())
            for k in ("STABLE","NOT_STABLE","SOLVER_ERROR")}
    if sum(totals.values())!=160:
        raise ValueError("incomplete case matrix")
    ensure_identical(OUT/"solve_ledger.json",dict(schema="isj_v9_solve_ledger_v1",
        execution_lock_sha256=sha(OUT/"execution_lock.json"),cases=ledger,totals=totals,
        reference_arrays_opened_for_scoring=False))
    print("SOLVE_TOTALS",json.dumps(totals),flush=True)
    print("SOLVE_LEDGER_SHA256",sha(OUT/"solve_ledger.json"),flush=True)
    print("V9_4_2_SOLVES_COMPLETE 160_ATTEMPTS NO_SCORES",flush=True)


def metrics(trajectory,clean):
    t,c=np.asarray(trajectory,float),np.asarray(clean,float)
    if (t.ndim!=2 or t.shape[1]!=3 or len(t)<=10 or c.shape!=t.shape
            or not np.isfinite(t).all() or not np.isfinite(c).all()):
        raise ValueError("invalid metric trajectory")
    c=c-c[0].copy()
    diff=t-c
    def rmse(x):
        return float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
    d=np.linalg.norm(diff,axis=1)
    return {PRIMARY:rmse(diff),
            "origin_aligned_oxTS_relative_2d_rmse_m":rmse(diff[:,:2]),
            "relative_displacement_1_frame_rmse_m":rmse(np.diff(t,axis=0)-np.diff(c,axis=0)),
            "relative_displacement_10_frame_rmse_m":rmse(t[10:]-t[:-10]-(c[10:]-c[:-10])),
            "mean_3d_m":float(d.mean()),"median_3d_m":float(np.median(d)),
            "p95_3d_m":float(np.percentile(d,95)),"max_3d_m":float(d.max()),
            "vertical_rmse_m":float(np.sqrt(np.mean(diff[:,2]**2)))}


def score(entries,fp,ledger_sha):
    ledger_path=OUT/"solve_ledger.json"
    if sha(ledger_path)!=ledger_sha:
        raise ValueError("explicit solve ledger SHA mismatch")
    ledger=read_json(ledger_path)
    if (read_json(OUT/"execution_lock.json")!=fp
            or ledger["execution_lock_sha256"]!=sha(OUT/"execution_lock.json")
            or set(ledger["cases"])!=set(entries)
            or sum(ledger["totals"].values())!=160):
        raise ValueError("unsealed or changed run")
    # Verify the entire matrix before opening even one reference array.
    for tag,entry in entries.items():
        base=OUT/tag
        group=ledger["cases"][tag]
        if (sha(base/"inputs.npz")!=group["inputs_sha256"]
                or sha(base/"context.json")!=group["context_sha256"]
                or set(group["cases"])!=set(METHODS)):
            raise ValueError("incomplete realization ledger")
        for method,r in group["cases"].items():
            case=base/method
            if sha(case/"case.json")!=r["case_sha256"]:
                raise ValueError("case report changed")
            rec=read_json(case/"case.json")
            checked_case(case,rec["input"])
            if (rec["status"]!=r["status"] or rec["poses_sha256"]!=r["poses_sha256"]
                    or rec["input"]["method"]!=method or rec["input"]["realization"]!=tag
                    or rec["input"]["data_hashes"]!=entry["hashes"]):
                raise ValueError("case ledger mismatch")
    if (OUT/"score_report.json").exists() or (OUT/"metrics.csv").exists():
        raise FileExistsError("refuse score overwrite")
    rows=[]
    for tag,entry in entries.items():
        with np.load(entry["path"]/"evaluation_reference.npz",allow_pickle=False) as z:
            clean=np.asarray(z["original_oxts_gps_local"],float)
        if clean.shape!=(entry["n"],3) or not np.isfinite(clean).all():
            raise ValueError("invalid reference")
        base=OUT/tag
        with np.load(base/"inputs.npz",allow_pickle=False) as z:
            degraded=z["degraded_gps"].copy()
        common=dict(realization=tag,sequence=entry["sequence"],cohort=entry["cohort"],
                    corruption_seed=entry["seed"],frames=entry["n"])
        rows.append(dict(**common,method="degraded_gps_only",status="RAW_BASELINE",**metrics(degraded,clean)))
        for method,r in ledger["cases"][tag]["cases"].items():
            row=dict(**common,method=method,status=r["status"])
            if r["status"]!="SOLVER_ERROR":
                poses=np.load(base/method/"poses.npy",allow_pickle=False)
                row.update(metrics(poses[:,:3,3],clean))
            rows.append(row)
    fields=["realization","sequence","cohort","corruption_seed","frames","method","status",*METRICS]
    with (OUT/"metrics.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_new(OUT/"score_report.json",dict(schema="isj_v9_final_synthetic_evaluation_v1",
        solve_ledger_sha256=ledger_sha,metrics_csv_sha256=sha(OUT/"metrics.csv"),
        independent_physical_gt=False,offline_batch=True,per_method_alignment=False,
        primary_metric=PRIMARY,solver_status_totals=ledger["totals"],
        historical_rechecks_not_independent=True,rows=rows,
        no_test_result_based_tuning=True))
    print("SCORED_ROWS",len(rows),flush=True)
    print("SCORE_REPORT_SHA256",sha(OUT/"score_report.json"),flush=True)
    print("V9_4_2_SCORING_COMPLETE",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument("--preflight",action="store_true")
    a.add_argument("--solve",action="store_true")
    a.add_argument("--score",action="store_true")
    p.add_argument("--token")
    p.add_argument("--ledger-sha")
    args=p.parse_args()
    if args.solve and args.token!="SOLVE_V9_FROZEN_160":
        p.error("explicit solve token required")
    if args.score and (args.token!="SCORE_V9_SEALED_160" or not args.ledger_sha):
        p.error("explicit score token and solve ledger SHA required")
    if args.preflight and (args.token or args.ledger_sha):
        p.error("preflight cannot release")
    lock,entries=input_manifest()
    fp=fingerprint()
    if args.score:
        score(entries,fp,args.ledger_sha)
        return
    models=load_models(lock)
    if args.preflight:
        # Graph construction and alignment checks only; no network inference
        # or graph optimization, and no score/reference array access.
        for tag,entry in entries.items():
            _,_,counts,_,alignment=graph_inputs(entry,lock)
            print("GRAPH_PREFLIGHT_OK",tag,counts,alignment,flush=True)
        print("V9_4_2_PREFLIGHT_OK 9_MODELS 8_REALIZATIONS 160_CASES NO_INFERENCE NO_SOLVES NO_SCORES",flush=True)
        return
    seal_solves(lock,entries,models,fp)


if __name__=="__main__":
    main()
