from __future__ import annotations

"""0027 DEVELOPMENT-ONLY: uniform LM continuation for fixed LiDAR rotation sigma.

Starts from each of the four existing saved Stage-6 fixed-FGO initial-LM
solutions in results/isj_dev_lidar_rot_sigma_v1. Reconstructs the EXACT
original graph by calling Stage-6's existing run_case, intercepting only
the GTSAM LM optimizer in memory to replace its start pose with each
SAVED case pose, and performing uniform convergence continuation.

Patch only: Stage6.OUT (fresh output directory), Stage6.LIDAR_R0
(frozen per-case value) and gtsam.LevenbergMarquardtOptimizer
(factory performing uniform continuation). Does NOT change the factor
graph code, measurements, graph noise except the same per-case LiDAR
rotation sigma, initial saved poses, prediction checkpoints, or prior
results. Runtime constructs the original initial trajectory but discards
it before continuation; actual LM initialization is the saved solution.

Outer convergence criterion: BOTH relative objective decrease <=1e-7
and position change RMSE <=1e-3 m, with max four additional 100-LM-
iteration passes, same for all cases. All stopping uses graph objective
and pose shift, NOT reference ATE. The saved 100-iteration solutions may
be quite close to stationary; if not, report NOT_STABLE, do not claim
method ranking. A near-stationary local LM solution is NOT a globally
optimal solution or independent test.

All outputs isolated in results/isj_dev_lidar_rot_sigma_convergence_v1.
Partial or mismatched prior outputs trigger hard error; no overwrites.
Original 0027 OXTS/GPS-derived positions are only used for post-hoc
ATE/RPE evaluation, NOT optimization or parameter selection.
0016 and prospective 0020 are never accessed.
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
from experiments.isj_supplement import run_dev_lidar_rot_sigma_ablation as pilot

SRC=pilot.OUT
OUT=ROOT/"results"/"isj_dev_lidar_rot_sigma_convergence_v1"
PROTOCOL="dev0027_uniform_convergence_lidar_rot_sigma_v1"
OBJECTIVE_RTOL=1e-7
POSE_SHIFT_RMSE_TOL_M=1e-3
MAX_EXTRA_PASSES=4
N=pilot.N_EXPECTED
K_RPE=(1,10,20)


def desired_spec(name,sigma):
    return {
        "protocol":PROTOCOL,
        "source_protocol":pilot.PROTOCOL,
        "source_case":name,
        "source_sigma_rad":float(sigma),
        "source_root":str(SRC),
        "destination_root":str(OUT),
        "max_extra_passes":MAX_EXTRA_PASSES,
        "objective_rtol":OBJECTIVE_RTOL,
        "pose_shift_rmse_tol_m":POSE_SHIFT_RMSE_TOL_M,
        "optimizer":"original Stage-6 LM 100 iterations each continuation",
        "reference":"0027 original OXTS/GPS local position, evaluation only",
    }


def get_source(name,sigma):
    """Verify source pose and provenance exactly before any writing."""
    source=SRC/name
    declared=pilot.spec_for(name,sigma)
    metadata=pilot.check_cache(source,declared,N)
    if metadata is None:
        raise FileNotFoundError(
            f"Missing 0027 sensitivity case {source}; "
            "run --stage fgo for original four frozen cases first"
        )
    poses=np.load(source/"poses.npy",allow_pickle=False)
    # The source metric was computed using this reference and these poses.
    measured=s6.evaluate(poses[:,:3,3])
    expected=float(metadata["metrics"]["ATE3D"])
    if abs(measured["ATE3D"]-expected)>1e-4:
        raise ValueError(
            f"{name}: saved source poses and source ATE disagree: "
            f"{measured['ATE3D']} vs {expected}"
        )
    return poses,metadata


def get_cached_result(folder:Path,expected:dict):
    if not folder.exists():
        return None
    poses=folder/"poses.npy"
    meta=folder/"isj_rot_sigma_convergence_manifest.json"
    if not poses.is_file() or not meta.is_file():
        raise RuntimeError(
            f"Partial pre-existing convergence output: {folder}. "
            "Refusing to overwrite; inspect manually."
        )
    value=np.load(poses,allow_pickle=False)
    if value.shape!=(N,4,4) or not np.isfinite(value).all():
        raise ValueError(f"Invalid cached convergence poses: {folder}")
    obj=json.loads(meta.read_text(encoding="utf-8"))
    if obj.get("spec")!=expected:
        raise ValueError(f"Conflicting convergence cache: {folder}")
    final=obj.get("final_metrics",{})
    if not math.isfinite(float(final.get("ATE3D",float("nan")))):
        raise ValueError(f"Invalid convergence metrics: {folder}")
    return obj


def poses_from_values(values,n):
    return np.stack(
        [s6._pose3_to_matrix(values.atPose3(i)) for i in range(n)],
        axis=0,
    )


def values_from_poses(gtsam,poses):
    v=gtsam.Values()
    for i,M in enumerate(poses):
        v.insert(i,s6._pose3_from_matrix(gtsam,M))
    return v


def relative_error_at_horizons(poses):
    estimated=np.asarray(poses,float)[:,:3,3]
    original=np.loadtxt(s6.GT,dtype=float)
    if original.ndim!=2 or original.shape[0]!=len(estimated):
        raise ValueError("Original 0027 OXTS/GPS eval length mismatch")
    original=original[:,-3:]
    if original.shape!=(N,3) or not np.isfinite(original).all():
        raise ValueError("Bad original OXTS/GPS evaluation trajectory")
    scores={}
    for k in K_RPE:
        e=(estimated[k:]-estimated[:-k])-(original[k:]-original[:-k])
        scores[f"RPE_translation_RMSE_k{k}_m"]=float(np.sqrt(np.mean(
            np.einsum("ij,ij->i",e,e)
        )))
    return scores


def continuation(
    gtsam,graph,source_poses,
    *,solver_class,max_extra_passes=MAX_EXTRA_PASSES,
    objective_rtol=OBJECTIVE_RTOL,shift_tol=POSE_SHIFT_RMSE_TOL_M,
):
    """Pure solver continuation, never uses evaluation/reference coordinates."""
    if source_poses.shape!=(N,4,4) or not np.isfinite(source_poses).all():
        raise ValueError("Source pose shape mismatch")
    current=values_from_poses(gtsam,source_poses)
    previous_pose=source_poses
    previous_cost=float(graph.error(current))
    if not math.isfinite(previous_cost):
        raise ValueError("Source graph objective not finite")
    history=[{
        "extra_pass":0,
        "objective":previous_cost,
        "relative_objective_drop":None,
        "position_shift_rmse_m":None,
    }]
    stationary=False
    warning=None
    for i in range(1,max_extra_passes+1):
        p=gtsam.LevenbergMarquardtParams()
        p.setMaxIterations(100)
        p.setRelativeErrorTol(1e-7)
        proposal=solver_class(graph,current,p).optimize()
        next_cost=float(graph.error(proposal))
        if not math.isfinite(next_cost):
            raise ValueError("Nonfinite continuation graph objective")
        if next_cost>previous_cost+max(1e-7,abs(previous_cost)*1e-10):
            warning=f"objective increased: {previous_cost} -> {next_cost}"
            break
        poses=poses_from_values(proposal,N)
        displacement=poses[:,:3,3]-previous_pose[:,:3,3]
        shift=float(np.sqrt(np.mean(np.einsum(
            "ij,ij->i",displacement,displacement
        ))))
        rel_drop=float(max(0.,(previous_cost-next_cost)/
                           max(abs(previous_cost),1.)))
        history.append({
            "extra_pass":i,"objective":next_cost,
            "relative_objective_drop":rel_drop,
            "position_shift_rmse_m":shift,
        })
        print(f"  pass {i}: graph_objective={next_cost:.9f} "
              f"relative_drop={rel_drop:.3e} "
              f"pose_shift_rmse_m={shift:.6f}",flush=True)
        current=proposal
        previous_pose=poses
        previous_cost=next_cost
        if rel_drop<=objective_rtol and shift<=shift_tol:
            stationary=True
            break
    return current,history,stationary,warning


class _InterceptStage6Optimizer:
    """Delegates saved-solution continuation while reusing Stage-6's graph."""

    def __init__(self,graph,unused_stage6_initial,unused_stage6_params,
                 *,gtsam,real_solver,source_poses,records):
        del unused_stage6_initial,unused_stage6_params
        if records:
            raise RuntimeError("Stage-6 attempted to create >1 optimizer")
        self.graph=graph
        self.gtsam=gtsam
        self.real_solver=real_solver
        self.source_poses=source_poses
        self.records=records

    def optimize(self):
        if int(self.graph.size())!=18092:
            raise ValueError(
                f"Unexpected original Stage-6 graph size={self.graph.size()}, "
                "expected 18092; refuse mismatched graph continuation"
            )
        values,history,stationary,warning=continuation(
            self.gtsam,self.graph,self.source_poses,
            solver_class=self.real_solver,
        )
        self.records.append({
            "history":history,
            "stationary_under_rule":stationary,
            "warning":warning,
            "factor_count":int(self.graph.size()),
        })
        return values


def run_one(name,sigma,poses,source_metadata):
    expected=desired_spec(name,sigma)
    cached=get_cached_result(OUT/name,expected)
    if cached is not None:
        print(f"{name}: CACHED ATE3D={cached['final_metrics']['ATE3D']:.6f}m")
        return cached

    import gtsam

    original_solver=gtsam.LevenbergMarquardtOptimizer
    records=[]
    def optimizer_factory(graph,initial,params):
        return _InterceptStage6Optimizer(
            graph,initial,params,gtsam=gtsam,real_solver=original_solver,
            source_poses=poses,records=records,
        )

    measurements=s6.load_measurements()
    if len(measurements.gps_local)!=N:
        raise ValueError("0027 frame count mismatch")
    # Reuse original Stage-6 graph literally. Patch only the same fixed
    # lidar rotation sigma, the output path and optimizer constructor.
    # All three patches are scoped and automatically restored.
    with (
        patch.object(s6,"OUT",OUT),
        patch.object(s6,"LIDAR_R0",float(sigma)),
        patch.object(gtsam,"LevenbergMarquardtOptimizer",optimizer_factory),
    ):
        result_metrics=s6.run_case(
            name=name,measurements=measurements,
            gps_mode="fixed",imu_mode="fixed",
            camera_mode="fixed",lidar_mode="fixed",
        )
    if len(records)!=1:
        raise RuntimeError("Stage-6 optimizer interception failed")
    destination=OUT/name
    output_poses=np.load(destination/"poses.npy",allow_pickle=False)
    expected_pose=output_poses
    if expected_pose.shape!=(N,4,4) or not np.isfinite(expected_pose).all():
        raise ValueError("Convergence output poses invalid")

    evaluation={
        **result_metrics,
        **relative_error_at_horizons(output_poses),
    }
    payload={
        "spec":expected,
        "source_metrics":source_metadata["metrics"],
        "final_metrics":evaluation,
        "source_reference_note":"original 0027 OXTS/GPS, not independent GT",
        "continuation":records[0],
        "notes":[
            "No LiDAR rotation sigma changed during continuation.",
            "No 0020 or 0016 drive accessed.",
            "All four cases use identical max extra passes/tolerances.",
            "Graph objective across differently weighted cases not comparable.",
            "Numerical convergence to a local solution is not external validity.",
        ],
    }
    (destination/"isj_rot_sigma_convergence_manifest.json").write_text(
        json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("audit","converge"),required=True)
    args=parser.parse_args(argv)

    print("0027 LI DAR ROTATION SIGMA: UNIFORM CONVERGENCE CONTINUATION")
    print(f"Source folder: {SRC}\nOutput folder: {OUT}")
    print("No 0016/0020, no retraining, no alteration of original Stage-6")
    cases=[]
    for name,sigma in pilot.SPECS:
        poses,meta=get_source(name,sigma)
        cache=get_cached_result(OUT/name,desired_spec(name,sigma))
        state=("CACHED" if cache is not None else "PENDING")
        print(f"  {name:<28s} source_ATE3D={meta['metrics']['ATE3D']:.6f}m "
              f"sigma={sigma:.6f}rad, convergence={state}")
        cases.append((name,sigma,poses,meta,cache))
    if args.stage=="audit":
        print("AUDIT ONLY: no solver run, no file writes")
        return

    results=[]
    for name,sigma,poses,meta,cached in cases:
        print(f"\nUNIFORM CONVERGENCE: {name}",flush=True)
        data=cached if cached is not None else run_one(name,sigma,poses,meta)
        status=data["continuation"]["stationary_under_rule"]
        metrics=data["final_metrics"]
        results.append({
            "name":name,"sigma_rad":sigma,
            "source_ATE3D":meta["metrics"]["ATE3D"],
            "final_ATE3D":metrics["ATE3D"],
            "final_ATE2D":metrics["ATE2D"],
            "final_ZRMSE":metrics["ZRMSE"],
            "final_RPE_k1":metrics["RPE_translation_RMSE_k1_m"],
            "final_RPE_k10":metrics["RPE_translation_RMSE_k10_m"],
            "stationary_under_rule":status,
        })
        print(f"  ATE3D={metrics['ATE3D']:.6f} "
              f"ATE2D={metrics['ATE2D']:.6f} "
              f"ZRMSE={metrics['ZRMSE']:.6f} "
              f"stationary={status}",flush=True)
    print("\nDEVELOPMENT-ONLY UNIFORM-CONTINUATION RESULTS")
    for r in results:
        print(r)
    if not all(r["stationary_under_rule"] for r in results):
        print("STOP: One or more cases NOT_STABLE; ranking is inconclusive")
    else:
        print("All cases STATIONARY under identical numeric rule; "
              "still only 0027, not an external ISJ evaluation.")


if __name__=="__main__":
    main()
