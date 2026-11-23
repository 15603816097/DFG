from __future__ import annotations

"""READ-ONLY development 0027 V3-label / V5-GRU / V6-Mamba provenance audit.

This is an inspection of existing trusted local checkpoints and
original development prediction artifacts. It performs NO network
training, FGO, inference, model deserialization on outside files,
checkpoint rewrites, or file writes. NEVER accesses 0016/0020.

Audits:
  - same-source V3 training archive and the V6 CLI default candidate;
  - exact V5/V6 saved checkpoint architecture, feature dimensions,
    training-window and normalization fingerprint vs each candidate;
  - source-time semantics of the V3 single future (+3) label versus
    the actual three-headed V5/V6 targets [t,t+3,t+2];
  - train->VAL and VAL->TEST leakage from future target lookahead;
  - target-aligned r/U array lengths and unpredicted placeholder tails.

A checkpoint matching a dataset's normalization DOES NOT prove the
data source because neither V5 nor V6 saves a cryptographic training
data digest / full invocation command. "No mismatch detected" is
distinct from "lineage proven."

Security: torch.load(weights_only=False) is required for locally
trusted original checkpoints containing NumPy scaler objects.
Do NOT direct this audit to untrusted files; pickle can execute code.
"""

import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.isj_supplement.audit_dev_uncertainty_risk_by_split import (
    split_ranges,
)

V3=ROOT/"results"/"predictive_factor_reliability_v3_same_source"/"factor_reliability_training_data.npz"
V6_DEFAULT=ROOT/"data"/"factor_reliability_training_data.npz"
V5_OUT=ROOT/"results"/"sensor_specific_reliability_v5_uncertainty"
V6_OUT=ROOT/"results"/"v6_mamba_uncertainty"
N_EXPECTED=4544
SENSORS=("gps","imu","camera")
HORIZON=3
WINDOW=64
LOOKAHEAD=3


def file_hash(path,chunk_size=1024*1024):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(chunk_size),b""):
            h.update(chunk)
    return h.hexdigest()


def read_candidate(path):
    path=Path(path)
    if not path.is_file():
        return None
    with np.load(path,allow_pickle=False) as npz:
        obj={}
        for s in SENSORS:
            for suffix in (
                "features",
                "current_factor_reliability",
                "future_factor_reliability",
            ):
                key=f"{s}_{suffix}"
                if key not in npz.files:
                    raise KeyError(f"{path}: missing {key}")
                arr=np.asarray(npz[key])
                if len(arr)!=N_EXPECTED or not np.isfinite(arr).all():
                    raise ValueError(f"{path}: invalid {key} shape/data")
                obj[key]=arr
    return obj


def source_timing_audit(n=N_EXPECTED):
    """Declarative code-semantics audit; no learned outputs or files read.

    V3 builder stores future[t]=current[t+HORIZON] (with padded
    last three frames). V5/V6 use that in output head index1.
    Other head index2 uses current[t+2], so head1/head2
    are nonchronological relative to claimed [t,t+1,t+2].
    """
    if n<WINDOW+HORIZON+10:
        raise ValueError("Insufficient development sequence length")
    timeline={}
    for model in ("GRU_V5","MAMBA_V6"):
        splits=split_ranges(model,n)
        row={}
        for label,(a,b) in splits.items():
            # For each output head h, which target frame is used?
            offsets=(0,LOOKAHEAD,2)
            crosses={
                "h1_stored_future_t_plus_3":
                    [int(t) for t in range(a,b) if t+LOOKAHEAD>=b],
                "h2_shifted_current_t_plus_2":
                    [int(t) for t in range(a,b) if t+2>=b],
            }
            # Only TRAIN and VAL cross internal split boundaries;
            # TEST crossing last valid target would be a padded-label
            # problem, not "leakage into a prospective test drive".
            row[label]={
                "range":(a,b),
                "head_target_offsets":offsets,
                "indices_with_label_beyond_split":crosses,
            }
        timeline[model]=row
    return timeline


def v5_expected_norm(features):
    n=len(features)
    cut=int(.7*n)
    x=np.asarray(features[:cut],dtype=np.float32)
    return x.mean(axis=0,keepdims=True),np.maximum(
        x.std(axis=0,keepdims=True),1e-6
    )


def v6_expected_norm(features):
    n=len(features)
    train_end=int((n-HORIZON+1)*.7)
    x=np.asarray(features[:train_end],dtype=np.float32)
    median=np.median(x,axis=0).astype(np.float32)
    q25=np.percentile(x,25,axis=0)
    q75=np.percentile(x,75,axis=0)
    scale=(q75-q25).astype(np.float32)
    scale[scale<1e-6]=1.
    return median,scale


def scaler_match(ckpt,features,kind):
    """Match *saved* normalization to a candidate; not proof of training lineage."""
    if kind=="GRU_V5":
        expected=v5_expected_norm(features)
        keys=("mean","std")
    elif kind=="MAMBA_V6":
        expected=v6_expected_norm(features)
        keys=("mean","scale")
    else:
        raise ValueError(kind)
    if any(key not in ckpt for key in keys):
        raise KeyError(f"Checkpoint missing normalization keys {keys}")
    a=np.asarray(ckpt[keys[0]],dtype=np.float32).reshape(-1)
    b=np.asarray(ckpt[keys[1]],dtype=np.float32).reshape(-1)
    left=np.asarray(expected[0],dtype=np.float32).reshape(-1)
    right=np.asarray(expected[1],dtype=np.float32).reshape(-1)
    if a.shape!=left.shape or b.shape!=right.shape:
        return False,float("inf"),float("inf")
    da=float(np.max(np.abs(a-left)))
    db=float(np.max(np.abs(b-right)))
    okay=(np.allclose(a,left,rtol=2e-5,atol=1e-4)
          and np.allclose(b,right,rtol=2e-5,atol=1e-4))
    return bool(okay),da,db


def prediction_placeholder_report(r,u,model):
    r=np.asarray(r,float).reshape(-1)
    u=np.asarray(u,float).reshape(-1)
    if r.shape!=(N_EXPECTED,) or u.shape!=(N_EXPECTED,):
        raise ValueError("0027 prediction length mismatch")
    if (not np.isfinite(r).all() or not np.isfinite(u).all()
        or np.any((r<0)|(r>1)) or np.any((u<0)|(u>1))):
        raise ValueError("Invalid r/U values")
    expected_r=.5 if model=="GRU_V5" else 1.
    first=(np.allclose(r[:WINDOW-1],expected_r)
           and np.allclose(u[:WINDOW-1],1.))
    last=(np.allclose(r[-2:],expected_r)
          and np.allclose(u[-2:],1.))
    return bool(first),bool(last)


def read_trusted_checkpoint(path):
    """ONLY original local, trusted research checkpoints (pickle warning)."""
    path=Path(path)
    if not path.is_file():
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to inspect trusted model checkpoints") from exc
    # These are user-controlled original project checkpoints. Never accept
    # third-party checkpoint paths for weights_only=False.
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if not isinstance(ck,dict):
        raise ValueError(f"Expected dictionary checkpoint: {path}")
    return ck


def inspect_candidate(label,path,candidate,v3):
    if candidate is None:
        print(f"DATA {label}: MISSING {path}")
        return
    print(f"DATA {label}: path={path}, SHA256={file_hash(path)}")
    if label=="V6_DEFAULT":
        matches={
            k:np.array_equal(v3[k],candidate[k])
            for k in v3
        }
        print(f"  V6 default archive arrays EXACT match V3: "
              f"{sum(matches.values())}/{len(matches)}")
        for k in matches:
            if not matches[k]:
                print(f"    DIFFERENT ARRAY: {k}")
        print("  Archive equality does NOT establish which --data argument "
              "was used to train the frozen checkpoint.")


def inspect_checkpoints(v3,candidates):
    for model,out,checkpoint_name in (
        ("GRU_V5",V5_OUT,"model.pt"),
        ("MAMBA_V6",V6_OUT,"best.pt"),
    ):
        print(f"\n========== {model} LOCAL CHECKPOINTS ==========")
        for sensor in SENSORS:
            path=out/sensor/checkpoint_name
            ck=read_trusted_checkpoint(path)
            if ck is None:
                print(f"  {sensor}: CHECKPOINT MISSING {path}; provenance UNKNOWN")
                continue
            x=v3[f"{sensor}_features"]
            backend=ck.get("backend","NOT_STORED")
            ck_window=ck.get("window","UNKNOWN")
            ck_horizon=ck.get("horizon","UNKNOWN")
            input_dim=ck.get("input_dim","UNKNOWN")
            print(f"  {sensor}: checkpoint={path} SHA256={file_hash(path)}")
            print(f"    backend={backend!r} window={ck_window} "
                  f"horizon={ck_horizon} input_dim={input_dim}")
            if ck_window!=WINDOW or ck_horizon!=HORIZON:
                print("    WARNING: checkpoint window/horizon differs "
                      "from assumed V3/V5/V6 code defaults")
            if input_dim!=x.shape[1]:
                print(f"    WARNING: checkpoint input_dim={input_dim}, "
                      f"V3 feature dim={x.shape[1]}")
            if model=="GRU_V5" and backend!="gru":
                print("    WARNING: V5 architecture NOT proven GRU; "
                      "its code dynamically selects Mamba when installed!")
            state=ck.get("state_dict") if model=="GRU_V5" else ck.get("model")
            if not isinstance(state,dict):
                raise ValueError(f"{sensor}/{model}: missing state_dict/model")
            weights=list(state)
            if model=="GRU_V5":
                print(f"    GRU state key present={any('gru.' in k for k in weights)} "
                      f"(check actual frozen architecture before GRU claims)")
            for data_name,data in candidates.items():
                if data is None:
                    continue
                matches,dm,ds=scaler_match(
                    ck,data[f"{sensor}_features"],model,
                )
                print(f"    scaler vs {data_name}: "
                      f"{'MATCH' if matches else 'MISMATCH'} "
                      f"max_abs_mean_delta={dm:.6g} "
                      f"max_abs_scale_delta={ds:.6g}")
            pr=out/f"{sensor}_reliability_target_aligned.txt"
            pu=out/f"{sensor}_uncertainty_target_aligned.txt"
            if not pr.is_file() or not pu.is_file():
                print(f"    PREDICTION FILES MISSING: {pr}, {pu}")
                continue
            r=np.loadtxt(pr,dtype=float)
            u=np.loadtxt(pu,dtype=float)
            first,last=prediction_placeholder_report(r,u,model)
            print(f"    prediction aligned n={len(r)} "
                  f"first{WINDOW-1} placeholder={'MATCH' if first else 'MISMATCH'} "
                  f"final2 placeholder={'MATCH' if last else 'MISMATCH'}")


def main():
    print("="*105)
    print("0027 READ-ONLY MODEL / LABEL LINEAGE AND TIME-ALIGNMENT AUDIT")
    print("NO retraining, FGO, inference or writes; 0016/0020 NOT accessed.")
    print("WARNING: torch.load(weights_only=False) reads ONLY trusted local "
          "model.pt / best.pt checkpoints: never inspect untrusted .pt files.")
    print("="*105)
    v3=read_candidate(V3)
    if v3 is None:
        raise FileNotFoundError(f"Original V3 same-source source data missing: {V3}")
    v6_default=read_candidate(V6_DEFAULT)
    inspect_candidate("V3_ORIGINAL",V3,v3,v3)
    inspect_candidate("V6_DEFAULT",V6_DEFAULT,v6_default,v3)

    print("\nSTATIC CODE-SEMANTICS VERDICT:")
    print("  V3 single future label[t] = current reliability[t+3] "
          "(last 3 padded).")
    print("  BOTH V5 and V6 three heads are [t,t+3,t+2], "
          "NOT intended [t,t+1,t+2].")
    print("  This is a training-target horizon/ordering mismatch "
          "for forecast heads, NOT a proof the h0 current head is wrong.")
    for model,splits in source_timing_audit().items():
        for part,info in splits.items():
            if part=="TEST":
                continue
            cross=info["indices_with_label_beyond_split"]
            print(f"  {model} {part} {info['range']}: "
                  f"head1 from out-of-split t indices="
                  f"{cross['h1_stored_future_t_plus_3']}; "
                  f"head2 out-of-split="
                  f"{cross['h2_shifted_current_t_plus_2']}")
    print("  IMPORTANT: V3 LiDAR/Camera/IMU proxy labels use "
          "full-trajectory fixed-FGO optimized reference poses; "
          "they are posthoc and may contain future time-series "
          "information from other measurements.")
    print("  V3 GPS label uses |degraded GPS - clean GPS|, "
          "requiring oracle supervision, not an online observable.")
    inspect_checkpoints(
        v3,{"V3_ORIGINAL":v3,"V6_DEFAULT":v6_default},
    )
    print("\nVERDICT: matching scaler/array data is compatible with a "
          "source but does NOT prove training used that exact source; "
          "checkpoint has NO immutable training data digest.")
    print("STOP: do not claim validated 3-step forecast or clean "
          "split-provenance until corrected targets and checkpoint "
          "retraining/frozen protocol are addressed.")


if __name__=="__main__":
    main()
