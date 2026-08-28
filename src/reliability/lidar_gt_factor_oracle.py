from __future__ import annotations
import numpy as np

def _normal_threshold(x):
    x=np.asarray(x,float); m=np.median(x); mad=np.median(np.abs(x-m))
    return float(m+3.0*1.4826*mad)

def build_oracle(trans_error, rot_error_deg, clean_mask):
    te=np.asarray(trans_error,float); re=np.asarray(rot_error_deg,float); clean=np.asarray(clean_mask,bool)
    tn=_normal_threshold(te[clean]); rn=_normal_threshold(re[clean])
    ts=float(np.percentile(te,95)); rs=float(np.percentile(re,95))
    t=np.clip((te-tn)/max(ts-tn,.10),0,1)
    r=np.clip((re-rn)/max(rs-rn,.50),0,1)
    sev=np.maximum(t,r)
    rel=np.clip(1-sev,.01,1)
    return dict(lidar_reliability=rel, translation_severity=t, rotation_severity=r,
                joint_severity=sev, trans_normal=np.array([tn]), rot_normal=np.array([rn]),
                trans_severe=np.array([ts]), rot_severe=np.array([rs]))
