import numpy as np

def severity_to_reliability(severity, gamma=1.5):
    s=np.clip(np.asarray(severity,dtype=float),0.0,1.0)
    return np.clip((1.0-s)**float(gamma),0.0,1.0)

def future_window_min(reliability,horizon=3):
    r=np.asarray(reliability,dtype=float).reshape(-1); n=len(r); h=max(int(horizon),1); out=np.empty_like(r)
    for i in range(n):
        a=min(i+1,n-1); b=min(i+h+1,n)
        out[i]=r[-1] if a>=b else np.min(r[a:b])
    return out

def build_future_labels(severity_dict,horizon=3,gamma=1.5):
    current={};future={}
    for sensor,severity in severity_dict.items():
        current[sensor]=severity_to_reliability(severity,gamma)
        future[sensor]=future_window_min(current[sensor],horizon)
    return current,future
