import numpy as np

def reliability_from_error(error, scale=8.0):
    e=np.asarray(error,dtype=float)
    return np.clip(np.exp(-0.5*(e/max(float(scale),1e-6))**2),0.0,1.0)

def future_window_min(r,horizon=5):
    r=np.asarray(r,dtype=float).reshape(-1); out=np.empty_like(r); n=len(r)
    for i in range(n):
        a=min(i+1,n-1); b=min(i+horizon+1,n)
        out[i]=r[-1] if a>=b else np.min(r[a:b])
    return out
