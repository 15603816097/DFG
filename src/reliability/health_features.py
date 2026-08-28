import numpy as np

def _stats(x,w):
    m=np.zeros_like(x,dtype=float); s=np.zeros_like(x,dtype=float)
    for i in range(len(x)):
        q=x[max(0,i-w+1):i+1]; m[i]=q.mean(0); s[i]=q.std(0)
    return m,s

def build_health_features(gps,acc,gyro,dt=0.1,window=10):
    gps=np.asarray(gps,float); acc=np.asarray(acc,float); gyro=np.asarray(gyro,float)
    dg=np.zeros_like(gps); dg[1:]=gps[1:]-gps[:-1]
    vel=dg/dt; dv=np.zeros_like(vel); dv[1:]=vel[1:]-vel[:-1]; ga=dv/dt
    dm,ds=_stats(dg,window); _,astd=_stats(acc,window); _,gstd=_stats(gyro,window)
    scal=np.c_[np.linalg.norm(vel,axis=1),np.linalg.norm(dg,axis=1),np.linalg.norm(ga,axis=1)]
    f=np.c_[dg,vel,ga,dm,ds,scal,acc,gyro,astd,gstd]
    f[~np.isfinite(f)]=0
    return f.astype(np.float32)
