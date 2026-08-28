from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SENSORS=("gps","imu","lidar","camera")

def _smoothstep(x):
    x=np.clip(np.asarray(x,float),0,1)
    return x*x*(3-2*x)

def _episode(n,start,rise_end,hold_end,recovery_end,peak=1.0):
    f=np.arange(n); c=np.zeros(n,float)
    m=(f>=start)&(f<rise_end)
    if np.any(m):
        x=(f[m]-start)/max(rise_end-start,1); c[m]=peak*_smoothstep(x)
    m=(f>=rise_end)&(f<hold_end); c[m]=peak
    m=(f>=hold_end)&(f<recovery_end)
    if np.any(m):
        x=(f[m]-hold_end)/max(recovery_end-hold_end,1); c[m]=peak*(1-_smoothstep(x))
    return np.clip(c,0,1)

def _bursts(n,items):
    c=np.zeros(n,float)
    for start,length,peak in items:
        if start<n:c[start:min(start+length,n)]=np.maximum(c[start:min(start+length,n)],peak)
    return np.clip(c,0,1)

def create_multisensor_degradation_plan(n_frames,seed=20260826):
    n=int(n_frames)
    if n<=0:raise ValueError("n_frames must be positive")
    gps=np.maximum.reduce([_episode(n,650,950,1350,1650,.70),_episode(n,1900,2200,2850,3200,1.0),_bursts(n,[(3650,35,1.0),(4050,25,.9)])])
    imu=np.maximum.reduce([_episode(n,1100,1450,1950,2250,.75),_episode(n,2750,3050,3550,3850,1.0),_bursts(n,[(2350,18,1.0),(4000,12,1.0)])])
    lidar=np.maximum.reduce([_episode(n,450,750,1200,1500,.65),_episode(n,2300,2600,3150,3500,1.0),_bursts(n,[(1800,50,.9),(3900,45,1.0)])])
    camera=np.maximum.reduce([_episode(n,850,1150,1700,2050,.80),_episode(n,2500,2850,3350,3700,1.0),_bursts(n,[(2050,40,.9),(4150,35,1.0)])])
    mode={}
    mode["gps"]=(gps>0).astype(np.int64); mode["gps"][_bursts(n,[(3650,35,1),(4050,25,1)])>0]=2
    mode["imu"]=(imu>0).astype(np.int64); mode["imu"][_bursts(n,[(2350,18,1),(4000,12,1)])>0]=2
    mode["lidar"]=(lidar>0).astype(np.int64); mode["lidar"][_bursts(n,[(1800,50,1),(3900,45,1)])>0]=2
    cm=np.zeros(n,np.int64); f=np.arange(n); active=camera>0
    cm[active&(f<2050)]=1;cm[active&(f>=2050)&(f<3000)]=2;cm[active&(f>=3000)]=3
    mode["camera"]=cm
    return {"seed":int(seed),"n_frames":n,"severity":{"gps":gps.astype(np.float32),"imu":imu.astype(np.float32),"lidar":lidar.astype(np.float32),"camera":camera.astype(np.float32)},"mode":mode}

def save_plan(plan,output_dir):
    d=Path(output_dir);d.mkdir(parents=True,exist_ok=True)
    arrays={}
    for s in SENSORS:
        arrays[f"{s}_severity"]=np.asarray(plan["severity"][s],np.float32)
        arrays[f"{s}_mode"]=np.asarray(plan["mode"][s],np.int64)
    np.savez(d/"degradation_plan.npz",**arrays)
    meta={"seed":int(plan["seed"]),"n_frames":int(plan["n_frames"]),"sensors":list(SENSORS)}
    (d/"degradation_plan.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")

def load_plan(plan_dir):
    d=Path(plan_dir);x=np.load(d/"degradation_plan.npz")
    sev={s:np.asarray(x[f"{s}_severity"],np.float32) for s in SENSORS}
    mode={s:np.asarray(x[f"{s}_mode"],np.int64) for s in SENSORS}
    return {"seed":20260826,"n_frames":len(sev["gps"]),"severity":sev,"mode":mode}
