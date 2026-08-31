from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.factor_graph.run_stage6_v6_mamba_factor_graph_ablation as s6

OUT = ROOT / "results" / "stage6_5_encoder_combination_ablation"
REFS = {
    "G0_all_fixed": 2.4563402209770127,
    "G1_v5_gru_all_uncertainty": 2.3712673769706125,
    "G2_v6_mamba_all_uncertainty": 2.3492851163213637,
    "G3_hybrid_gps_mamba_imu_gru_camera_mamba": 2.049112566703376,
}

def build_cases():
    return [
        ("D0_gru_gru_gru","GRU","GRU","GRU"),
        ("D1_mamba_gru_gru","Mamba","GRU","GRU"),
        ("D2_gru_mamba_gru","GRU","Mamba","GRU"),
        ("D3_gru_gru_mamba","GRU","GRU","Mamba"),
        ("D4_mamba_mamba_gru","Mamba","Mamba","GRU"),
        ("D5_mamba_gru_mamba","Mamba","GRU","Mamba"),
        ("D6_gru_mamba_mamba","GRU","Mamba","Mamba"),
        ("D7_mamba_mamba_mamba","Mamba","Mamba","Mamba"),
    ]

def mode(enc):
    return {"GRU":"v5_uncertainty","Mamba":"v6_uncertainty"}[enc]

def gt_xyz():
    a=np.loadtxt(s6.GT,dtype=float)
    if a.shape[1]>3: a=a[:,-3:]
    return a[:,:3]

def detailed(traj):
    g=gt_xyz(); e=np.asarray(traj,float)
    if e.shape[1]>3: e=e[:,-3:]
    n=min(len(g),len(e)); d=e[:n,:3]-g[:n,:3]
    x,y,z=d.T; xy=np.linalg.norm(d[:,:2],axis=1); xyz=np.linalg.norm(d,axis=1)
    return {
        "X_RMSE":float(np.sqrt(np.mean(x*x))),
        "Y_RMSE":float(np.sqrt(np.mean(y*y))),
        "Z_RMSE":float(np.sqrt(np.mean(z*z))),
        "X_Bias":float(np.mean(x)),"Y_Bias":float(np.mean(y)),"Z_Bias":float(np.mean(z)),
        "XY_P95":float(np.percentile(xy,95)),"XYZ_P95":float(np.percentile(xyz,95)),
        "Z_MAE":float(np.mean(np.abs(z))),"Z_P95Abs":float(np.percentile(np.abs(z),95)),
    }

def sanity():
    print("="*100); print("STAGE-6 SANITY"); print("="*100)
    for name,ref in REFS.items():
        p=s6.OUT/name/"trajectory.txt"
        if not p.exists(): raise FileNotFoundError(p)
        v=detailed(np.loadtxt(p))["Z_RMSE"] # only for loading; ATE below
        m=s6.evaluate(np.loadtxt(p))
        delta=abs(m["ATE3D"]-ref)
        print(f"{name:50s} ATE3D={m['ATE3D']:.9f} ref={ref:.9f} delta={delta:.2e}")
        if delta>5e-4: raise RuntimeError(f"Stage6 reference mismatch: {name}")
    print("[sanity] PASS")

def pred_stats(n):
    rows=[]
    for sensor in ("gps","imu","camera"):
        for src,enc in (("v5","GRU"),("v6","Mamba")):
            r,u=s6.load_prediction(src,sensor,n)
            rows.append({"sensor":sensor,"encoder":enc,
                         "r_mean":float(r.mean()),"r_std":float(r.std()),
                         "u_mean":float(u.mean()),"u_std":float(u.std()),
                         "confidence_mean":float((1-u).mean())})
    p=OUT/"prediction_statistics.csv"
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return p

def effects(rows):
    d={(r["gps_encoder"]=="Mamba",r["imu_encoder"]=="Mamba",r["camera_encoder"]=="Mamba"):r for r in rows}
    out=[]
    for metric in ("ATE3D","ATE2D","ZRMSE"):
        for axis,label in enumerate(("GPS","IMU","Camera")):
            vals=[]
            for b in d:
                if not b[axis]:
                    q=list(b); q[axis]=True; q=tuple(q)
                    vals.append(d[q][metric]-d[b][metric])
            out.append({"metric":metric,"effect":label+"_Mamba_minus_GRU_main",
                        "delta":float(np.mean(vals)),"note":"negative_is_better"})
        for a,b,label in ((0,1,"GPSxIMU"),(0,2,"GPSxCamera"),(1,2,"IMUxCamera")):
            third=({0,1,2}-{a,b}).pop(); vals=[]
            for tv in (False,True):
                q=[False]*3; q[third]=tv; q00=tuple(q)
                q[a]=True; q10=tuple(q)
                q[a]=False; q[b]=True; q01=tuple(q)
                q[a]=True; q11=tuple(q)
                vals.append(d[q11][metric]-d[q10][metric]-d[q01][metric]+d[q00][metric])
            out.append({"metric":metric,"effect":label+"_interaction",
                        "delta":float(np.mean(vals)),"note":"large_abs_means_coupling"})
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sanity()
    meas=s6.load_measurements(); n=len(meas.gps_local)
    ps=pred_stats(n)
    rows=[]
    old=s6.OUT
    try:
        s6.OUT=OUT
        for name,g,i,c in build_cases():
            kw={"gps_mode":mode(g),"imu_mode":mode(i),"camera_mode":mode(c),"lidar_mode":"uncertainty"}
            met=s6.run_case(name,meas,**kw)
            traj=np.loadtxt(OUT/name/"trajectory.txt")
            rows.append({"case":name,"gps_encoder":g,"imu_encoder":i,"camera_encoder":c,
                         "lidar_policy":"frozen_stage6_uncertainty",**met,**detailed(traj)})
    finally:
        s6.OUT=old

    cp=OUT/"stage6_5_encoder_cube.csv"
    with cp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ef=effects(rows); ep=OUT/"stage6_5_encoder_effects.csv"
    with ep.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(ef[0])); w.writeheader(); w.writerows(ef)

    rank=sorted(rows,key=lambda r:r["ATE3D"]); lines=["Stage 6.5 Encoder Combination Analysis","="*90,""]
    for k,r in enumerate(rank,1):
        gain=(REFS["G0_all_fixed"]-r["ATE3D"])/REFS["G0_all_fixed"]*100
        lines.append(f"{k:02d}. {r['case']:26s} ATE3D={r['ATE3D']:.6f} ATE2D={r['ATE2D']:.6f} ZRMSE={r['ZRMSE']:.6f} vs_fixed={gain:+.2f}%")
    lines += ["","Encoder effects (Mamba-GRU; negative is beneficial for error metrics):"]
    lines += [f"{e['metric']:7s} {e['effect']:32s} {e['delta']:+.6f}" for e in ef]
    sp=OUT/"stage6_5_summary.txt"; sp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n"+"="*100+"\nFINAL STAGE-6.5 RANKING\n"+"="*100)
    print("\n".join(lines[3:3+len(rank)]))
    print("\nOutputs:",cp,ep,ps,sp,sep="\n")

if __name__=="__main__":
    main()
