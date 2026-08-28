import argparse, os, numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_PATH=os.path.join(ROOT,"results","ground_truth","trajectory.txt")
FIXED_PATH=os.path.join(ROOT,"results","fair_progressive_fixed_fg","trajectory.txt")
BASE=os.path.join(ROOT,"results","covariance_mapping_sensitivity")

def tag(x): return f"{float(x):.2f}".replace(".","p")

def metric(gt,tr):
    n=min(len(gt),len(tr)); e=tr[:n]-gt[:n]; e3=np.linalg.norm(e,axis=1); e2=np.linalg.norm(e[:,:2],axis=1)
    return dict(ATE3D=float(np.sqrt(np.mean(e3**2))),ATE2D=float(np.sqrt(np.mean(e2**2))),Mean3D=float(np.mean(e3)),Max3D=float(np.max(e3)),ZRMSE=float(np.sqrt(np.mean(e[:,2]**2))))

def load_fixed(gt):
    if not os.path.exists(FIXED_PATH): raise FileNotFoundError(FIXED_PATH)
    return metric(gt,np.loadtxt(FIXED_PATH,float))["ATE3D"]

def stage_gamma(gt,fixed):
    rows=[]; sigma_max=20.0
    for gamma in [1,2,3,4,5]:
        folder=os.path.join(BASE,"gamma",f"gamma_{tag(gamma)}_sigmaMax_{tag(sigma_max)}")
        tp=os.path.join(folder,"trajectory.txt"); sp=os.path.join(folder,"sigma.txt")
        if not os.path.exists(tp): continue
        m=metric(gt,np.loadtxt(tp,float)); sm=float(np.loadtxt(sp,float).mean()); imp=100*(fixed-m["ATE3D"])/fixed
        rows.append((float(gamma),sigma_max,m,sm,imp))
    if not rows: raise RuntimeError("No gamma results")
    best=min(rows,key=lambda x:x[2]["ATE3D"])
    print("="*90); print("GAMMA SENSITIVITY")
    print(f'{"Gamma":>8}{"ATE3D":>12}{"ATE2D":>12}{"SigmaMean":>14}{"vs Fixed":>14}')
    for g,s,m,sm,imp in rows: print(f"{g:8.2f}{m['ATE3D']:12.6f}{m['ATE2D']:12.6f}{sm:14.6f}{imp:13.2f}%")
    os.makedirs(BASE,exist_ok=True)
    with open(os.path.join(BASE,"best_gamma.txt"),"w") as f: f.write(f"{best[0]:.8f}\n")
    with open(os.path.join(ROOT,"results","covariance_gamma_sensitivity.csv"),"w") as f:
        f.write("Gamma,SigmaMax,ATE3D,ATE2D,SigmaMean,ImprovementVsFixed_percent\n")
        for g,s,m,sm,imp in rows: f.write(f"{g},{s},{m['ATE3D']:.8f},{m['ATE2D']:.8f},{sm:.8f},{imp:.4f}\n")
    print("Best gamma:",best[0],"ATE3D:",best[2]["ATE3D"],"Improvement:",best[4],"%")

def stage_sigma(gt,fixed):
    bg=os.path.join(BASE,"best_gamma.txt")
    if not os.path.exists(bg): raise FileNotFoundError(bg)
    gamma=float(open(bg).read().strip()); rows=[]
    for sigma_max in [10,15,20,25,30]:
        folder=os.path.join(BASE,"sigma_max",f"gamma_{tag(gamma)}_sigmaMax_{tag(sigma_max)}")
        tp=os.path.join(folder,"trajectory.txt"); sp=os.path.join(folder,"sigma.txt")
        if not os.path.exists(tp): continue
        m=metric(gt,np.loadtxt(tp,float)); sm=float(np.loadtxt(sp,float).mean()); imp=100*(fixed-m["ATE3D"])/fixed
        rows.append((gamma,float(sigma_max),m,sm,imp))
    if not rows: raise RuntimeError("No sigma_max results")
    best=min(rows,key=lambda x:x[2]["ATE3D"])
    print("="*90); print("SIGMA_MAX SENSITIVITY | fixed gamma =",gamma)
    print(f'{"SigmaMax":>10}{"ATE3D":>12}{"ATE2D":>12}{"SigmaMean":>14}{"vs Fixed":>14}')
    for g,s,m,sm,imp in rows: print(f"{s:10.2f}{m['ATE3D']:12.6f}{m['ATE2D']:12.6f}{sm:14.6f}{imp:13.2f}%")
    with open(os.path.join(ROOT,"results","covariance_sigma_max_sensitivity.csv"),"w") as f:
        f.write("Gamma,SigmaMax,ATE3D,ATE2D,SigmaMean,ImprovementVsFixed_percent\n")
        for g,s,m,sm,imp in rows: f.write(f"{g},{s},{m['ATE3D']:.8f},{m['ATE2D']:.8f},{sm:.8f},{imp:.4f}\n")
    final=os.path.join(ROOT,"results","best_covariance_mapping.txt")
    with open(final,"w") as f:
        f.write("horizon=3\nprediction_weight=1.0\nfeedback_weight=0.0\nsigma_min=3.0\n")
        f.write(f"sigma_max={best[1]:.8f}\ngamma={best[0]:.8f}\nATE3D={best[2]['ATE3D']:.8f}\nImprovementVsFixed={best[4]:.4f}%\n")
    print("FINAL BEST: sigma_min=3.0 sigma_max=",best[1],"gamma=",best[0])
    print("ATE3D:",best[2]["ATE3D"],"Improvement:",best[4],"%")
    print("Saved:",final)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=["gamma","sigma_max"],required=True); args=ap.parse_args()
    gt=np.loadtxt(GT_PATH,float); fixed=load_fixed(gt)
    stage_gamma(gt,fixed) if args.stage=="gamma" else stage_sigma(gt,fixed)

if __name__=="__main__": main()
