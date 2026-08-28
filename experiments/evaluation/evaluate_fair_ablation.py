import os,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT=os.path.join(ROOT,"results","ground_truth","trajectory.txt")
OUT=os.path.join(ROOT,"results","fair_ablation_evaluation.txt")
METHODS=[
("Fixed Covariance FG","fair_progressive_fixed_fg"),
("Current Reliability FG","fair_current_reliability_fg"),
("Predictive Only FG","fair_predictive_only_fg"),
("Predictive + Feedback FG","fair_predictive_feedback_fg"),
("Oracle Reliability FG","fair_oracle_progressive_fg")]

def calc(gt,tr):
    n=min(len(gt),len(tr));e=tr[:n]-gt[:n];e3=np.linalg.norm(e,axis=1);e2=np.linalg.norm(e[:,:2],axis=1)
    return [np.sqrt(np.mean(e3**2)),np.sqrt(np.mean(e2**2)),np.mean(e3),np.max(e3),np.mean(e2),np.max(e2),np.sqrt(np.mean(e[:,2]**2))]

def main():
    gt=np.loadtxt(GT);rows=[]
    print("="*80);print("FAIR PROGRESSIVE GPS DEGRADATION ABLATION");print("="*80)
    for name,folder in METHODS:
        p=os.path.join(ROOT,"results",folder,"trajectory.txt")
        if not os.path.exists(p):raise FileNotFoundError(p+"\nRun: python experiments/run_fair_ablation.py")
        tr=np.loadtxt(p);m=calc(gt,tr);rows.append((name,*m))
        print("\n"+name);print("ATE_3D_RMSE: %.6f | ATE_2D_RMSE: %.6f | Mean3D: %.6f | Max3D: %.6f | Z_RMSE: %.6f"%(m[0],m[1],m[2],m[3],m[6]))
    fixed=rows[0][1];lines=["Method,ATE3D,ATE2D,Mean3D,Max3D,Mean2D,Max2D,ZRMSE,ImprovementVsFixed(%)\n"]
    print("\n"+"="*80);print("%-32s %10s %10s %12s"%("Method","ATE3D","ATE2D","vs Fixed"))
    for r in rows:
        imp=100*(fixed-r[1])/max(fixed,1e-12);print("%-32s %10.4f %10.4f %11.2f%%"%(r[0],r[1],r[2],imp))
        lines.append("%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.4f\n"%((*r,imp)))
    with open(OUT,"w") as f:f.writelines(lines)
    print("\nSaved:",OUT)
if __name__=="__main__":main()
