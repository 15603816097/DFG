import argparse, os, sys, numpy as np, gtsam
from gtsam import symbol,Values,NonlinearFactorGraph,LevenbergMarquardtOptimizer,Pose3,Point3,Rot3,PriorFactorPose3,PriorFactorVector,GPSFactor,ImuFactor,BetweenFactorConstantBias,PreintegratedImuMeasurements,PreintegrationParams,noiseModel

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from src.loader.imu_loader import IMULoader

DATASET=os.path.join(ROOT,"dataset","kitti","2011_10_03","2011_10_03_drive_0027_sync")
GPS_PATH=os.path.join(ROOT,"results","progressive_gps_degradation","gps_corrupted.txt")
PRIOR_PATH=os.path.join(ROOT,"results","horizon_sensitivity","h3","predictive_prior_target_aligned.txt")
DT=0.1
SIGMA_MIN=3.0

def load_imu():
    l=IMULoader(DATASET); a=[]; g=[]
    for i in range(len(l)):
        d=l[i]; a.append(d["acceleration"]); g.append(d["angular_velocity"])
    return np.asarray(a,float),np.asarray(g,float)

def params():
    p=PreintegrationParams.MakeSharedU(9.81)
    p.setAccelerometerCovariance(np.eye(3)*0.1)
    p.setGyroscopeCovariance(np.eye(3)*0.01)
    p.setIntegrationCovariance(np.eye(3)*0.001)
    return p

def map_sigma(r,sigma_max,gamma):
    r=np.clip(np.asarray(r,float),0,1)
    return np.clip(SIGMA_MIN+(1-r)**gamma*(sigma_max-SIGMA_MIN),SIGMA_MIN,sigma_max)

def tag(x): return f"{float(x):.2f}".replace(".","p")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gamma",type=float,required=True)
    ap.add_argument("--sigma-max",type=float,required=True)
    ap.add_argument("--stage",choices=["gamma","sigma_max"],required=True)
    args=ap.parse_args()
    if args.gamma<=0: raise ValueError("gamma must be > 0")
    if args.sigma_max<=SIGMA_MIN: raise ValueError("sigma_max must be > sigma_min")
    if not os.path.exists(PRIOR_PATH): raise FileNotFoundError(PRIOR_PATH)
    gps=np.loadtxt(GPS_PATH,float); prior=np.loadtxt(PRIOR_PATH,float).reshape(-1); acc,gyro=load_imu()
    n=min(len(gps),len(prior),len(acc),len(gyro))
    gps,prior,acc,gyro=gps[:n],np.clip(prior[:n],0,1),acc[:n],gyro[:n]
    sigma=map_sigma(prior,args.sigma_max,args.gamma)
    out=os.path.join(ROOT,"results","covariance_mapping_sensitivity",args.stage,f"gamma_{tag(args.gamma)}_sigmaMax_{tag(args.sigma_max)}")
    os.makedirs(out,exist_ok=True)
    print("="*80);print("COVARIANCE MAPPING SENSITIVITY")
    print("Stage:",args.stage,"| H=3 | alpha=1.0 | feedback=0.0")
    print("sigma_min:",SIGMA_MIN,"sigma_max:",args.sigma_max,"gamma:",args.gamma)
    print("Sigma min/max/mean:",sigma.min(),sigma.max(),sigma.mean())

    graph=NonlinearFactorGraph(); initial=Values(); p=params(); b0=gtsam.imuBias.ConstantBias()
    pn=noiseModel.Isotropic.Sigma(6,0.1); vn=noiseModel.Isotropic.Sigma(3,1.0); bn=noiseModel.Isotropic.Sigma(6,1e-3)
    p0=Pose3(Rot3(),Point3(*gps[0]))
    graph.add(PriorFactorPose3(symbol("x",0),p0,pn))
    graph.add(PriorFactorVector(symbol("v",0),np.zeros(3),vn))
    initial.insert(symbol("x",0),p0); initial.insert(symbol("v",0),np.zeros(3)); initial.insert(symbol("b",0),b0)

    def robust_noise(s):
        base=noiseModel.Isotropic.Sigma(3,float(s))
        return noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(1.345),base)

    for i in range(n-1):
        pim=PreintegratedImuMeasurements(p,b0); pim.integrateMeasurement(acc[i],gyro[i],DT)
        graph.add(ImuFactor(symbol("x",i),symbol("v",i),symbol("x",i+1),symbol("v",i+1),symbol("b",i),pim))
        graph.add(BetweenFactorConstantBias(symbol("b",i),symbol("b",i+1),gtsam.imuBias.ConstantBias(),bn))
        graph.add(GPSFactor(symbol("x",i),Point3(*gps[i]),robust_noise(sigma[i])))
        initial.insert(symbol("x",i+1),Pose3(Rot3(),Point3(*gps[i+1])))
        initial.insert(symbol("v",i+1),np.zeros(3)); initial.insert(symbol("b",i+1),b0)
        if i%500==0: print("Add factors:",i)

    graph.add(GPSFactor(symbol("x",n-1),Point3(*gps[-1]),robust_noise(sigma[-1])))
    print("Graph size:",graph.size()); print("Optimizing...")
    result=LevenbergMarquardtOptimizer(graph,initial).optimize()
    traj=np.array([[*result.atPose3(symbol("x",i)).translation()] for i in range(n)],float)
    np.savetxt(os.path.join(out,"trajectory.txt"),traj,fmt="%.8f")
    np.savetxt(os.path.join(out,"predictive_reliability.txt"),prior,fmt="%.8f")
    np.savetxt(os.path.join(out,"sigma.txt"),sigma,fmt="%.8f")
    with open(os.path.join(out,"config.txt"),"w") as f:
        f.write(f"horizon=3\nprediction_weight=1.0\nfeedback_weight=0.0\nsigma_min={SIGMA_MIN}\nsigma_max={args.sigma_max}\ngamma={args.gamma}\n")
    print("Trajectory:",traj.shape); print("Saved:",out)

if __name__=="__main__": main()
