import os,sys,numpy as np,gtsam
from gtsam import symbol,Values,NonlinearFactorGraph,LevenbergMarquardtOptimizer,Pose3,Point3,Rot3,PriorFactorPose3,PriorFactorVector,GPSFactor,ImuFactor,BetweenFactorConstantBias,PreintegratedImuMeasurements,PreintegrationParams,noiseModel
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from src.loader.imu_loader import IMULoader
from src.reliability.reliability_mapper import reliability_to_sigma
from src.reliability.feedback_corrector import residual_reliability,fuse_reliability
DATASET=os.path.join(ROOT,"dataset","kitti","2011_10_03","2011_10_03_drive_0027_sync")
GPS=os.path.join(ROOT,"results","progressive_gps_degradation","gps_corrupted.txt")
PRED=os.path.join(ROOT,"results","predictive_reliability_new","predicted_reliability.txt")
GT=os.path.join(ROOT,"results","ground_truth","trajectory.txt")
DT=.1

def load_imu():
    l=IMULoader(DATASET);a=[];g=[]
    for i in range(len(l)):
        d=l[i];a.append(d["acceleration"]);g.append(d["angular_velocity"])
    return np.asarray(a,float),np.asarray(g,float)

def load_data(pred=False,gt=False):
    if not os.path.exists(GPS): raise FileNotFoundError(GPS)
    gps=np.loadtxt(GPS);a,g=load_imu();xs=[gps,a,g];p=t=None
    if pred:
        if not os.path.exists(PRED): raise FileNotFoundError(PRED)
        p=np.loadtxt(PRED).reshape(-1);xs.append(p)
    if gt:
        if not os.path.exists(GT): raise FileNotFoundError(GT)
        t=np.loadtxt(GT);xs.append(t)
    n=min(map(len,xs))
    return gps[:n],a[:n],g[:n],None if p is None else p[:n],None if t is None else t[:n]

def reactive(gps,scale=6.0):
    r=np.zeros(len(gps))
    for i in range(2,len(gps)):
        q=gps[i-1]+gps[i-1]-gps[i-2];r[i]=np.linalg.norm(gps[i]-q)
    if len(r)>2:r[:2]=r[2]
    return residual_reliability(r,scale),r

def oracle(gps,gt,scale=8.0):
    e=np.linalg.norm(gps-gt,axis=1)
    return np.clip(np.exp(-.5*(e/max(scale,1e-6))**2),0,1),e

def sigma(r): return reliability_to_sigma(np.clip(r,0,1),3.0,30.0,2.0)

def run(gps,a,g,sig,out,title,reliability=None,extra=None,huber=True):
    n=min(len(gps),len(a),len(g),len(sig));gps,a,g,sig=gps[:n],a[:n],g[:n],np.asarray(sig)[:n]
    graph=NonlinearFactorGraph();initial=Values();params=PreintegrationParams.MakeSharedU(9.81)
    params.setAccelerometerCovariance(np.eye(3)*.1);params.setGyroscopeCovariance(np.eye(3)*.01);params.setIntegrationCovariance(np.eye(3)*.001)
    b0=gtsam.imuBias.ConstantBias();pn=noiseModel.Isotropic.Sigma(6,.1);vn=noiseModel.Isotropic.Sigma(3,1);bn=noiseModel.Isotropic.Sigma(6,1e-3)
    p0=Pose3(Rot3(),Point3(*gps[0]));graph.add(PriorFactorPose3(symbol("x",0),p0,pn));graph.add(PriorFactorVector(symbol("v",0),np.zeros(3),vn))
    initial.insert(symbol("x",0),p0);initial.insert(symbol("v",0),np.zeros(3));initial.insert(symbol("b",0),b0)
    print("="*60);print(title);print("Frames:",n);print("Sigma min/max/mean:",sig.min(),sig.max(),sig.mean())
    def gpsnoise(s):
        base=noiseModel.Isotropic.Sigma(3,float(s))
        return noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(1.345),base) if huber else base
    for i in range(n-1):
        pim=PreintegratedImuMeasurements(params,b0);pim.integrateMeasurement(a[i],g[i],DT)
        graph.add(ImuFactor(symbol("x",i),symbol("v",i),symbol("x",i+1),symbol("v",i+1),symbol("b",i),pim))
        graph.add(BetweenFactorConstantBias(symbol("b",i),symbol("b",i+1),gtsam.imuBias.ConstantBias(),bn))
        graph.add(GPSFactor(symbol("x",i),Point3(*gps[i]),gpsnoise(sig[i])))
        initial.insert(symbol("x",i+1),Pose3(Rot3(),Point3(*gps[i+1])));initial.insert(symbol("v",i+1),np.zeros(3));initial.insert(symbol("b",i+1),b0)
        if i%500==0:print("Add factors:",i)
    graph.add(GPSFactor(symbol("x",n-1),Point3(*gps[-1]),gpsnoise(sig[-1])))
    print("Graph:",graph.size());print("Optimizing...")
    result=LevenbergMarquardtOptimizer(graph,initial).optimize()
    tr=np.array([[*result.atPose3(symbol("x",i)).translation()] for i in range(n)],float)
    os.makedirs(out,exist_ok=True);np.savetxt(os.path.join(out,"trajectory.txt"),tr,fmt="%.8f");np.savetxt(os.path.join(out,"sigma.txt"),sig,fmt="%.8f")
    if reliability is not None:np.savetxt(os.path.join(out,"reliability.txt"),np.asarray(reliability)[:n],fmt="%.8f")
    for k,v in (extra or {}).items():np.savetxt(os.path.join(out,k),np.asarray(v)[:n],fmt="%.8f")
    print("Trajectory:",tr.shape);print("Saved:",out)
