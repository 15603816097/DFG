import os,sys,numpy as np,gtsam
from gtsam import symbol,Values,NonlinearFactorGraph,LevenbergMarquardtOptimizer,Pose3,Point3,Rot3,PriorFactorPose3,PriorFactorVector,GPSFactor,ImuFactor,BetweenFactorConstantBias,PreintegratedImuMeasurements,PreintegrationParams,noiseModel
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from src.loader.imu_loader import IMULoader
from src.reliability.reliability_mapper import reliability_to_sigma
from src.reliability.feedback_corrector import residual_reliability,fuse_reliability
DATASET=os.path.join(ROOT,"dataset","kitti","2011_10_03","2011_10_03_drive_0027_sync")
GPS=os.path.join(ROOT,"results","progressive_gps_degradation","gps_corrupted.txt")
PRED=os.path.join(ROOT,"results","predictive_reliability_new","predicted_reliability.txt")
OUT=os.path.join(ROOT,"results","predictive_reliability_fg")
COMP=os.path.join(ROOT,"results","dynamic_covariance_fg")
DT=.1

def imu():
    l=IMULoader(DATASET);a=[];g=[]
    for i in range(len(l)):
        d=l[i];a.append(d["acceleration"]);g.append(d["angular_velocity"])
    return np.asarray(a,float),np.asarray(g,float)

def motion_residual(gps):
    r=np.zeros(len(gps))
    for i in range(2,len(gps)):
        q=gps[i-1]+(gps[i-1]-gps[i-2]);r[i]=np.linalg.norm(gps[i]-q)
    return r

def params():
    p=PreintegrationParams.MakeSharedU(9.81);p.setAccelerometerCovariance(np.eye(3)*.1)
    p.setGyroscopeCovariance(np.eye(3)*.01);p.setIntegrationCovariance(np.eye(3)*.001);return p

def main():
    gps=np.loadtxt(GPS);pred=np.loadtxt(PRED).reshape(-1);a,g=imu();n=min(len(gps),len(pred),len(a))
    gps,pred,a,g=gps[:n],pred[:n],a[:n],g[:n]
    res=motion_residual(gps);fb=residual_reliability(res,6);final=fuse_reliability(pred,fb,.7)
    sig=reliability_to_sigma(final,3,30,2)
    graph=NonlinearFactorGraph();initial=Values();pa=params();b0=gtsam.imuBias.ConstantBias()
    pn=noiseModel.Isotropic.Sigma(6,.1);vn=noiseModel.Isotropic.Sigma(3,1);bn=noiseModel.Isotropic.Sigma(6,1e-3)
    p0=Pose3(Rot3(),Point3(*gps[0]));graph.add(PriorFactorPose3(symbol("x",0),p0,pn));graph.add(PriorFactorVector(symbol("v",0),np.zeros(3),vn))
    initial.insert(symbol("x",0),p0);initial.insert(symbol("v",0),np.zeros(3));initial.insert(symbol("b",0),b0)
    print("="*60);print("Predictive + Feedback Continuous Dynamic FG")
    for i in range(n-1):
        pim=PreintegratedImuMeasurements(pa,b0);pim.integrateMeasurement(a[i],g[i],DT)
        graph.add(ImuFactor(symbol("x",i),symbol("v",i),symbol("x",i+1),symbol("v",i+1),symbol("b",i),pim))
        graph.add(BetweenFactorConstantBias(symbol("b",i),symbol("b",i+1),gtsam.imuBias.ConstantBias(),bn))
        base=noiseModel.Isotropic.Sigma(3,float(sig[i]));rob=noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(1.345),base)
        graph.add(GPSFactor(symbol("x",i),Point3(*gps[i]),rob))
        initial.insert(symbol("x",i+1),Pose3(Rot3(),Point3(*gps[i+1])));initial.insert(symbol("v",i+1),np.zeros(3));initial.insert(symbol("b",i+1),b0)
        if i%500==0:print("Add factors:",i)
    base=noiseModel.Isotropic.Sigma(3,float(sig[-1]));rob=noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(1.345),base)
    graph.add(GPSFactor(symbol("x",n-1),Point3(*gps[-1]),rob));print("Graph:",graph.size());print("Optimizing...")
    result=LevenbergMarquardtOptimizer(graph,initial).optimize();traj=[]
    for i in range(n):
        t=result.atPose3(symbol("x",i)).translation();traj.append([float(t[0]),float(t[1]),float(t[2])])
    traj=np.asarray(traj);os.makedirs(OUT,exist_ok=True);os.makedirs(COMP,exist_ok=True)
    for name,data in [("trajectory.txt",traj),("predicted_reliability.txt",pred),("feedback_reliability.txt",fb),("final_reliability.txt",final),("sigma.txt",sig),("residual.txt",res)]:
        np.savetxt(os.path.join(OUT,name),data,fmt="%.8f")
    np.savetxt(os.path.join(COMP,"trajectory.txt"),traj,fmt="%.8f")
    print("Trajectory:",traj.shape);print("Final reliability mean:",final.mean());print("Sigma:",sig.min(),sig.max(),sig.mean());print("Saved:",OUT)
if __name__=="__main__":main()
