import os,sys,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from src.dataset.multisensor_predictive_dataset import SENSORS,load_multisensor_npz
DATA=os.path.join(ROOT,"results","multisensor_reliability","multisensor_reliability_data.npz")
PRED=os.path.join(ROOT,"results","multisensor_predictive_reliability")
def m(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);mae=np.mean(np.abs(y-p));rmse=np.sqrt(np.mean((y-p)**2))
    corr=np.corrcoef(y,p)[0,1] if np.std(y)>1e-12 and np.std(p)>1e-12 else 0.0
    return float(mae),float(rmse),float(corr)
def main():
    raw=load_multisensor_npz(DATA);h=int(raw["horizon"]);n=raw["length"];start=63+h
    print("="*92);print("MULTI-SENSOR TARGET-ALIGNED PREDICTIVE RELIABILITY");print("Horizon:",h)
    for s in SENSORS:
        p=np.loadtxt(os.path.join(PRED,f"{s}_predictive_prior_target_aligned.txt"),float).reshape(-1);y=raw["current_labels"][s];end=min(n,len(y),len(p))
        mae,rmse,corr=m(y[start:end],p[start:end]);print(f"\n{s.upper()}\nMAE         : {mae:.6f}\nRMSE        : {rmse:.6f}\nCorrelation : {corr:.6f}")
if __name__=="__main__": main()
