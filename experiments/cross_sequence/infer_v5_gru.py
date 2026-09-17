from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.cross_sequence.stage7_config import V5_DIR, TEST_SEQUENCES, sequence_out
from experiments.uncertainty_v5.reliability_uncertainty_model_v5 import ReliabilityUncertaintyV5

def infer(sequence,sensor):
    od=sequence_out(sequence); data=np.load(od/"features.npz",allow_pickle=False)
    x=np.asarray(data[f"{sensor}_features"],np.float32); n=len(x)
    ck=torch.load(V5_DIR/sensor/"model.pt",map_location="cpu",weights_only=False,)
    window=int(ck.get("window",64)); mu=np.asarray(ck["mean"],np.float32); sd=np.asarray(ck["std"],np.float32)
    model=ReliabilityUncertaintyV5(int(ck["input_dim"]),int(ck["hidden_dim"]),int(ck["layers"]),float(ck["dropout"]),int(ck["horizon"]))
    model.load_state_dict(ck["state_dict"]);model.eval()
    r=np.full(n,0.5,float);u=np.ones(n,float)
    with torch.no_grad():
        for t in range(window-1,n):
            xx=(x[t-window+1:t+1]-mu)/sd
            rr,uu,_,_=model(torch.from_numpy(xx[None].astype(np.float32)))
            r[t]=float(rr[0,0]);u[t]=float(uu[0,0])
    out=od/"predictions"/"v5";out.mkdir(parents=True,exist_ok=True)
    np.savetxt(out/f"{sensor}_reliability.txt",r,fmt="%.9f");np.savetxt(out/f"{sensor}_uncertainty.txt",u,fmt="%.9f")
    print(sequence,sensor,"V5",len(r),"rmean",r.mean(),"umean",u.mean())

def main():
    p=argparse.ArgumentParser();p.add_argument("--sequence",required=True,choices=TEST_SEQUENCES);a=p.parse_args()
    for s in ("gps","imu","camera"):infer(a.sequence,s)
if __name__=="__main__":main()
