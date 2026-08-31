from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.dataset.factor_reliability_dataset import SENSORS,PredictiveFactorReliabilityDataset,compute_normalization,apply_normalization
from src.model.multisensor_predictive_reliability_model import MultiSensorPredictiveReliabilityModel

SEED=20260827;SL=64;BS=64;EPOCHS=120;PATIENCE=18;LR=5e-4;WD=1e-4;CW=.30;FW=1.;LOW=2.5
def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--data",default=str(ROOT/"results/predictive_factor_reliability_v2_corrected/factor_reliability_training_data.npz"))
 p.add_argument("--output-dir",default=str(ROOT/"results/predictive_factor_reliability_v2_corrected"))
 a=p.parse_args();seed();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 r=np.load(a.data,allow_pickle=False);h=int(np.asarray(r["horizon"]).reshape(-1)[0])
 f={};c={};u={};lens=[]
 for s in SENSORS:
  f[s]=np.asarray(r[f"{s}_features"],np.float32);c[s]=np.asarray(r[f"{s}_current_factor_reliability"],np.float32).reshape(-1)
  u[s]=np.asarray(r[f"{s}_future_factor_reliability"],np.float32).reshape(-1);lens += [len(f[s]),len(c[s]),len(u[s])]
 n=min(lens)
 for s in SENSORS:f[s]=f[s][:n];c[s]=c[s][:n];u[s]=u[s][:n]
 tr=int(n*.70);va=int(n*.85);norm=compute_normalization(f,tr);fn=apply_normalization(f,norm)
 np.savez_compressed(out/"factor_reliability_normalization.npz",**{f"{s}_{k}":norm[s][k] for s in SENSORS for k in ("mean","std")})
 ids=np.arange(SL-1,n-h,dtype=np.int64);ti=ids[ids<tr];vi=ids[(ids>=tr)&(ids<va)];xi=ids[ids>=va]
 def ld(ii,sh):return DataLoader(PredictiveFactorReliabilityDataset(fn,c,u,ii,sequence_length=SL),batch_size=BS,shuffle=sh)
 tl,vl,xl=ld(ti,True),ld(vi,False),ld(xi,False)
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 m=MultiSensorPredictiveReliabilityModel(gps_dim=fn["gps"].shape[1],imu_dim=fn["imu"].shape[1],lidar_dim=fn["lidar"].shape[1],
 camera_dim=fn["camera"].shape[1],sensor_embed_dim=32,hidden_dim=128,num_layers=2,dropout=.10).to(dev)
 print("Device:",dev,"Backend:",m.backend,"Frames:",n,"H:",h,"Train/Val/Test:",len(ti),len(vi),len(xi))
 opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD);best=float("inf");be=-1;wait=0;bp=out/"best_factor_reliability_model.pt"
 for ep in range(1,EPOCHS+1):
  m.train();ls=[]
  for b in tl:
   x={s:b[f"{s}_feature"].to(dev) for s in SENSORS};pr=m(x["gps"],x["imu"],x["lidar"],x["camera"]);loss=torch.zeros((),device=dev)
   for s in SENSORS:
    yc=b[f"{s}_current_label"].to(dev);yf=b[f"{s}_future_label"].to(dev)
    loss += CW*torch.mean((pr[s]["current"]-yc)**2)+FW*torch.mean((1+LOW*(1-yf)**2)*(pr[s]["future"]-yf)**2)
   loss/=len(SENSORS);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();ls.append(loss.item())
  m.eval();vm=[]
  with torch.no_grad():
   for s in SENSORS:
    yy=[];pp=[]
    for b in vl:
     x={q:b[f"{q}_feature"].to(dev) for q in SENSORS};pr=m(x["gps"],x["imu"],x["lidar"],x["camera"])
     yy.append(b[f"{s}_future_label"].numpy());pp.append(pr[s]["future"].cpu().numpy())
    y=np.concatenate(yy);q=np.concatenate(pp);vm.append(np.mean((y-q)**2))
  sc=float(np.mean(vm));print(f"Epoch {ep:03d} train={np.mean(ls):.6f} val={sc:.6f}")
  if sc<best:best=sc;be=ep;wait=0;torch.save(m.state_dict(),bp)
  else:
   wait+=1
   if wait>=PATIENCE:print("Early stopping");break
 m.load_state_dict(torch.load(bp,map_location=dev));m.eval();metrics={}
 with torch.no_grad():
  for s in SENSORS:
   yy=[];pp=[]
   for b in xl:
    x={q:b[f"{q}_feature"].to(dev) for q in SENSORS};pr=m(x["gps"],x["imu"],x["lidar"],x["camera"])
    yy.append(b[f"{s}_future_label"].numpy());pp.append(pr[s]["future"].cpu().numpy())
   y=np.concatenate(yy);q=np.concatenate(pp);corr=float(np.corrcoef(y,q)[0,1]) if np.std(y)>1e-12 and np.std(q)>1e-12 else 0.
   metrics[s]={"mae":float(np.mean(abs(y-q))),"rmse":float(np.sqrt(np.mean((y-q)**2))),"corr":corr}
 summary={"best_epoch":be,"best_validation_mse":best,"backend":m.backend,"sequence_length":SL,"horizon":h,"test_metrics":metrics}
 (out/"training_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 print("Best epoch:",be);[print(s,metrics[s]) for s in SENSORS];print("Saved:",bp)
if __name__=="__main__":main()
