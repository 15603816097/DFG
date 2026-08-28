from __future__ import annotations
import os,sys,random,csv
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from src.dataset.multisensor_predictive_dataset import SENSORS,load_multisensor_npz,compute_normalization,apply_normalization,MultiSensorPredictiveDataset
from src.model.multisensor_predictive_reliability_model import MultiSensorPredictiveReliabilityModel

DATA_PATH=os.path.join(ROOT,"results","multisensor_reliability","multisensor_reliability_data.npz")
OUTPUT_DIR=os.path.join(ROOT,"results","multisensor_predictive_reliability")
SEED=20260826;SEQ=64;BATCH=64;EPOCHS=100;PATIENCE=15;LR=5e-4;WD=1e-4
CURRENT_W=.35;FUTURE_W=1.0;LOW_REL_EMPHASIS=1.5

def set_seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    mae=float(np.mean(np.abs(y-p)));rmse=float(np.sqrt(np.mean((y-p)**2)))
    corr=float(np.corrcoef(y,p)[0,1]) if np.std(y)>1e-12 and np.std(p)>1e-12 else 0.0
    acc=float(np.mean((p>=.5)==(y>=.5)));bad=y<.5
    recall=float(np.mean(p[bad]<.5)) if np.any(bad) else 1.0
    return {"MAE":mae,"RMSE":rmse,"Correlation":corr,"BinaryAccuracy@0.5":acc,"BadRecall@0.5":recall}

def to_device(batch,device):
    return {s:batch[f"{s}_feature"].to(device) for s in SENSORS}

@torch.no_grad()
def collect(model,loader,device):
    model.eval();ids=[];out={s:{k:[] for k in ("current_y","current_p","future_y","future_p")} for s in SENSORS}
    for batch in loader:
        x=to_device(batch,device);pred=model(x["gps"],x["imu"],x["lidar"],x["camera"])
        ids.extend(batch["frame_id"].numpy().tolist())
        for s in SENSORS:
            out[s]["current_y"].extend(batch[f"{s}_current_label"].numpy().tolist())
            out[s]["current_p"].extend(pred[s]["current"].cpu().numpy().tolist())
            out[s]["future_y"].extend(batch[f"{s}_future_label"].numpy().tolist())
            out[s]["future_p"].extend(pred[s]["future"].cpu().numpy().tolist())
    ids=np.asarray(ids,np.int64)
    for s in SENSORS:
        for k in out[s]: out[s][k]=np.asarray(out[s][k],float)
    return ids,out

def main():
    set_seed();os.makedirs(OUTPUT_DIR,exist_ok=True)
    raw=load_multisensor_npz(DATA_PATH);features=raw["features"];current=raw["current_labels"];future=raw["future_labels"];horizon=raw["horizon"];n=raw["length"]
    train_end=int(n*.70);val_end=int(n*.85)
    norm=compute_normalization(features,train_end);xnorm=apply_normalization(features,norm)
    valid=np.arange(SEQ-1,n,dtype=np.int64);train_ids=valid[valid<train_end];val_ids=valid[(valid>=train_end)&(valid<val_end)];test_ids=valid[valid>=val_end]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*92);print("MULTI-SENSOR PREDICTIVE RELIABILITY TRAINING");print("Device:",device,"Frames:",n,"Horizon:",horizon);print("Train/Val/Test:",len(train_ids),len(val_ids),len(test_ids))
    for s in SENSORS: print(f"{s:8s} dim={xnorm[s].shape[1]}")
    def loader(ids,shuffle):
        return DataLoader(MultiSensorPredictiveDataset(xnorm,current,future,ids,SEQ),batch_size=BATCH,shuffle=shuffle)
    train_loader=loader(train_ids,True);val_loader=loader(val_ids,False);test_loader=loader(test_ids,False)
    model=MultiSensorPredictiveReliabilityModel(xnorm["gps"].shape[1],xnorm["imu"].shape[1],xnorm["lidar"].shape[1],xnorm["camera"].shape[1],32,128,2,.1).to(device)
    print("Temporal backend:",model.backend)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    best=float("inf");best_epoch=-1;wait=0
    for epoch in range(1,EPOCHS+1):
        model.train();losses=[]
        for batch in train_loader:
            x=to_device(batch,device);pred=model(x["gps"],x["imu"],x["lidar"],x["camera"]);loss=torch.zeros((),device=device)
            for s in SENSORS:
                yc=batch[f"{s}_current_label"].to(device);yf=batch[f"{s}_future_label"].to(device)
                lc=torch.mean((pred[s]["current"]-yc)**2)
                wf=1.0+LOW_REL_EMPHASIS*(1.0-yf);lf=torch.mean(wf*(pred[s]["future"]-yf)**2)
                loss=loss+CURRENT_W*lc+FUTURE_W*lf
            loss=loss/len(SENSORS);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.item()))
        _,vo=collect(model,val_loader,device);mses=[];corrs=[]
        for s in SENSORS:
            y,p=vo[s]["future_y"],vo[s]["future_p"];mses.append(float(np.mean((y-p)**2)));corrs.append(f"{s}:{metrics(y,p)['Correlation']:.3f}")
        v=float(np.mean(mses));print(f"Epoch {epoch:03d} | train {np.mean(losses):.6f} | val {v:.6f} | "+" ".join(corrs))
        if v<best:
            best=v;best_epoch=epoch;wait=0;torch.save(model.state_dict(),os.path.join(OUTPUT_DIR,"best_model.pt"))
        else:
            wait+=1
            if wait>=PATIENCE: print("Early stopping.");break
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR,"best_model.pt"),map_location=device))
    test_frame_ids,to=collect(model,test_loader,device)
    print("\n"+"="*92);print("BEST EPOCH:",best_epoch);print("="*92)
    rows=[]
    for s in SENSORS:
        cm=metrics(to[s]["current_y"],to[s]["current_p"]);fm=metrics(to[s]["future_y"],to[s]["future_p"])
        print("\n"+s.upper());print("Current:",cm);print("Future :",fm)
        rows.append({"sensor":s,"head":"current",**cm});rows.append({"sensor":s,"head":"future",**fm})
    with open(os.path.join(OUTPUT_DIR,"test_metrics.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["sensor","head","MAE","RMSE","Correlation","BinaryAccuracy@0.5","BadRecall@0.5"]);w.writeheader();w.writerows(rows)
    with open(os.path.join(OUTPUT_DIR,"test_metrics.txt"),"w",encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\nhorizon={horizon}\nbackend={model.backend}\n\n")
        for r in rows: f.write(f"{r['sensor']} {r['head']} MAE={r['MAE']:.8f} RMSE={r['RMSE']:.8f} Corr={r['Correlation']:.8f} Acc={r['BinaryAccuracy@0.5']:.8f} BadRecall={r['BadRecall@0.5']:.8f}\n")
    norm_kwargs={"sequence_length":np.asarray([SEQ],np.int64),"horizon":np.asarray([horizon],np.int64)}
    for s in SENSORS:
        norm_kwargs[f"{s}_mean"]=norm[s]["mean"];norm_kwargs[f"{s}_std"]=norm[s]["std"]
    np.savez(os.path.join(OUTPUT_DIR,"normalization.npz"),**norm_kwargs)
    all_ids,ao=collect(model,loader(valid,False),device)
    for s in SENSORS:
        cur=np.ones(n);src=np.ones(n);target=np.ones(n);source=np.full(n,-1,np.int64)
        cur[all_ids]=ao[s]["current_p"];src[all_ids]=ao[s]["future_p"]
        for sf,pred in zip(all_ids,ao[s]["future_p"]):
            tf=int(sf)+int(horizon)
            if tf<n: target[tf]=pred;source[tf]=int(sf)
        for i in range(n):
            if source[i]<0: target[i]=cur[i]
        np.savetxt(os.path.join(OUTPUT_DIR,f"{s}_current_prediction.txt"),cur,fmt="%.8f")
        np.savetxt(os.path.join(OUTPUT_DIR,f"{s}_future_prediction_source_aligned.txt"),src,fmt="%.8f")
        np.savetxt(os.path.join(OUTPUT_DIR,f"{s}_predictive_prior_target_aligned.txt"),target,fmt="%.8f")
        np.savetxt(os.path.join(OUTPUT_DIR,f"{s}_prediction_source_frame.txt"),source,fmt="%d")
    np.savetxt(os.path.join(OUTPUT_DIR,"test_frame_ids.txt"),test_frame_ids,fmt="%d")
    print("\nSaved:",OUTPUT_DIR)
if __name__=="__main__": main()
