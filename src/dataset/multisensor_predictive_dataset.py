from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

SENSORS=("gps","imu","lidar","camera")

def load_multisensor_npz(path):
    data=np.load(Path(path),allow_pickle=False)
    features={s:np.asarray(data[f"{s}_features"],np.float32) for s in SENSORS}
    current={s:np.asarray(data[f"{s}_current_label"],np.float32).reshape(-1) for s in SENSORS}
    future={s:np.asarray(data[f"{s}_future_label"],np.float32).reshape(-1) for s in SENSORS}
    n=min([len(features[s]) for s in SENSORS]+[len(current[s]) for s in SENSORS]+[len(future[s]) for s in SENSORS])
    for s in SENSORS:
        features[s]=features[s][:n]
        current[s]=np.clip(current[s][:n],0,1)
        future[s]=np.clip(future[s][:n],0,1)
    horizon=3
    if "horizon" in data:
        horizon=int(np.asarray(data["horizon"]).reshape(-1)[0])
    return {"features":features,"current_labels":current,"future_labels":future,"horizon":horizon,"length":n}

def compute_normalization(features,train_end):
    out={}
    for s in SENSORS:
        x=np.asarray(features[s][:train_end],np.float32)
        mean=x.mean(0);std=x.std(0);std[std<1e-6]=1.0
        out[s]={"mean":mean.astype(np.float32),"std":std.astype(np.float32)}
    return out

def apply_normalization(features,normalization):
    return {s:((features[s]-normalization[s]["mean"])/normalization[s]["std"]).astype(np.float32) for s in SENSORS}

class MultiSensorPredictiveDataset(Dataset):
    def __init__(self,features,current_labels,future_labels,frame_ids,sequence_length=64):
        self.features={s:torch.from_numpy(np.asarray(features[s],np.float32)) for s in SENSORS}
        self.current={s:torch.from_numpy(np.asarray(current_labels[s],np.float32)) for s in SENSORS}
        self.future={s:torch.from_numpy(np.asarray(future_labels[s],np.float32)) for s in SENSORS}
        self.frame_ids=np.asarray(frame_ids,np.int64)
        self.sequence_length=int(sequence_length)
    def __len__(self): return len(self.frame_ids)
    def __getitem__(self,index):
        frame=int(self.frame_ids[index]);start=frame-self.sequence_length+1
        if start<0: raise IndexError("frame_ids must start at sequence_length-1")
        item={"frame_id":frame}
        for s in SENSORS:
            item[f"{s}_feature"]=self.features[s][start:frame+1]
            item[f"{s}_current_label"]=self.current[s][frame]
            item[f"{s}_future_label"]=self.future[s][frame]
        return item
