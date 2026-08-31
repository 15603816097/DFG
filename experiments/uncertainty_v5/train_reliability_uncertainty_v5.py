from __future__ import annotations
import json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.uncertainty_v5.reliability_uncertainty_model_v5 import ReliabilityUncertaintyV5, evidential_loss

DATA = ROOT / "results" / "predictive_factor_reliability_v3_same_source" / "factor_reliability_training_data.npz"
OUT = ROOT / "results" / "sensor_specific_reliability_v5_uncertainty"

WINDOW = 64
HORIZON = 3
HIDDEN_DIM = 96
LAYERS = 2
DROPOUT = 0.1
EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4
SEED = 42
SENSORS = ("gps", "imu", "camera")

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def find_key(data, candidates, required=True):
    for k in candidates:
        if k in data.files:
            return k
    if required:
        raise KeyError(f"Missing keys {candidates}. Available={data.files}")
    return None

def load_sensor(sensor):
    data = np.load(DATA, allow_pickle=False)
    fk = find_key(data, [f"{sensor}_features", f"features_{sensor}"])
    ck = find_key(data, [
        f"{sensor}_current_factor_reliability",
        f"{sensor}_reliability",
        f"reliability_{sensor}",
        f"{sensor}_label",
    ])
    hk = find_key(data, [
        f"{sensor}_future_factor_reliability",
        f"{sensor}_future_reliability",
        f"future_reliability_{sensor}",
    ], required=False)

    x = np.asarray(data[fk], np.float32)
    cur = np.asarray(data[ck], np.float32).reshape(-1)
    n = min(len(x), len(cur))
    x, cur = x[:n], np.clip(cur[:n], 0, 1)

    fut = None
    if hk is not None:
        fut = np.asarray(data[hk], np.float32)
        if fut.ndim == 1:
            fut = fut[:, None]
        fut = fut[:n]

    print(f"{sensor.upper()} keys: features={fk}, current={ck}, future={hk}")
    print(f"{sensor.upper()} shapes: x={x.shape}, current={cur.shape}, future={None if fut is None else fut.shape}")
    return x, cur, fut

def build_targets(cur, fut):
    n = len(cur)
    y = np.full((n, HORIZON), np.nan, np.float32)
    y[:, 0] = cur

    if fut is not None and fut.ndim == 2 and fut.shape[1] > 0:
        usable = min(HORIZON - 1, fut.shape[1])
        y[:, 1:1+usable] = fut[:, :usable]
        start = 1 + usable
    else:
        start = 1

    for h in range(start, HORIZON):
        y[:-h, h] = cur[h:]
    return y

class DS(Dataset):
    def __init__(self, x, y, idx, mu, sd):
        self.x=x; self.y=y; self.idx=np.asarray(idx,int); self.mu=mu; self.sd=sd
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        t=int(self.idx[j])
        xx=(self.x[t-WINDOW+1:t+1]-self.mu)/self.sd
        return torch.from_numpy(xx.astype(np.float32)), torch.from_numpy(self.y[t].astype(np.float32)), t

def corr(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return 0.0
    return float(np.corrcoef(a,b)[0,1])

def loader(x,y,idx,mu,sd,shuffle=False):
    return DataLoader(DS(x,y,idx,mu,sd), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

def train_sensor(sensor, device):
    x, cur, fut = load_sensor(sensor)
    y = build_targets(cur, fut)
    n = len(cur)

    valid = np.arange(WINDOW-1, n)
    valid = valid[np.all(np.isfinite(y[valid]), axis=1)]

    cut1 = int(0.70*n)
    cut2 = int(0.85*n)
    tr = valid[valid < cut1]
    va = valid[(valid >= cut1) & (valid < cut2)]
    te = valid[valid >= cut2]

    mu = x[:cut1].mean(0, keepdims=True)
    sd = np.maximum(x[:cut1].std(0, keepdims=True), 1e-6)

    m = ReliabilityUncertaintyV5(x.shape[1], HIDDEN_DIM, LAYERS, DROPOUT, HORIZON).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)

    best = float("inf"); state=None; bad=0; best_epoch=-1
    print(f"\n{sensor.upper()} backend={m.backend} frames={n} dim={x.shape[1]} train/val/test={len(tr)}/{len(va)}/{len(te)}")

    for ep in range(EPOCHS):
        m.train(); train_loss=[]
        for xb,yb,_ in loader(x,y,tr,mu,sd,True):
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad()
            r,u,a,b=m(xb)
            loss=evidential_loss(yb,r,a,b,ep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),5.0)
            opt.step()
            train_loss.append(loss.item())

        m.eval(); val_loss=[]
        with torch.no_grad():
            for xb,yb,_ in loader(x,y,va,mu,sd):
                xb,yb=xb.to(device),yb.to(device)
                r,u,a,b=m(xb)
                val_loss.append(evidential_loss(yb,r,a,b,ep).item())

        v=float(np.mean(val_loss))
        print(f"{sensor} epoch {ep+1:02d} train={np.mean(train_loss):.6f} val={v:.6f}")
        if v < best - 1e-5:
            best=v; bad=0; best_epoch=ep+1
            state={k:q.detach().cpu().clone() for k,q in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break

    if state is None:
        raise RuntimeError(f"{sensor}: no valid checkpoint")
    m.load_state_dict(state)

    # Test metrics
    ps=[]; ys=[]; us=[]
    m.eval()
    with torch.no_grad():
        for xb,yb,_ in loader(x,y,te,mu,sd):
            r,u,a,b=m(xb.to(device))
            ps.extend(r[:,0].cpu().numpy()); ys.extend(yb[:,0].numpy()); us.extend(u[:,0].cpu().numpy())
    ps=np.asarray(ps); ys=np.asarray(ys); us=np.asarray(us)

    metrics={
        "sensor":sensor,
        "backend":m.backend,
        "best_epoch":best_epoch,
        "best_val":best,
        "test_mae":float(np.mean(np.abs(ps-ys))),
        "test_rmse":float(np.sqrt(np.mean((ps-ys)**2))),
        "test_corr":corr(ps,ys),
        "uncertainty_test_min":float(us.min()),
        "uncertainty_test_max":float(us.max()),
        "uncertainty_test_mean":float(us.mean()),
        "uncertainty_test_std":float(us.std()),
    }

    od=OUT/sensor
    od.mkdir(parents=True,exist_ok=True)
    torch.save({
        "state_dict":m.state_dict(),
        "input_dim":x.shape[1],
        "hidden_dim":HIDDEN_DIM,
        "layers":LAYERS,
        "dropout":DROPOUT,
        "horizon":HORIZON,
        "window":WINDOW,
        "mean":mu,
        "std":sd,
        "backend":m.backend,
        "best_epoch":best_epoch,
    }, od/"model.pt")
    (od/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print("TEST",metrics)

    # Target-aligned predictions
    r_all=np.full(n,0.5,float)
    u_all=np.ones(n,float)
    with torch.no_grad():
        for xb,yb,ti in loader(x,y,valid,mu,sd):
            r,u,a,b=m(xb.to(device))
            rr=r[:,0].cpu().numpy(); uu=u[:,0].cpu().numpy()
            for t,rv,uv in zip(ti.numpy(),rr,uu):
                r_all[int(t)]=float(rv)
                u_all[int(t)]=float(uv)

    np.savetxt(OUT/f"{sensor}_reliability_target_aligned.txt",r_all,fmt="%.9f")
    np.savetxt(OUT/f"{sensor}_uncertainty_target_aligned.txt",u_all,fmt="%.9f")
    print(f"{sensor} reliability min/max/mean/std: {r_all.min():.6f}/{r_all.max():.6f}/{r_all.mean():.6f}/{r_all.std():.6f}")
    print(f"{sensor} uncertainty min/max/mean/std: {u_all.min():.6f}/{u_all.max():.6f}/{u_all.mean():.6f}/{u_all.std():.6f}")

def main():
    seed_all()
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    OUT.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    print("Dataset:",DATA)

    data=np.load(DATA,allow_pickle=False)
    print("Available arrays:")
    for k in data.files:
        print(" ",k,np.asarray(data[k]).shape)

    for s in SENSORS:
        train_sensor(s,device)

    print("\nV5 TRAINING COMPLETE")
    print("Output:",OUT)

if __name__ == "__main__":
    main()
