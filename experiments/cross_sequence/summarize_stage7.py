from __future__ import annotations
import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.cross_sequence.stage7_config import TEST_SEQUENCES,OUT_ROOT
cases={}
for seq in TEST_SEQUENCES:
    p=OUT_ROOT/seq/"stage7_results.csv"
    if not p.exists():raise FileNotFoundError(p)
    with p.open() as f:
        for r in csv.DictReader(f):cases.setdefault(r["case"],[]).append((seq,r))
rows=[]
for case,items in cases.items():
    vals=np.array([float(r["ATE3D"]) for _,r in items]);a2=np.array([float(r["ATE2D"]) for _,r in items]);z=np.array([float(r["ZRMSE"]) for _,r in items])
    rows.append({"case":case,"n_sequences":len(vals),"ATE3D_mean":vals.mean(),"ATE3D_std":vals.std(),
                 "ATE2D_mean":a2.mean(),"ZRMSE_mean":z.mean()})
fixed=next(r["ATE3D_mean"] for r in rows if r["case"]=="S0_fixed")
for r in rows:r["vs_fixed_mean_pct"]=(fixed-r["ATE3D_mean"])/fixed*100
rows.sort(key=lambda x:x["ATE3D_mean"])
p=OUT_ROOT/"stage7_cross_sequence_summary.csv";p.parent.mkdir(parents=True,exist_ok=True)
with p.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print("="*100);print("STAGE-7 CROSS-SEQUENCE SUMMARY")
for i,r in enumerate(rows,1):print(f"{i:02d}. {r['case']:20s} ATE3D={r['ATE3D_mean']:.6f}±{r['ATE3D_std']:.6f} vs_fixed={r['vs_fixed_mean_pct']:+.2f}%")
print("Saved:",p)
