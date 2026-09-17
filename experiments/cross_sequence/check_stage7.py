from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.cross_sequence.stage7_config import *
bad=[]
for seq in TEST_SEQUENCES:
    od=sequence_out(seq)
    for p in (od/"features.npz",od/"sensor_data.npz",od/"ground_truth.txt",od/"lidar_factor_data.npz",od/"camera_factor_data.npz"):
        if not p.exists():bad.append(str(p))
    if (od/"features.npz").exists():
        d=np.load(od/"features.npz");lens={len(d[k]) for k in d.files}
        if len(lens)!=1:bad.append(f"{seq}: feature lengths={lens}")
    # leakage guard: Stage-7 outputs must never contain 0027 target-aligned files.
    for p in od.rglob("*0027*"):
        bad.append(f"leakage-like filename: {p}")
if bad:
    print("\n".join(bad));raise SystemExit("STAGE7 CHECK FAILED")
print("STAGE7 DATA CHECK PASS")
