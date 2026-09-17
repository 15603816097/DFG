
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v6_mamba_uncertainty"

for sensor in ("gps", "imu", "camera"):
    r = np.loadtxt(OUT / f"{sensor}_reliability_target_aligned.txt")
    u = np.loadtxt(OUT / f"{sensor}_uncertainty_target_aligned.txt")
    m = json.loads((OUT / sensor / "metrics.json").read_text())

    assert len(r) == len(u) == 4544
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(u))
    assert np.min(r) >= 0.0 and np.max(r) <= 1.0
    assert np.min(u) >= 0.0 and np.max(u) <= 1.0
    assert m["backend"] == "mamba"

    print(
        f"{sensor.upper():8s} Corr={m['test_corr']:.6f} | "
        f"r={r.min():.4f}/{r.max():.4f}/{r.mean():.4f} | "
        f"u={u.min():.4f}/{u.max():.4f}/{u.mean():.4f}"
    )

print("V6 OUTPUT CHECK PASS")
