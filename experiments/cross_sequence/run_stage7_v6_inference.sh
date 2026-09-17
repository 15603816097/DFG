#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "============================================================"
echo "Stage 7 V6 Official Mamba FROZEN INFERENCE"
echo "This is inference only. DO NOT TRAIN on test sequences."
echo "============================================================"
python - <<'PY'
import torch
from mamba_ssm import Mamba
print("torch",torch.__version__,"cuda",torch.cuda.is_available())
print("mamba_ssm import PASS")
PY
for seq in 2011_10_03_drive_0034_sync 2011_10_03_drive_0042_sync 2011_10_03_drive_0047_sync; do
  python experiments/cross_sequence/infer_v6_mamba.py --sequence "$seq"
done
