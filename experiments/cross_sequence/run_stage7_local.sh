#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "============================================================"
echo "Stage 7 local: build data + V5 inference"
echo "NO MODEL TRAINING"
echo "============================================================"
for seq in 2011_10_03_drive_0034_sync 2011_10_03_drive_0042_sync 2011_10_03_drive_0047_sync; do
  python experiments/cross_sequence/build_sequence_data.py --sequence "$seq"
  python experiments/cross_sequence/infer_v5_gru.py --sequence "$seq"
done
python experiments/cross_sequence/check_stage7.py
echo
echo "Local preprocessing complete."
echo "Next: V6 frozen inference. If local mamba_ssm is unavailable, use the cloud script."
