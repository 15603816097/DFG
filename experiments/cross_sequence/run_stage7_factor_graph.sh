#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for seq in 2011_10_03_drive_0034_sync 2011_10_03_drive_0042_sync 2011_10_03_drive_0047_sync; do
  python experiments/cross_sequence/run_stage7_factor_graph.py --sequence "$seq"
done
python experiments/cross_sequence/summarize_stage7.py
