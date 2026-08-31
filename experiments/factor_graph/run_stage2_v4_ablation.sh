#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "STAGE 2: 0027 V4 FACTOR-GRAPH ABLATION"
echo "============================================================"

python experiments/factor_graph/run_stage2_v4_factor_graph_ablation.py

echo
echo "============================================================"
echo "STAGE 2 COMPLETE"
echo "============================================================"
