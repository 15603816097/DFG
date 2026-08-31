#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "STAGE 5 / V5-GRU FACTOR GRAPH ABLATION"
echo "============================================================"

python -m py_compile \
    experiments/factor_graph/run_stage5_v5_gru_factor_graph_ablation.py

echo "Syntax PASS"

python experiments/factor_graph/run_stage5_v5_gru_factor_graph_ablation.py
