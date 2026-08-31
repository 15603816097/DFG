#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo "Stage 6: V5-GRU vs V6 Official Mamba Factor Graph"
echo "Project: $ROOT"
echo "============================================================"

python experiments/factor_graph/run_stage6_v6_mamba_factor_graph_ablation.py
