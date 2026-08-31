#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
echo "============================================================"
echo "Stage 6.5: Encoder Combination + XY/Z Mechanism"
echo "============================================================"
python experiments/factor_graph/run_stage6_5_encoder_combination_ablation.py
