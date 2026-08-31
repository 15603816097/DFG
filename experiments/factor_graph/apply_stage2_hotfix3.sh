#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python experiments/factor_graph/apply_stage2_hotfix3.py

echo
echo "Hotfix 3 complete."
echo "Run:"
echo "bash experiments/factor_graph/run_stage2_v4_ablation.sh"
