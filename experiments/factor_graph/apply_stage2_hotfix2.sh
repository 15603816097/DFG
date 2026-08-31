#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/factor_graph/apply_stage2_hotfix2.py
echo
echo "Hotfix 2 complete. Rerun:"
echo "bash experiments/factor_graph/run_stage2_v4_ablation.sh"
