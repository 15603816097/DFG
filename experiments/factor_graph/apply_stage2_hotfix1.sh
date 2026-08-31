#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/factor_graph/apply_stage2_hotfix1.py
echo
echo "Hotfix complete. Now rerun:"
echo "bash experiments/factor_graph/run_stage2_v4_ablation.sh"
