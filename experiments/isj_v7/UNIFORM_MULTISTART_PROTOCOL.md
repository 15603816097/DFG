# V7 stage 10 — Equal-budget multi-start, read-only FGO DEV comparison

## Why stage 10 is needed

On development KITTI 0027 VAL (680 h0 anchors, 3180:3860),
uniform Fixed-initialization audit of LM 100-iteration continuation
produced the following OXTS-correlated, **non-independent** results:

| FGO GPS sigma | 3D RMSE m | 2D RMSE m | local RPE1 m | local RPE10 m |
| --- | ---: | ---: | ---: | ---: |
| Fixed, continued | 3.1110 | 2.7361 | 0.1957 | 1.3217 |
| Mamba-R, restarted from saved Fixed | 4.0512 | 3.0977 | 0.2195 | 1.6468 |
| Mamba mean sigma, restarted from saved Fixed | 4.3102 | 3.7522 | 0.2267 | 1.6980 |
| Mamba permuted sigma, restarted from saved Fixed | 3.7842 | 3.2918 | 0.2138 | 1.5371 |

Yet Mamba-R continued from its own previous optimized poses had
3D RMSE 3.2662m and objective 914.3809 versus 4.0512m and
objective 1154.8263 from saved Fixed initialization **under the
same Mamba-R graph**. Thus solver basin matters. Cross-case
objectives have different scales and MUST NOT be compared.

## Read-only protocol (post-hoc, NOT confirmatory)

New script: `experiments/isj_v7/audit_uniform_multistart.py`.

- Rebuilds EXACT V7 four-sensor batch FGO for each case.
- Same KITTI 0027 dev VAL 680-frame subgraph and exact GPS prediction anchors.
- Same three initial states offered to EVERY covariance method:
  1. 'lidar': raw LiDAR-integrated initial trajectory;
  2. 'fixed_saved': saved Fixed graph result;
  3. 'mamba_saved': saved Mamba-R graph result.
- The common pool itself was chosen *after* examining results and
  therefore remains a post-hoc exploration, even if used identically.
- GTSAM LM <=100 iterations per pass; same convergence budgets;
  <=4 repeated passes per start unless explicitly set 2..12.
- Stop only when relative objective improvement <=1e-6 AND position
  shift RMSE <=0.01m within an LM pass.
- Select lowest observed **within-method** graph objective of the three
  starts. No OXTS reference is loaded or used before all selections.
- Each method prints every start's objective, pass count and stability.
  If starts fail stability, do not call results converged.
- AFTER ALL graphs and selections, compute 3D/2D and 1-/10-step
  OXTS-relative displacement diagnostics. OXTS/GPS are not independent.
- NO project files modified and no previous model/trajectory overwritten.
  Users can tee the stdout to logs.
- Only seed 43 initially. Original saved Fixed and Mamba-R
  pilot outputs are required as validated starting trajectories.
- The script accepts Fixed, GRU96-R, GRU106-R, Mamba-R, Mamba-RU,
  Mamba mean sigma, Mamba permuted sigma, and optional No-GPS.
  Other modalities maintain fixed covariance; Mamba-RU is a bounded
  *uncalibrated* evidence proxy, not calibrated measurement covariance.

## Suggested server commands

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
set +e; set +u; set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
V7_TEST_RC=$?
echo V7_TEST_RC=$V7_TEST_RC
if [ "$V7_TEST_RC" -eq 0 ]; then
  python -u -m experiments.isj_v7.audit_uniform_multistart
  PRECHECK_RC=$?
  echo UNIFORM_PREFLIGHT_RC=$PRECHECK_RC
  if [ "$PRECHECK_RC" -eq 0 ]; then
    mkdir -p logs
    python -u -m experiments.isj_v7.audit_uniform_multistart \
      --run --seed 43 --max-passes 4 \
      --cases fixed gru96_r gru106_r mamba_r mamba_r_u \
              mamba_mean_sigma mamba_perm_sigma \
      > logs/isj_v7_uniform_multistart_seed43.log 2>&1
    SOLVE_RC=$?
    echo UNIFORM_SOLVE_RC=$SOLVE_RC
    grep -E '^(SOLVE|RESULT|V7_UNIFORM)' \
      logs/isj_v7_uniform_multistart_seed43.log
  fi
fi
~~~

No held-out 0016/0020 evaluation, no independent state-estimation
GT, no prospective iSAM2 inference and no correction to the
well-documented Beta evidence miscalibration is claimed here.


## Consolidation / cleanup after uniform multi-start

The previous single-case `audit_fg_convergence.py` was superseded
by `audit_uniform_multistart.py`. Its shared source-hash, saved-pose,
trajectory-consistency and sigma-integrity checks are now implemented
inside the active uniform multi-start module. The redundant single-case
script, its overlapping test file and `CONVERGENCE_AUDIT.md` were
removed. Regression tests for the retained integrity checks continue
to run in `test_isj_v7_uniform_multistart.py`.

The seed43 uniform audit (all three starts numerically stable under the
configured stopping test) produced 3D OXTS-relative RMSE:
Fixed 2.911867 m; GRU96 3.304539 m; GRU106 3.277028 m;
Mamba-R 3.266197 m; Mamba-RU 3.354056 m;
Mamba-Mean 3.758092 m; Mamba-Permuted 3.689006 m.
Fixed is the best of these seven tested settings, so the current
heuristic dynamic GPS covariance does **not** demonstrate improved
state estimation. Numerically stable LM does not establish a unique
or global minimum; OXTS/GPS reference is not independent and these
remain **development-only/post-hoc** findings.
