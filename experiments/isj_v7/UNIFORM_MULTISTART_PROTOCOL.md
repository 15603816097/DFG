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

## Stage 11 — train-only axis-aware GPS noise model (DEVELOPMENT)

The observed seed43 uniform multi-start pilot (all 7 legacy graph cases
reported `stabilized=True`) found Fixed 3D RMSE=2.911867m versus
Mamba-R=3.266197m, GRU96-R=3.304539m, GRU106-R=3.277028m,
Mamba-RU=3.354056m. These are OXTS/GPS-correlated and development-only.
Thus the original isotropic heuristic does not demonstrate a state
estimation advantage; no unseen drive has been tested.

Train-only 0027 GPS diagnostics at anchors 63..3177 (3115 samples):
- Mean/median 3D error = 2.9871m / 0.9845m;
  RMSE=5.7482m; P99=27.3887m.
- XYZ raw error RMS = [4.1457, 3.7861, 1.2329] m;
  XYZ bias [-0.5663, -0.3946, 0.0213] m.
- Lag-1 error corr in X/Y/Z ~ [0.1881, 0.1491, 0.0342].
- Synthetic-oracle label groups [0,.2),[.2,.4),[.4,.6),
  [.6,.8),[.8,1] contained [442,225,241,298,1909] samples.
  Their 3D error RMSE: [14.1104,5.5832,4.1941,2.9225,0.7662] m.

Two NEW base-family hypotheses use only the fixed V7 TRAIN rows:

1. `axis_fixed`: GPS XYZ sigmas are the training error RMS per axis,
   repeated at every of the same 680 h0 validation anchors.
2. `axis_gru96_r`, `axis_gru106_r`, `axis_mamba_r`: Compute
   *per-axis train-only RMS* in five synthetic-oracle reliability bins;
   monotonically limit noisy reversals, floor each at 0.1m, then
   linearly interpolate these train-fitted values at the *saved*,
   time-aligned model's h0 reliability prediction. All model predictions
   remain source-hash verified, and all models share one TRAIN mapping.

The training oracle reliability label is **defined from the same
clean-vs-degraded synthetic GPS error**; it is not independent data.
This procedure learns the oracle-conditioned error pattern, not an
independently validated predictive covariance. The network output
distribution has NOT been calibrated to oracle labels, and score
interpolation is only an exploratory transfer hypothesis.
Furthermore, GPS factors in the original FGO are rebased to frame
3180. Their shared origin error induces cross-time correlations;
the new per-frame XYZ diagonals do not model that common bias.
Historical original 5m fixed/legacy Mamba/GRU remains byte-for-byte
identical when no XYZ override is provided, to protect comparisons.

Existing implementation files were updated IN PLACE, rather than
adding new one-off scripts:
- `src/factor_graph/four_sensor_graph.py`: per-frame GPS sigmas accept
  either `(n,)` isotropic or `(n,3)` XYZ. Same GTSAM Pose3 sigma order.
- `experiments/isj_v7/run_gps_covariance_pilot.py`: computes train
  axis RMS bins and exposes four extra `axis_*` cases.
- `experiments/isj_v7/audit_uniform_multistart.py`: unchanged main
  solver and common-start protocol, now accepts the new case names
  (while retaining old seven-case default to avoid silent changes).
- Existing test file adds train-only leakage, shape, monotonic mapping
  and scalar-versus-XYZ-equivalent graph regression tests.

The first experiment should be just the new axis-aware cases plus
the original Fixed baseline (no need to rerun neural training):

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
set +e; set +u; set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
TEST_EXIT=$?
echo V7_TEST_EXIT=$TEST_EXIT
if [ "$TEST_EXIT" -eq 0 ]; then
  mkdir -p logs
  python -u -m experiments.isj_v7.audit_uniform_multistart \
    --seed 43 --cases fixed axis_fixed axis_gru96_r \
                        axis_gru106_r axis_mamba_r \
    > logs/isj_v7_axis_train_preflight.log 2>&1
  PREFLIGHT_EXIT=$?
  echo AXIS_PREFLIGHT_EXIT=$PREFLIGHT_EXIT
  if [ "$PREFLIGHT_EXIT" -eq 0 ]; then
    python -u -m experiments.isj_v7.audit_uniform_multistart \
      --run --seed 43 --max-passes 4 \
      --cases fixed axis_fixed axis_gru96_r \
              axis_gru106_r axis_mamba_r \
      > logs/isj_v7_axis_train_multistart_seed43.log 2>&1
    RUN_EXIT=$?
    echo AXIS_MULTISTART_EXIT=$RUN_EXIT
    grep -E '^(SOLVE|RESULT|V7_UNIFORM)' \
      logs/isj_v7_axis_train_multistart_seed43.log
    if [ "$RUN_EXIT" -ne 0 ]; then
      tail -n 80 logs/isj_v7_axis_train_multistart_seed43.log
    fi
  else
    tail -n 80 logs/isj_v7_axis_train_preflight.log
  fi
fi
~~~

All comparisons are same *explored* validation, offline batch LM,
and use correlated OXTS reference only after candidate selection.
Future final study must build a truly independent evaluation-reference
and handle time-correlated GNSS errors; no improvement is guaranteed.
