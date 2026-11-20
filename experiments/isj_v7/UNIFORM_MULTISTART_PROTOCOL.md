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

## Stage 12 — Frozen post-hoc 4x GPS information cap (one setting only)

The existing 0027 VAL confidence diagnostic showed:

| h0 model | VAL MAE to synthetic oracle | r>=0.9 fraction | high r>=0.9 on oracle r<0.4 |
| --- | ---: | ---: | ---: |
| GRU96 | 0.15595 | 48.82% | 2 / 680 |
| GRU106 | 0.15272 | 48.97% | 2 / 680 |
| Mamba | 0.14001 | 51.76% | 2 / 680 |

Thus the hypothesis that Mamba uniquely mislabels many BAD GPS
observations as highly reliable is NOT supported by this diagnostic.
Mamba was actually more conservative than the GRUs when synthetic
oracle reliability was <0.4 (mean predicted score ~0.262 vs
~0.309/0.325), while emitting high reliability more often overall.

Yet the unbounded relative axis-precision scaling reached:
maximum X 67.04x, Y 61.62x, Z 15.50x relative to TRAIN
fixed-axis precision. Mamba hit the max at the median; GRU did
not. These severe factors may cause instability from imperfect
prediction calibration, correlated GPS errors or numerical
conditioning. This does not establish a causal explanation.

Exactly ONE conservative exploratory cap is introduced *after* seeing
those developer diagnostics. No other cap values are evaluated:

  sigma_cap(t,a) = max(sigma_axis_predicted(t,a),
                        sigma_axis_train_fixed(a) / sqrt(4))

Therefore 1/sigma_cap^2 <= 4/sigma_axis_train_fixed^2 in each XYZ
axis. This retains severe-observation downweighting, limits
high-information observations, and is identical for all three
backbones. It is NOT a calibrated GPS likelihood or prespecified
confirmation. It does NOT use validation reference in factors.

Added in-place to `run_gps_covariance_pilot.py`, sharing the
existing multi-start read-only FGO script:

- `axis_gru96_r_cap4`
- `axis_gru106_r_cap4`
- `axis_mamba_r_cap4`

The original three `axis_*_r` cases and `axis_fixed` remain
unchanged. The uniform audit still uses the same three starts,
four-sensor factors and GTSAM stopping rule, and chooses a result
by each graph's objective alone.

Read-only smoke and actual dev diagnostic:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
set +e; set +u; set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
TEST_RC=$?
echo V7_TEST_RC=$TEST_RC
if [ "$TEST_RC" -eq 0 ]; then
  mkdir -p logs
  python -u -m experiments.isj_v7.audit_uniform_multistart \
    --seed 43 --cases axis_fixed axis_gru96_r_cap4 \
                      axis_gru106_r_cap4 axis_mamba_r_cap4 \
    > logs/isj_v7_cap4_preflight.log 2>&1
  PRE_RC=$?
  echo CAP4_PREFLIGHT_RC=$PRE_RC
  if [ "$PRE_RC" -eq 0 ]; then
    python -u -m experiments.isj_v7.audit_uniform_multistart \
      --run --max-passes 12 --seed 43 \
      --cases axis_fixed axis_gru96_r_cap4 \
              axis_gru106_r_cap4 axis_mamba_r_cap4 \
      > logs/isj_v7_cap4_multistart_seed43.log 2>&1
    RUN_RC=$?
    echo CAP4_MULTISTART_RC=$RUN_RC
    grep -E '^(SOLVE|RESULT|V7_UNIFORM)' \
      logs/isj_v7_cap4_multistart_seed43.log
    if [ "$RUN_RC" -ne 0 ]; then
      tail -n 80 logs/isj_v7_cap4_multistart_seed43.log
    fi
  else
    tail -n 80 logs/isj_v7_cap4_preflight.log
  fi
fi
~~~

If capped Mamba does not beat `axis_fixed` or capacity-matched GRU106
under adequate convergence, record the negative result. Do not tune
the cap against already explored VAL, report as independent test,
or modify trained checkpoints. Independently referenced evaluation
and prospective causal FGO remain unresolved prerequisites.

## Stage 13 — development settings locked after Cap4 diagnostic (2026-11-20)

This is a DEVELOPMENT-only algorithm-settings lock, NOT final test
approval. Completed on known 0027 VAL GPS h0 anchors 3180..3859,
with three common post-hoc start poses and GTSAM LM <=100 iterations
per pass up to 12 passes. All four selected Cap4 cases reported
`selected_stabilized=True` under the stated numerical stopping rule.

**Complete OXTS/GPS-correlated development results (meters):**

| GPS covariance | origin-aligned 3D RMSE | 2D RMSE | 1-frame displacement RMSE | 10-frame displacement RMSE |
| --- | ---: | ---: | ---: | ---: |
| Train-axis fixed | 2.58063 | 2.48535 | 0.18258 | 0.99919 |
| GRU96 train-axis Cap4 | 2.08302 | 2.05902 | 0.18269 | 1.05476 |
| GRU106 train-axis Cap4 | 2.10895 | 2.08656 | 0.18357 | 1.06937 |
| Mamba train-axis Cap4 | 2.15383 | 2.10639 | 0.17533 | 1.02140 |

Factual comparisons only: GRU96-Cap4 lowest 3D/2D position RMSE;
Mamba-Cap4 lowest 1-frame displacement RMSE; train-axis fixed
lowest 10-frame displacement RMSE. Do NOT claim Mamba has best
overall state-estimation accuracy. Do NOT relabel displacement
metrics as full SE(3) pose RPE.

**Development-selected algorithm settings locked here:**

- Frozen 0027 TRAIN-only per-axis RMS and five-bin oracle-conditioned
  mapping from `run_gps_covariance_pilot.py`.
- Only Cap4, fixed exactly at 4x max per-axis TRAIN-fixed precision.
- Input features from degraded GPS only, 64-frame causal history;
  GRU and Mamba model/checkpoint as already trained, seed43 initially.
- h0 head only for GPS covariance; forecast heads h1/h2 are NOT used
  in the V7 four-sensor FGO and cannot support a predictive FGO claim.
- Other three sensor covariance models fixed as in V7 legacy pilot.
- All comparisons are offline batch FGO; no prospective iSAM2 claim.
- Multi-start solution selected by each graph's own objective, never
  reference RMSE. Common starts include explored Fixed and Mamba
  pilot solutions; this initialization is post-hoc and does NOT
  transfer automatically to an unseen sequence.
- Do not experiment with Cap2/Cap8, extra model checkpoints or
  alternative trajectory alignment based on 0027 VAL outcomes.

**Remaining independent evaluation gates (NOT passed):**

1. Confirm the reserved raw 0016/0020 drive directories and all
   required LiDAR, stereo camera, GPS and IMU files exist, without
   loading them or checking target metrics during readiness audit.
2. Adapt V7 saved model feature extraction/scaler, degradation protocol
   and original four-sensor factor generation to new sequences;
   `experiments/cross_sequence/` Stage7 builds different V5/V6
   inputs and is NOT an interchangeable V7 estimator pipeline.
3. For any new sequence, derive factor graph initial states only from
   available non-reference measurements under one frozen common
   initialization procedure (the 0027 saved Mamba/Fixed trajectories
   are NOT available for new drives). Record optimizer failure rate.
4. Freeze primary outcome: origin-aligned 3D position RMSE in m;
   secondary: origin-aligned 2D RMSE, displacement error over 1
   and 10 frames, stability and compute cost. Do not present position
   scores against OXTS GPS as physically independent absolute ATE.
5. Build a valid independent measurement-reference separation OR
   explicitly position the experiment as generalization to unseen
   drives of **synthetic GPS corruption** relative to original OXTS.
   Synthetic corruption reference independence is NOT physical
   positioning GT independence. Cross-time GPS-origin covariance
   remains unmodeled in this FGO.
6. After prerequisites, approve and document a ONE-TIME heldout
   release, no tuning or test-driven code revision afterwards.

This stage freezes settings to prevent further 0027 validation
cherry-picking. It does NOT assert actual data readiness, adequate
independent GT, statistically significant gains, or final paper quality.
Full source code/protocol implementation remains versioned under
Git commit a0056b95b15c53b5f4926cfa0a61e39a040918c0, with
Cap4 reference analysis saved in
`logs/isj_v7_cap4_multistart_seed43.log` on the user's server.



## Stage 14 — closed held-out inventory and causal inference adapter

User's **metadata-only**, no-content inventory confirmed the following:
- KITTI 2011_09_30_drive_0016_sync: 279 files in each of OXTS,
  Velodyne, left camera and right camera directories.
- KITTI 2011_09_30_drive_0020_sync: 1104 files in each of the
  same four sensor directories.
- Both dates are 2011_09_30, **different from development
  2011_10_03**, so neither reuse of that calibration directory nor
  the old Stage7 `DATE_DIR=2011_10_03` config is acceptable.

File counts do **not** prove valid timestamps, calibration, frame
alignment, generated factors, or independent trajectory truth. Both
held-out streams remain closed until implementation and release gates.

New reusable function in the *existing*
`experiments/isj_v7/train_gps_dev.py`:
`predict_frozen_gps_features_causally(model, features, train_median,
train_iqr, window=64, clip=20, batch_size=64)`.

- Takes a **degraded-GPS-only 21D feature stream** (same V7
  `src.multisensor_reliability.features.gps_features`),
  **training-only frozen median/IQR**, and an existing pretrained
  GRU/Mamba model; never reads target clean GPS, labels or GT.
- Returns full h0/h1/h2 predictions and Beta evidence proxies only at
  chronological anchor indices `63..n-1`; does not fabricate
  predictions for the cold-start first 63 frames.
- Uses window `[t-63..t]` with no future features; pure in-memory
  read-only processing with immutable model checkpoint and batch
  size independent predictions, enforced by new synthetic regression
  tests (future mutation, invalid scalers and cold-start policy).
- This is a **component**, not a held-out inference CLI, independent
  evaluation or approved final FGO. It has not read 0016/0020 contents.

Before the actual cross-sequence V7 release, remaining implementation
work includes: securely loading/validating frozen model checkpoints,
injecting synthetic GPS noise using an explicitly frozen recipe on
new drives, building same-definition four-sensor factors with each
drive's correct date-specific calibration, no 0027 saved-pose initial
states, a 63-frame cold-start policy shared by all methods, a frozen
optimizer budget that is feasible for these lengths, and an audited
test-once reference/evaluation protocol. Do not quietly substitute the
old Stage7 V5/V6 estimator, which has different training/measurement
provenance.

Next permissible operations: read only **filesystem metadata** for
`dataset/kitti/2011_09_30/calib_cam_to_cam.txt`,
`calib_imu_to_velo.txt`, and `calib_velo_to_cam.txt`, inspect
existing V7 training source code and run synthetic unit tests.
**No 0016/0020 sensor contents, ground truth or metrics** should be
inspected at this readiness stage.

## Stage 15 — frozen V7 checkpoint loading gate (no heldout content)

Existing V7 training checkpoints include PyTorch state_dict plus NumPy
train-prefix median/IQR scalers. With modern torch.load defaults,
these older NumPy-containing checkpoint files need explicit allowlisting
to remain compatible with weights_only=True. NEVER use
weights_only=False or arbitrary pickle fallback.

The original train_gps_dev.py now provides
`load_frozen_v7_gps_checkpoint(case,seed,device)` for
gru96_r, gru106_r, and mamba_r, with a strict allowlist.
It verifies the frozen training archive and its provenance hashes,
checkpoint/backend/model kwargs/parameter counts, saved
metric fingerprints and strict model tensor state. It compares the
stored normalization scalers bit-for-bit to the frozen train prefix.
It returns the model, scalers, and checkpoint/metric SHA256s with
no new-sequence data or reference loaded, and no training/writes.

The original checkpoints are development-only models; loading them
does not confer final-paper training approval or independent evidence.
Untrusted checkpoint sources must not be processed.

Read-only synthetic regression and actual saved-model loading gate:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
echo V7_TEST_EXIT=$?
python - <<'PY'
from experiments.isj_v7.train_gps_dev import (
    load_frozen_v7_gps_checkpoint,
)
for case in ('gru96_r', 'gru106_r', 'mamba_r'):
    model, train_median, train_iqr, provenance = (
        load_frozen_v7_gps_checkpoint(
            case, seed=43, device='auto',
        )
    )
    print(
        'FROZEN_MODEL_READY',
        case,
        'backend=', provenance['backend'],
        'device=', next(model.parameters()).device,
        'parameters=', provenance['parameters'],
        'checkpoint_sha256=', provenance['checkpoint_sha256'],
        'metrics_sha256=', provenance['metrics_sha256'],
        'data_sha256=', provenance['dataset_sha256'],
        'scaler_shape=', train_median.shape,
        'iqr_min=', float(train_iqr.min()),
        flush=True,
    )
print('FROZEN_MODEL_LOAD_GATE_COMPLETE; NO_HELDOUT_READ')
PY
~~~

The model artifact is not exported or rewritten. This tests real
locally saved checkpoint readability but does NOT read 0016/0020
measurements. If the saved Mamba checkpoint requires a different
installed CUDA/mamba_ssm runtime, report the error rather than
substituting GRU. Actual V7 four-sensor heldout data preparation
is still gated by date-specific calibration and a frozen degradation
and optimization protocol.

### Stage 15 compatibility patch — legacy run_family default

The initial frozen-load gate exposed a mismatch between the historical
GRU96/Mamba baseline metadata accepted by the already validated
get_development_run() and the new strict checkpoint loader. An absent
run_family is now interpreted as baseline ONLY for original GRU96/Mamba
base checkpoints. Explicit wrong family values still fail. GRU106
capacity checkpoints must explicitly declare capacity_gru_h106.
Known dataset SHA256, normalizer, train/validation provenance, expected
parameter count, network architecture and weights_only=True are NOT
relaxed. The error message now identifies *which metadata field names*
failed verification rather than emitting only an opaque ValueError.
Regression covers missing historical run_family in both metadata and
checkpoint, wrong explicit family, and useful field diagnostics.
No heldout 0016/0020 contents or test results were accessed.

## Stage 16 — in-memory full-sequence V7 factor/covariance bridge

Do not open reserved 0016/0020 yet. Following the successful server
gate (109 V7 tests pass and existing GRU96, GRU106, Mamba checkpoints
all load on CUDA with expected 114630/139290/139590 parameters), two
helpers were added to the existing run_gps_covariance_pilot.py:

1. rebase_frozen_v7_measurements(m) validates every four-sensor array,
   copies it, and rebases degraded GPS relative to the first degraded
   measurement. It reads neither OXTS clean reference nor GT; no write.
2. frozen_v7_sequence_gps_sigmas(degraded_gps_local, *, case,
   train_axis_fit, model, train_median, train_iqr, model_provenance)
   accepts arbitrary length >=64. Cases are locked to:
   axis_fixed, axis_gru96_r_cap4, axis_gru106_r_cap4, axis_mamba_r_cap4.
   It uses exactly the V7 canonical GPS features from the SAME
   degraded GPS stream (train protocol dt and rolling feature window),
   frozen model weights and saved TRAIN-only median/IQR, h0 prediction
   at anchors 63..n-1, frozen five-bin XYZ mapping and exactly Cap4.
   The first 63 frames use the identical TRAIN fixed-axis sigma for
   EVERY case; neither padding nor future-observation prediction.
   The h1/h2 forecasting heads are not used for GPS weighting.

These are pure in-memory **components**, not a held-out evaluation
runner or released benchmark. Date-specific factor generation,
frozen synthetic corruption realization, and one-time independent
evaluation protocol still require independent approval. The original
0027 680-anchor pilot path has not been changed.

Six tests (synthetic arrays only) check cold-start equality,
original interpolation/Cap4 equivalence, prefix causality, strict
case/model validation, non-mutating four-sensor rebasing and
GTSAM graph-only construction with all four factors; no output
folders written. Historical V7 tests must still pass.

### Next gate: 0027 DEV VAL parity, without opening TEST or heldout

Run this read-only equivalence check, using original saved 0027 DEV
GPS values only through frame 3859. This tests whether newly loaded
frozen model inference gives the same Cap4 sigmas as the *saved*
0027 validation predictions used in prior experiments, not whether
0016/0020 generalizes:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
echo V7_TEST_EXIT=$?
python - <<'PY'
import numpy as np
from experiments.isj_v7.train_gps_dev import (
    load_frozen_v7_gps_checkpoint,
)
from experiments.isj_v7.run_gps_covariance_pilot import (
    DEGRADED, load_train_axis_noise, frozen_v7_sequence_gps_sigmas,
    prepare_case,
)
from experiments.isj_v7.supervision_gate import read_policy, validate_policy

policy=read_policy(); validate_policy(policy)
fit=load_train_axis_noise(policy)
with np.load(DEGRADED / 'degraded_sensor_data.npz', allow_pickle=False) as z:
    gps=np.asarray(z['degraded_gps_local'][:3860], np.float64)
assert gps.shape == (3860,3)
for case, backend in (
    ('axis_gru96_r_cap4','gru96_r'),
    ('axis_gru106_r_cap4','gru106_r'),
    ('axis_mamba_r_cap4','mamba_r'),
):
    model, median, iqr, info=load_frozen_v7_gps_checkpoint(
        backend, seed=43, device='auto')
    new, meta=frozen_v7_sequence_gps_sigmas(
        gps, case=case, train_axis_fit=fit, model=model,
        train_median=median, train_iqr=iqr, model_provenance=info,
    )
    old,_=prepare_case(case,43)
    assert new.shape == (3860,3) and old.shape == (680,3)
    gap=float(np.max(np.abs(new[3180:3860]-old)))
    ok=gap <= 1e-3
    print('STREAM_PARITY',case,'max_sigma_abs_diff_m=',gap,
          'pass=',ok,flush=True)
    if not ok: raise RuntimeError('Frozen stream did not reproduce DEV')
print('V7_FROZEN_STREAM_PARITY_OK; DEV_ONLY; NO_HELDOUT')
PY
~~~

Note that this parity audit must not load 0027 internal TEST frames
3862 onward, nor 0016/0020. If any parity mismatch occurs,
do not proceed to heldout; investigate sensor realization,
feature scaling/window alignment, checkpoint or inference runtime.

## Stage 17 — sealed one-time 0016 + 0020 pipeline (PRE-RELEASE)

**STATUS: NOT RELEASED.** The new durable entrypoint
`experiments/isj_v7/cross_sequence_v7.py` combines the four-sensor
sensor/factor producer, frozen saved GRU/Mamba inference, shared
LiDAR-only initial pose, GTSAM LM solve and a delayed synthetic
OXTS-relative evaluation for BOTH reserved drives. Unlike the older
Stage7 pipeline, it imports the exact original 0027 four-sensor
`corrupt_local_gps`, `corrupt_imu`, `corrupt_lidar`, `corrupt_camera`,
the original `LidarOdometry`/`StereoVisualOdometry` hyperparameters,
and the original `BodyFrameConverter` factor format. All original
frozen V7 training inputs, 4x information cap, checkpoint SHA256
verification and train-only noise mapping remain unchanged.

### Predeclared heldout methodology (never tuned on reserved sequences)

- Two drives RELEASED TOGETHER: 2011_09_30_drive_0016_sync
  (279 frames) and 2011_09_30_drive_0020_sync (1104 frames).
- Date-specific KITTI 2011_09_30 stereo and sensor extrinsics;
  all four stream filename indices and modality timestamps checked.
- Frozen 0027 degradation plan resampled once in normalized frame
  time for each drive; same 20260826 corruption streams as original
  0027, and same generated measurements for all four methods.
- Four frozen methods: TRAIN Axis-Fixed, GRU96-Cap4,
  GRU106-Cap4 (capacity matched), Mamba-Cap4 (seed 43).
- Model window 64; first 63 GPS factors use common fixed-axis
  noise; later factors use h0 only (h1/h2 NOT used); all cases
  share exactly the same GPS, IMU, LiDAR and Camera factors,
  except for the GPS covariance weighting.
- Common deterministic LiDAR-integrated graph initial values;
  initial pose identity with origin-relative corrupted GPS.
  No saved dev Fixed/Mamba solution, no use of test reference
  to initialize, select or stop optimization.
- At most 12 LM passes (100 iterations each), relative objective
  reduction <=1e-6 AND translation shift RMSE <=0.01m;
  flag failures and instability, do NOT remove failed drives.
- One initialization only is predeclared on new sequences.
  Dev three-start best-basin results are NOT directly comparable.
- Primary outcome: origin-relative 3D RMSE versus ORIGINAL
  UNCORRUPTED OXTS GPS, secondary: origin-relative 2D,
  1-frame and 10-frame displacement RMSE, convergence and
  wall-clock solver time. Results are TESTED synthetic GPS
  corruption recovery, NOT independent physical state GT.
- Only after BOTH sequences have finished ALL four solves
  will the benchmark runner open the saved clean OXTS-derived
  reference for metric computation. Outputs written under
  `results/isj_v7/heldout_cap4_v1/`, refuse overwrite/release
  if folder exists. An interrupted release remains marked
  `RUN_INCOMPLETE`: don't silently delete and rerun.
- No arbitrary user case/sequence/Cap threshold/seed controls
  in the release interface. Future papers must disclose the
  post-hoc 0027 method selection and reused-source evaluation.

### Test the program WITHOUT opening the reserved sensor streams

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
echo V7_TEST_EXIT=$?
python -m experiments.isj_v7.cross_sequence_v7 --preflight
echo V7_CROSS_PREFLIGHT_EXIT=$?
~~~

`--preflight` reads only directory entries, file sizes and flags.
It does not read OXTS, KITTI timestamps, .bin, camera images or
the 0016/0020 target trajectories, and never constructs any
factor graph or output folder. New synthetic regression tests
cover metadata-only behavior, timestamps, names, one-time release
token gate, four frozen cases, diagnostics and a GTSAM graph solve.

The **actual** heldout invocation is intentionally NOT given as
the next step: first review tests and metadata-only preflight;
then approve opening both reserved drives ONCE under the
already version-controlled frozen protocol. This file/code commit
must remain frozen before any test metric is inspected.

## Stage 18 — 0020 early-timestamp forensic incident and audited recovery

**THE RESERVED TEST WAS RELEASED AND INTERRUPTED. Do not describe it as
unopened/pristine or as an unchanged predeclared evaluation.**
Initial one-time run was from code commit 2bfeb6f on 2026-11-20.
0016 finished its entire LiDAR/stereo data build (279 frames).
0020 failed after reading timestamps but BEFORE reading OXTS positions,
LiDAR point clouds or camera pixels, and BEFORE any solve
because 0020/velodyne exceeded the original 100ms timestamp gate.
The archive under results/isj_v7/heldout_cap4_v1/ contains
RUN_INCOMPLETE and 0016's four sensor/factor archives and manifest;
0020 has no build manifest or folder, and heldout_scores.json is absent.

Read-only timestamp diagnostic *after release* found:
- 0016: 279/279 frames, all 3 sensor streams strictly increasing and
  no absolute OXTS offset >100ms (max 11.694ms).
- 0020: 1104/1104 frames, all 3 sensor streams strictly increasing;
  ALL exceeded 100ms at the SAME frame IDs [0, 1, 2] only.
  Maximum absolute deltas: Velodyne .123113s, camera02 .124902s,
  camera03 .125527s; p95 absolute deltas 9.021, 10.814, 11.371ms.
- Zero index shift had the lowest median absolute difference across
  all modalities; there is no global one-frame offset. This does NOT
  prove the three opening frames are physically synchronized.
- Only the timestamp audit was performed; no test metrics were read
  or computed and no GRU/Mamba inference or FGO optimization ran.

**Explicit POST-RELEASE engineering exception** in
validate_frozen_time_alignment(): for 0020 ONLY, allow exactly
frames [0, 1, 2] to have a shared negative [-150,-100) ms offset
against OXTS *in all three modalities*. Their per-frame cross-modal
spread must be <=10ms. All other frames and all of 0016 retain
the existing <=100ms tolerance, strict monotonicity and frame IDs.
No frame deletion, index realignment, timestamp rewrite, covariance
retuning, model retraining, or GRU/Mamba/Cap4 optimizer changes.
This threshold is a disclosed **post-hoc data compatibility fix**
derived from observed heldout timestamps, not a train-only policy.

`cross_sequence_v7.py` now provides
`--resume-0020-startup3 --release-token AUDITED_V7_0020_STARTUP3`.
It ONLY resumes from the existing original RUN_INCOMPLETE marker,
checks preserved 0016 build provenance and all five required files,
refuses an existing 0020 folder, score file or missing marker,
and builds **0020 only**. It does not overwrite/recompute 0016.
After both builds, it uses the SAME frozen four cases, covariance
and shared LiDAR initial trajectory as the initial one-time protocol.
Only after all eight solves does it open OXTS-relative reference
and write evaluation scores. The report explicitly flags an
interrupted first release and the timestamp policy exception.

Reproducibility and citation caveat: 0020 has 3 initial frames whose
inter-sensor time offsets exceed the original allowable maximum;
these remain inside the graph and should be disclosed in the paper.
An untouched independent physical benchmark is NOT established.

**Required gate BEFORE invoking recovery:**

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
echo V7_TEST_EXIT=$?
python - <<'PY'
from pathlib import Path
from experiments.isj_v7.cross_sequence_v7 import DEST
print('RUN_INCOMPLETE=', (DEST/'RUN_INCOMPLETE').is_file())
print('FIRST_BUILD=', (DEST/'0016'/'build_manifest.json').is_file())
print('0020_ABSENT=', not (DEST/'0020').exists())
print('SCORES_ABSENT=', not (DEST/'heldout_scores.json').exists())
PY
~~~

Only when the 129 synthetic/unit tests and the four status indicators
pass, the audited recovery command can be executed exactly ONCE.
Do not delete or overwrite 0016, nor run the old `--release-all`.
