# ISJ-V7 Stage 0: temporal protocol preflight (0027 only)

This is **not** a new predictive model and **not** permission to train on V3 proxy labels. It is the first *read-only* protocol milestone: deterministic target chronology, split-safe anchor indices, train-prefix-only robust normalization, raw-data SHA256 and sensor-array fingerprints. There is no checkpoint modification, FGO, GPU workload, or contact with reserved KITTI 0016/0020.

## Known V5/V6 issue

V3's only stored future label is `current[t+3]` with padded tail. V5 and V6 treated that column as head 1, then used `current[t+2]` for head 2. Thus their heads are `[t,t+3,t+2]`, not `[t,t+1,t+2]`. Earlier split construction also allowed a few future-label targets to cross train/val boundaries. The fixed preflight **ignores the V3 future column** and constructs chronological heads directly from `current`.

## Proposed V7 timing specification

- `window=64`: features for time `t-63..t`, never future features.
- `horizon=3` means **three heads**, offsets `[0,1,2]` (now plus two future frames). Future *three* steps would require **four** heads `[0,1,2,3]`.
- For 4544 frames, split raw timestamps `[0,3180)`, `[3180,3862)`, `[3862,4544)` by fixed 70% and 85% cutoffs; no separate model-dependent split.
- Allowed anchor ranges for `horizon=3`: TRAIN `[63,3178)` (3115 anchors), VAL `[3180,3860)` (680 anchors), internal TEST `[3862,4542)` (680 anchors). End-exclusive notation.
- Last two anchors of each split are excluded, as their targets would enter the next split; test's final two frames are also omitted. Validation input window may read earlier **feature** frames as historical context, but never future features, and no target may cross a split boundary.
- GRU and Mamba should both use **the same** robust scaler (median/IQR), fit only on TRAIN prefix `features[:3180]`. This preflight only computes fingerprints, not transformed tensors.

## Do not misinterpret the successful audit

- **V3 current GPS reliability** uses clean-vs-degraded GPS oracle data. The GPS oracle label may be used as *supervised* truth in a development simulation, not as an online feature, but its generation and intended evaluation target must be explicit.
- **V3 current IMU/LiDAR/Camera reliability** uses a whole-trajectory fixed-FGO estimate as offline reference. It can encode future measurements, violating prospective causal/independent supervision assumptions even when V7 `t` indexing and split boundaries are corrected.
- Original 0027 features may also contain full-trajectory or offline preprocessing. This preflight has **not** audited feature causality or the independence of measurement/reference sources.
- 0027 is heavily explored. Its within-drive TEST is **not** an external untouched test. Reserve 0020 and 0016. Do not select methods or thresholds using those drives.
- Therefore `approved_for_final_training=false` is deliberately hardcoded for this V3 archive. **Stop at preflight**, develop independent/causally defensible labels and verify feature provenance before launching an actual V7 train/test pipeline.

## Server commands

```bash
cd /root/projects/DFG
conda activate dfg-isj
# Local edit to the older provenance test is preserved by fast-forward pull.
git pull --ff-only
python -m pytest tests/test_isj_v7_protocol.py -v
echo "V7_PREFLIGHT_TEST_EXIT=$?"
# Only after all six tests pass. Read-only, no writing:
python -m experiments.isj_v7.protocol > logs/isj_v7_preflight_0027.log
rc=$?
echo "V7_PREFLIGHT_EXIT=$rc"
cat logs/isj_v7_preflight_0027.log
```

You may separately record the JSON report in an experiment registry **after** reviewing it, along with the exact Git commit and approved source provenance. No learning claims follow from these audit fingerprints alone.


## Stage 1: V3 input-feature / label-measurement consistency audit

Important **static source-code discrepancy**: The V2 corrected feature builder calls
`corrupt_gps()` with RNG stream offset `9176`, while
the degraded four-sensor measurement builder calls `corrupt_local_gps()`
with offset `31`. Both may use the same severity schedule but
generally draw different GPS noise values. The V3 label builder imports GPS/IMU
*features* from V2 and computes GPS *labels* using the separately generated
degraded measurement archive. **No numeric assertion is made until the saved
0027 arrays are actually audited.**

The new *read-only* `audit_feature_label_streams.py` computes:

1. `gps_features(V3 label-degraded_gps_local)` and compares to
   frozen `V3 gps_features` using the original configuration.
2. `imu_features(V3 label-degraded_acc, degraded_gyro)`
   and compares to frozen `V3 imu_features`.
3. Recomputes `||degraded GPS - clean GPS||` and compares against
   `V3 diagnostic_gps_error_m` to independently confirm
   which physical GPS realization supervised the legacy target.
4. If available, compares V2 `corrected_feature_debug.npz`
   GPS corruption array directly against the label-degraded GPS
   and against V3's archived GPS feature representation.
5. Prefix-invariance tests for GPS/IMU feature functions check that altering
   measurements only **after** frame 79 cannot change features through
   frame 79. This is limited to these pure Python feature functions and
   does not establish the causality of all upstream data preparation.

This audit does **not** read raw KITTI 0020/0016, load any neural network,
perform optimization, or rewrite V3/V5/V6 artifacts. All saved measurements
and labels must remain immutable.

```bash
cd /root/projects/DFG
conda activate dfg-isj
set +e; set +u; set +o pipefail
git status --short
git pull --ff-only
python -m pytest \
    tests/test_isj_v7_protocol.py \
    tests/test_isj_v7_feature_label_streams.py \
    tests/test_isj_v7_feature_prefix_causality.py -v
echo "V7_LINEAGE_TEST_EXIT=$?"
# ONLY if all tests pass:
mkdir -p logs
python -m experiments.isj_v7.audit_feature_label_streams \
    > logs/isj_v7_feature_label_streams_0027.log 2>&1
echo "V7_LINEAGE_AUDIT_EXIT=$?"
cat logs/isj_v7_feature_label_streams_0027.log
```

**Interpretation**: `gps_input_matches_gps_label_measurement=false`
is a factual incompatibility of the *saved* GPS input and the corresponding
oracle-target measurement stream; it is NOT an indicator of predictive
generalization. Even with numerical match, the whole-trajectory FGO proxy
label and reuse of 0027 remain independent unresolved limitations.
No training on these V3 labels is approved.


## Stage 1 confirmed result on saved 0027 data (2026-11-19)

User verified all 13 protocol/stream/prefix tests PASSED. Read-only real
0027 audit exited zero. **Actual GPS input-feature vs label-measurement
mismatch CONFIRMED**:

| comparison | mismatched frames | mismatched values | max absolute delta |
|---|---:|---:|---:|
| frozen V3 GPS 21D features vs rebuilding from V3 label's degraded GPS | 2791/4544 | 57071 | 16063.6284 in feature units |
| V2 debug corrupted GPS 3D vs V3 label degraded GPS 3D | 2659/4544 | 7973 | 55.3394 m |
| frozen V3 GPS features vs rebuilding from V2 debug GPS | 0 | 0 | 0 |
| frozen V3 IMU features vs V3 label degraded IMU | 0 | 0 | 0 |
| V3 GPS oracle-error diagnostic vs computed degraded-minus-clean GPS | 0 | 0 | 0 |

The larger mismatched count in 21D features than in 3D inputs is
consistent with past-window difference/statistic propagation; it is
not a localization ATE. **Do not present historical V5/V6 GPS
reliability or conditional uncertainty gains as a validated deployable
fault detector.**

### Next narrow gate: read-only canonical GPS *dry-run*

`audit_gps_canonical_dryrun.py` reconstructs one GPS feature
candidate solely from **the exact** `degraded_gps_local` array used by
the original V3 oracle labels. It separately verifies GPS `error_m`
and `exp(-(error / 5m)^2)`-with-clipping against the saved V3
diagnostic and current reliability labels. It defines the original 0027
frame split and train-only robust scaler and hashes candidate features
and split-purged labels **in memory only**. No file is written by the
module; in particular, the original V3 NPZ and V5/V6 weights remain
unchanged. This audit does not validate independent physical GPS
truth; clean OXTS GPS is used only as the existing synthetic supervision
oracle, never the model input.

```bash
cd /root/projects/DFG
conda activate dfg-isj
set +e; set +u; set +o pipefail
git pull --ff-only
python -m pytest \
    tests/test_isj_v7_protocol.py \
    tests/test_isj_v7_feature_label_streams.py \
    tests/test_isj_v7_feature_prefix_causality.py \
    tests/test_isj_v7_gps_canonical_dryrun.py -v
code=$?
echo "GPS_CANONICAL_TEST_EXIT=$code"
if [ "$code" -eq 0 ]; then
  mkdir -p logs
  python -u -m experiments.isj_v7.audit_gps_canonical_dryrun \
      > logs/isj_v7_gps_canonical_dryrun_0027.log 2>&1
  echo "GPS_CANONICAL_AUDIT_EXIT=$?"
  cat logs/isj_v7_gps_canonical_dryrun_0027.log
fi
```

Even if this dry-run confirms the exact synthetic oracle mapping, it
will still return `approved_for_final_training=false`. Before training
a final paper-ready model, freeze label policy, ensure consistent
measurement streams for all modalities (especially camera/physical
LiDAR/FGO), avoid lookahead/entangled trajectory supervision, and set
independent cross-drive validation. Preserve reserved 0016/0020 until
then. Do not modify V3/V5/V6 files to make old experiments appear fixed.


## Stage 2: archived Camera / physical LiDAR factor-label lineage (read-only)

The GPS canonical dry-run was verified on 2026-11-19: all **18 tests
PASSED**, audit exit 0. Rebuilding 21D GPS features from the exact
degraded GPS measurement stream underlying the original oracle labels
produced SHA256 `e1919787f2a812d793c7ae673e562f9c09104510310d446b178d48b42525e7ae`.
V3 oracle GPS errors and mapped labels matched precisely, with clean
GPS used only as an offline supervision reference. This does **not**
erase the 2791/4544 old-feature mismatch or approve independent training.

Next, `audit_camera_lidar_factor_lineage.py` tests whether
the frozen V3 Camera/LiDAR supervision truly matches the **archived**
degraded stereo-VO between-factors and physical degraded LiDAR
between-factors, respectively, against the exact saved
`results/four_sensor_fixed_fg/poses.npy` reference used by the
original V3 label builder. It recomputes each target-aligned
measurement-vs-fixed-FGO translation/rotation error, invalid-factor
`+inf` diagnostic and minimum reliability, and the Gaussian
error-to-reliability mapping. It hashes all four source files and does
not rewrite data, run FGO, load raw KITTI or infer with networks.
It is a **cross-artifact consistency check reusing the historical
mapping definitions**, NOT an independent external validation.

This check deliberately leaves
`feature_input_vs_factor_raw_measurement_verified=false`.
V2 Camera features are computed from a degraded *single left frame*,
while the archived camera factors use degraded *stereo pairs plus next
left frame*; V2 LiDAR features use physically degraded single scans
while archived LiDAR factors are odometry computed on successive
degraded scans. Same original code family is not proof the exact
archived image/point-cloud realization was used.

**Most importantly** the fixed-FGO reference uses an *entire* 0027
trajectory optimized with sensor data, and so is NOT an external
independent oracle. Even perfect factor-label matches do not authorize
paper-ready V7 training on these Camera/IMU/LiDAR labels.

```bash
cd /root/projects/DFG
conda activate dfg-isj
set +e
set +u
set +o pipefail
git pull --ff-only

python -m pytest \
  tests/test_isj_v7_protocol.py \
  tests/test_isj_v7_feature_label_streams.py \
  tests/test_isj_v7_feature_prefix_causality.py \
  tests/test_isj_v7_gps_canonical_dryrun.py \
  tests/test_isj_v7_camera_lidar_factor_lineage.py -v
V7_MODALITY_TEST_EXIT=$?
echo "V7_MODALITY_TEST_EXIT=$V7_MODALITY_TEST_EXIT"

if [ "$V7_MODALITY_TEST_EXIT" -eq 0 ]; then
  mkdir -p logs
  python -u -m experiments.isj_v7.audit_camera_lidar_factor_lineage \
    > logs/isj_v7_camera_lidar_factor_lineage_0027.log 2>&1
  echo "V7_MODALITY_AUDIT_EXIT=$?"
  cat logs/isj_v7_camera_lidar_factor_lineage_0027.log
fi
```

Next research gate should be **method-first independent supervision
policy and evaluation protocol**, not more opportunistic tuning on
within-drive 0027 "TEST". Maintain the held-out 0016/0020 embargo.


## Stage 3 — supervised-label policy and preregistration guard

The saved 0027 Camera/LiDAR proxy-label audit succeeded on the server
(**23/23 tests passed**, exit 0). All 4543 LiDAR factors are marked
valid; Camera has 4461 valid and 82 invalid. Both modalities' archived
factor errors and reliability labels match their original saved
factor measurements and full-trajectory fixed-FGO reference, but that
reference is not independent ground truth. Exact original image/point-cloud
input-to-factor provenance remains unverified.

Review `SUPERVISION_AND_EVALUATION_POLICY.md` before any V7 training
design. `supervision_policy_v1.json` is an immutable-status
development policy tracking 0027 lineage and SHA256 across five source
artifacts. `supervision_gate.py` validates this policy,
can read-only hash the known files, and **refuses** to authorize final
training or independent-generalization claims. It never inspects 0016/0020
nor writes data.

```bash
cd /root/projects/DFG
conda activate dfg-isj
git pull --ff-only
python -m pytest tests/test_isj_v7_supervision_gate.py -v
python -m experiments.isj_v7.supervision_gate --verify-sources
# Deliberate refusal test (expected exit=2):
python -m experiments.isj_v7.supervision_gate --require-final-training
```

No V7 model was retrained by this stage. The next actual engineering
work is **new, same-realization synthetic supervision plus causal input
availability and a frozen independent evaluation plan**; do not
relabel old full-trajectory FGO proxies as independent truth.

## Stage 4: same-realization GPS development dataset (train + validation only)

Stage-3 tests: 33 passed and five frozen source SHA256s verified on 0027.
Final-paper training remains BLOCKED. This Stage-4 tool produces only a
GPS synthetic-injection **development** candidate from the exact
degraded GPS measurement stream used by the historical V3 labels.

Read-only default: `python -m experiments.isj_v7.build_dev_gps_candidate`.
Explicit export: `python -m experiments.isj_v7.build_dev_gps_candidate --write-dev-candidate`.

Source policy is validated and the frozen V3/degraded source SHA256 hashes
are checked before any export. The exporter first reconstructs GPS features
and verifies the synthetic labels via the prior canonical dry-run.
Neither clean GPS nor corruption severity or mode is ever saved as a model input.

Destination: `results/isj_v7/dev_gps_0027_same_source_v1/`.
New files are `gps_train_val.npz` and `provenance.json`. Never overwrite
the original V3, V5, V6 artifacts. Existing target directories are refused.

Data arrays for 4544-frame 0027:
- `gps_features`: float32 `(3862,21)`, ONLY train+val time prefix.
- `train_anchors`: int64 `(3115,)`, anchor t=63..3177.
- `train_targets`: float32 `(3115,3)`, reliability at t,t+1,t+2.
- `val_anchors`: int64 `(680,)`, anchor t=3180..3859.
- `val_targets`: float32 `(680,3)`, targets within VAL through t=3861.
- `train_median`, `train_iqr`: float32 `(21,)`, fit only from features[:3180].

The internal 0027 TEST range `[3862,4544)` is completely omitted from
the exported feature/target archive. Training-target values are only
synthetic injected-GPS reliability relative to original OXTS/GPS;
they are not independent physical ground truth. Clean GPS and other
oracle reference arrays are excluded from saved training arrays.

Run on server after pulling:

```bash
cd /root/projects/DFG
conda activate dfg-isj
set +e; set +u; set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -v
V7_TEST_EXIT=$?; echo V7_TEST_EXIT=$V7_TEST_EXIT
if [ "$V7_TEST_EXIT" -eq 0 ]; then
  mkdir -p logs
  python -u -m experiments.isj_v7.build_dev_gps_candidate \
    > logs/isj_v7_dev_gps_preview.log 2>&1
  V7_PREVIEW_EXIT=$?; echo V7_PREVIEW_EXIT=$V7_PREVIEW_EXIT
  cat logs/isj_v7_dev_gps_preview.log
  if [ "$V7_PREVIEW_EXIT" -eq 0 ]; then
    python -u -m experiments.isj_v7.build_dev_gps_candidate \
      --write-dev-candidate
    echo V7_EXPORT_EXIT=$?
  fi
fi
```

Do not treat this step as final training approval or touch 0016/0020.
Multi-modal, time-causal, independent evaluation design remains open.

## Stage 5 — first V7 GPS-only GRU/Mamba DEVELOPMENT training

See `DEV_GPS_TRAINING.md` for verified protocol, model definitions,
read-only smoke tests, optional CUDA/Mamba environment checks, and
complete training commands. New files:

- `model_dev_gps.py`: explicit GRU/Mamba backbones and common Beta evidential head.
- `train_gps_dev.py`: dev-only, SHA256-checked train+validation training and metrics.
- `tests/test_isj_v7_gps_training.py`: input/target integrity, model/backprop, Mamba fail-closed and evaluation checks.

This stage explicitly trains **only on the GPS synthetic-injection
development candidate**. It does not run any independent 0016/0020
test, FGO benchmark, or clean-GPS input. The Stage-3 final-training
release gate remains closed.
