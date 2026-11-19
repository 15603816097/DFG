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
