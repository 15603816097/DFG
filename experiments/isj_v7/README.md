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
