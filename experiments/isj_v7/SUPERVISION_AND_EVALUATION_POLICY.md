# ISJ-V7 — Stage 3 supervision policy and prospective evaluation protocol

Status: **DEVELOPMENT-ONLY / DRAFT METHOD PROTOCOL**. This document records current evidence and proposed future design; it does **not** assert that final training, paper results, or independent held-out evaluation are complete.

## 1. Verified facts on development KITTI 2011_10_03_drive_0027_sync

- V7 chronology: input feature window `[t-63,t]`; 3 supervised heads at `[t,t+1,t+2]`. For 4544 frames, train anchors 63..3177 (3115), validation 3180..3859 (680), exploratory internal test 3862..4541 (680). Each partition's future labels remain inside its own partition. Robust feature scaling must be fit on train prefix only.
- Original V3 GPS features **differ** from features of GPS measurements used for its labels on 2791 / 4544 frames; V2-debug original GPS stream differs from V3-label GPS stream on 2659 frames. This is a confirmed same-realization mismatch, not a model failure-rate statistic.
- Rebuilding GPS features from V3 *label-generation* degraded GPS in memory yields features with SHA256 `e1919787f2a812d793c7ae673e562f9c09104510310d446b178d48b42525e7ae`; re-derived V3 oracle GPS labels and error diagnostics match exactly. No corrected archive has yet been materialized.
- V3 IMU features match the corresponding degraded IMU measurements. However, the original IMU **labels** were computed using an entire-trajectory optimized pose reference.
- V3 archived Camera/LiDAR factor-to-proxy-label mappings match their stored factor measurements and full-trajectory FGO reference. Physical LiDAR had 4543/4543 marked valid pairs; Camera had 4461/4543 valid and 82 invalid. Historical label matching **does not** establish that the original per-scan / per-image model inputs used the same measurement realizations as the factor estimators.
- The earlier V5 GRU and V6 Mamba models have horizon-order mismatch `[t,t+3,t+2]` and split-boundary contamination; old results remain **exploratory**, not corrected baselines.
- Neither the 0027 within-drive TEST segment nor full-trajectory FGO reference constitutes unseen independent truth. **Do not use 0016 or 0020** for model development or tune thresholds on them.

## 2. Separate three notions of "reliability"

| Construct | Permissible source / meaning | Can be fed to an online estimator? |
|---|---|---|
| **Synthetic injection supervision** | Recorded corruption realization, clean-vs-degraded injected displacement, dropout / attack indicator, or experiment-generated severity for the *exact* measurement used. GPS canonical candidate has been verified in memory. | No: clean/oracle reference and corruption schedule are **targets/diagnostics only**, never inputs. A model prediction from causal observations may be used. |
| **Posthoc relative-factor error proxy** | Error of Camera/LiDAR/IMU factor relative to the **fixed full-trajectory FGO** poses. Not independent of sensors or future measurements. | No: cannot enter online features, covariances, model selection or independent evaluation. Historical descriptions only pending replacement. |
| **Causal online reliability evidence** | Pre-update sensor innovations and their uncertainty; ICP fitness / correspondence quality; stereo inlier/reprojection quality, or motion consistency available at the decision time. | Yes only after availability-time audit, with an explicitly causal update order and no use of measurement's own posterior to certify itself. |

### Proposed model direction (not yet implemented or validated)

Use the **same** causally available inputs and labels for GRU versus Mamba, compare reliability-only predictions and evidential uncertainty, then test bounded, predeclared noise mapping into an incremental factor graph. Current estimate at time `t` may use data observed up to `t`; forecasting `t+1,t+2` must use no data beyond `t`. Do not silently conflate "three output heads" with "three future steps". Keep prediction, pre-update feedback and optimization posterior separate in code to avoid circular residual gates.

Potential study components:

1. **Static noise** and **robust-kernel factor graph** as nonlearning baselines.
2. **Causal residual-only** adaptive weights without recurrent learning.
3. **GRU reliability-only** and **Mamba reliability-only** under identical model-selection and input policies.
4. **Mamba reliability + uncertainty**; quantify incremental value beyond reliability itself with proper scoring, calibration and temporal dependence-aware uncertainty intervals.
5. **Feedback/forecast ablations**: without forecast, without uncertainty, without pre-update feedback, static covariance. Add factor-memory/lifetime only if empirical evidence supports it.

These are **planned comparisons**, not completed results or proven innovations.

## 3. Requirements before a paper-ready train/evaluate command exists

- Select a coherent sensor measurement stream for each modality, including precise frame-to-factor timestamp alignment, failure handling and a causal observation-availability map.
- Generate or collect **source-independent enough** supervisory references for the claim being tested; if only synthetic injections are available, restrict conclusions to **synthetic corruption diagnosis**. In particular, original KITTI OXTS/GPS should not be portrayed as independent GPS ground truth for the same GPS sensor.
- Audit GPS, IMU, Camera and LiDAR target independence individually. Do not recycle full-trajectory FGO optimized pose labels across train/validation boundaries and claim strict split independence.
- Freeze model architectures, seeds, optimizer budgets, loss weights, labeling protocol and source hashes. Use matched train/val indices and a single fixed normalization protocol. Record the exact Git commit in checkpoints.
- Declare **primary outcome(s)** (e.g. 3D ATE RMSE and RPE at 1 and 10 frames), secondary outcomes (failure rate, availability, uncertainty calibration, runtime), trajectory alignment and exclusion rules before opening an independent test drive.
- Report results by **independent drive/scenario**; do not treat correlated per-frame errors as independent replicates. Require an improvement supported across methods/metrics, not a single favorable ATE.
- Keep 0020 and 0016 closed while method selection is in progress. An independent test is run only **once after** the protocol and baseline settings are sealed; no retrospective hyperparameter tuning on it.
- Since fixed FGO and OXTS are not automatically independent GT, validate evaluation reference independence or explicitly qualify all state-estimation metrics as reference-relative / synthetic.
- Stage-3 manifest sets `final_training_approved=false`; any final training program must reject it. **Do not** flip this flag in the current policy version; produce a reviewed new manifest/version after the above gates actually hold.

## 4. Read-only policy-gate commands

```bash
cd /root/projects/DFG
conda activate dfg-isj
python -m pytest tests/test_isj_v7_supervision_gate.py -v
python -m experiments.isj_v7.supervision_gate --verify-sources
# Should be deliberately blocked; use only to verify the safety gate:
python -m experiments.isj_v7.supervision_gate --require-final-training
# Expected exit code 2: final training is NOT approved.
```

The `--verify-sources` command hashes only the five **known development 0027 artifacts**, never searches the entire dataset or touches reserved drives. It creates no data files. Do not add `--require-final-training` to any workflow expecting success.

## 5. Next engineering milestone

Build *separate, source-traceable* candidate datasets for GPS synthetic injection and future IMU/Camera/LiDAR synthetic conditions; do **not** overwrite historical artifacts. Design Camera/LiDAR exact-observation (scan/image) vs factor-odometry checks, define the true decision timestamps, and implement a one-way causal interface (`observations -> predictor -> pre-update weight -> factor graph`). Test this interface with synthetic inputs before retraining large models.

The current Stage-3 gate is a research-integrity guard, not proof that V7 is scientifically final.
