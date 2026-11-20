# V7 cross-sequence synthetic-degradation results (0016, 0020)

**Status: COMPLETE with a disclosed POST-RELEASE synchronization exception.**
Values are transcribed from the user's terminal output after audited recovery.
Canonical on-server evidence: results/isj_v7/heldout_cap4_v1/heldout_scores.json,
logs/isj_v7_heldout_0016_0020.log and
logs/isj_v7_heldout_recovery_0020.log.
Persisted report/artefact SHA256 inventory should be checked separately.

## Run and protocol lineage

Initial reserved release: commit 2bfeb6f482418f488f384a200554f6a43cc7f60c.
Audited recovery: commit 71cf91a259ce967cc1ae9a39f8ef64cd051c487b.
Original release built all 0016 measurements but interrupted on
0020's first three OXTS-to-LiDAR/camera timestamp offsets. Recovery
preserved original 0016 files, rebuilt only 0020, then solved all
eight graphs and evaluated AFTER all solves. The post-release
exception permits a common [-150,-100) ms negative offset on
exactly 0020 frame indices [0,1,2] for three modalities; all
other frames keep the <=100 ms bound, no frames reindexed/deleted.
DO NOT call this a completely untouched/preregistered benchmark.
Before recovery: 129 V7 tests passed; recovery exit code 0.

## Main 3D position result (origin-relative OXTS/GPS RMSE, m, lower better)

| Method | 0016, 279 frames | 0020, 1104 frames | reduction vs fixed 0016 | reduction vs fixed 0020 |
|---|---:|---:|---:|---:|
| Axis-Fixed | 2.652556 | 21.015816 | — | — |
| GRU96-Cap4 | 2.473211 | **3.203241** | 6.76% | 84.76% |
| GRU106-Cap4 | 2.632897 | 4.211687 | 0.74% | 79.96% |
| Mamba-Cap4 | **2.446713** | 6.984370 | 7.76% | 66.77% |

Mamba beats GRU96 by just **0.026499 m (1.07%)** on 0016.
On 0020 Mamba's error is **3.781129 m (118.04%) higher** than
GRU96. There is no consistent Mamba-over-GRU generalization.
These are descriptive differences; no independent statistical
significance test was performed.

## Secondary metrics (m) and equal-rule offline LM solver

| Drive | Method | 2D RMSE | relative displacement 1-frame RMSE | relative displacement 10-frame RMSE | passes | stable | LM seconds |
|---|---|---:|---:|---:|---:|---|---:|
| 0016 | Axis-Fixed | 2.615367 | 0.277797 | 1.651896 | 2 | yes | 0.502 |
| 0016 | GRU96-Cap4 | 2.432298 | 0.298717 | 1.782934 | 2 | yes | 0.345 |
| 0016 | GRU106-Cap4 | 2.595701 | 0.299493 | 1.782394 | 2 | yes | 0.447 |
| 0016 | Mamba-Cap4 | **2.420881** | **0.271891** | **1.599708** | 2 | yes | 0.478 |
| 0020 | Axis-Fixed | 16.554789 | 1.045543 | 9.391053 | 2 | yes | 1.033 |
| 0020 | GRU96-Cap4 | **2.853548** | **0.343497** | **2.092813** | 3 | yes | 4.880 |
| 0020 | GRU106-Cap4 | 3.484441 | 0.452936 | 3.220537 | 2 | yes | 2.628 |
| 0020 | Mamba-Cap4 | 6.599064 | 0.589063 | 4.889186 | 2 | yes | 2.094 |

Factors per graph:
0016 GPS=279, IMU=278, LiDAR=278, Camera=275 (1111 including prior).
0020 GPS=1104, IMU=1103, LiDAR=1103, Camera=1063 (4374 including prior).
LM seconds include neither sensor factor generation nor neural inference.

## Valid claim boundaries

- All methods use identical generated GPS/IMU/LiDAR/stereo measurements
  within each drive, common LiDAR-integrated initial poses, seed 43,
  frozen TRAIN-only five-bin XYZ noise mapping and Cap4 information cap.
  First 63 frames use identical Axis-Fixed sigmas; remaining frames
  use only current h0 network reliability for factor weights.
- This is OFFLINE BATCH Pose3 FGO, **not** incremental iSAM2.
  The h1/h2 forecast or evidence-uncertainty heads are **not tested**
  as predictive factor-graph weighting mechanisms.
- OXTS GPS generates both synthetic corrupted GPS measurements and
  the uncorrupted evaluation reference. It is not independent physical
  position ground truth. The reported errors are fixed-origin
  position/displacement diagnostics, not full SE(3) RPE.
- Smaller cross-method graph objectives cannot establish better
  state estimation because GPS factor covariances differ.
  Numerical stable=True is not a global optimality guarantee.
- The two drives represent two sequences (unequal length), one
  checkpoint seed and one optimization initialization; do not assert
  architecture-level superiority or statistical significance.
- The 0020 timestamp synchronization policy was modified after the
  first reserved sequence release. Preserve this fact and all logs.

## Manuscript direction

The results support a *conditional* conclusion that frozen
reliability-based dynamic GPS covariance can substantially improve
synthetic-corruption recovery relative to a fixed-noise baseline,
especially on 0020, while architecture choice strongly affects
cross-sequence robustness. GRU96 is strongest on the longer drive.
The current evidence does **not** justify a claim that Mamba is the
superior or generally robust method. Do not tune on 0016/0020
after observing these scores. Any additional confirmatory study
requires newly declared data and, ideally, independent ground truth.

## Post-run archive integrity verification (2026-11-20)

The user executed the read-only local archive validation after
recovery completed. The script verified that the score report exists,
`RUN_INCOMPLETE` does not exist, exactly the 0016 and 0020 drives
and four expected cases are present, each case's numerical solver
reported `stabilized=True`, the report explicitly records
`independent_physical_gt=False` and
`recovered_from_interrupted_initial_release=True`.

Confirmed file-level SHA256 digests (calculated on the Ubuntu server):

| Archive file | SHA256 |
|---|---|
| `results/isj_v7/heldout_cap4_v1/heldout_scores.json` | `acc333e491e3ed189d04bc5c6082f1d609342823b6a3c1114459e06c1be82589` |
| `results/isj_v7/heldout_cap4_v1/0016/build_manifest.json` | `99f617050e1df40e6eb4ceacfc2c3d0fcd05876791b6399cdcb8d0b9bbfbb268` |
| `results/isj_v7/heldout_cap4_v1/0020/build_manifest.json` | `7e60a9250fb284e96c80ffdf6926f8de7e7a4b0cf6751326ce07fd70f4161517` |

Final printed marker: `V7_HELDOUT_ARCHIVE_AUDIT_PASSED`.
No solver execution, retraining, reference recalculation, or
changes to reserved results were made by this archive check.

**Freeze boundary:** Do not retune or rerun on 0016/0020 after
observing the results. The 0020 timestamp exception remains
explicitly post-release. This archive verification confirms
file identity and reported metadata, not independent physical
ground truth, global optimality, forecast-head benefit, or
statistical superiority of Mamba over GRU.
