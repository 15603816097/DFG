# V8.3: observed 0027 development-only FGO results and integrity caveats

**Status:** 2026-11-22. The user ran the frozen 16-case 0027 DEV graph
experiment from V8 commit `556e277`; server reported `155 passed`
and `V8_3_GRAPH_EXIT=0`. All 16 graphs met the numerical-stability
flag after 2-4 LM passes, with the same factor counts
(GPS=680, IMU=679, LiDAR=679, Camera=670; plus one initial pose prior).
Final JSON stored on the user's server at
`results/isj_v8/dev_0027_feedforward_fgo_v1/v8_dev_feedforward_fgo_results.json`.
The console transcript included the following positional metrics;
the server-side JSON SHA256 is still **to be printed by the next read-only audit**.

## Original 3D RMSE observations, metre (lower is better)

| Method | GRU96 | GRU106 | Mamba |
|---|---:|---:|---:|
| h0(t), current observation available | 2.088096 | 7.230812 | **2.073255** |
| h0(t-1), causal persistence | 2.133697 | 6.434239 | 2.352062 |
| h1(t-1), one-step causal prior | 5.954731 | 6.751801 | 2.258977 |
| h2(t-2), two-step causal prior | 6.501540 | 2.216046 | 6.732990 |
| 0.5 h1(t-1) + 0.5 h2(t-2) | 2.221284 | **2.150280** | 6.644273 |

Training-axis fixed baseline: 3D RMSE = **7.107889 m**.
Best 3D RMSE of all 16 cases: **Mamba current h0(t) = 2.073255 m**.
These metrics compare to an OXTS/GPS-correlated synthetic reference,
not independent surveyed physical ground truth.

## Matched one-step persistence comparisons

For each encoder, the following comparisons share the same
680 frames, fixed startup policy, 5-bin train-only XYZ covariance
mapping, Cap4 safety rule, sensor factors and initial trajectory.

| Encoder | h1(t-1) relative 3D RMSE gain vs h0(t-1) | 50/50 blend relative 3D gain vs h0(t-1) |
|---|---:|---:|
| GRU96 | **-179.08%** (degradation) | **-4.10%** (degradation) |
| GRU106 | **-4.94%** (degradation) | **+66.58%** |
| Mamba | **+3.96%** | **-182.49%** (degradation) |

Negative values indicate that the forecast method had **larger**
3D RMSE than past-h0 persistence. Thus there is **no consistent**
positive FGO effect attributable to the future reliability heads.
The small Mamba h1 gain does not establish robust performance.

**Missing matched two-step baseline:** the original script used to
print the results attempted to read an uncomputed case named
`gru96_r__h0_tminus2_persistence`, causing a `KeyError`
after all 16 valid results were printed. The original experiment
did **not** include any model's `h0(t-2)` FGO case, so do not
silently compare h2(t-2) to h0(t-1) and do not invent a gain.
Correct summary script reports h2 absolute RMSE only and
marks the missing matched baseline explicitly.

## Major numerical caveat

Several variations of the same frozen 680-frame graph family
yield 3D errors near 2 m, whereas others yield near 6-7 m.
Even modest changes in mean per-axis sigma accompany radically
different reported errors. The recorded `stable=True` condition
is a local numerical stopping criterion, not a certificate of
unique or global optimum. The evidence is compatible with, but
does not yet prove, local-solution sensitivity, factor weighting
sensitivity, or another implementation issue. Do not select a
single unusually good case and call it a robust model improvement.

A new **read-only** audit script checks the original JSON's
case list, result structure, source boundaries, factor counts,
LM-pass stability, 3D/2D/displacement metrics, covariance summaries
and SHA256. It never reruns the solver.

## Scientific scope and next decision

- V8.1 and V8.2 showed small improvements in *synthetic reliability*
  forecast RMSE, not state-estimation accuracy.
- V8.3 shows no consistent benefit from mapping those forecasts
  into the original offline factor graph.
- Prediction-based weights are temporally causal, **but the Pose3
  optimizer uses the whole 680-frame batch**; no online
  low-latency estimator, iSAM2, residual feedback or calibrated
  uncertainty is demonstrated.
- V8.3 was a repeatedly explored **0027 development** evaluation,
  not new independent confirmation; it cannot repair V7's
  measurement-correlated OXTS/GPS reference.
- After printing the SHA256 and the solver convergence diagnostics,
  independently examine optimizer sensitivity on 0027 DEV
  without touching frozen 0016/0020. Do not alter hyperparameters
  based on those already-scored reserved sequences.

**Do not** write a positive predictive FGO claim into the ISJ
manuscript until an appropriate new validation protocol supports it.