# Stage 9 — Read-only factor graph convergence audit

Seed43 V7 dev-pilot results (79 tests pass, all controls computed):

| GPS covariance | 3D RMSE m | 2D RMSE m | disp-1 m | disp-10 m |
| --- | ---: | ---: | ---: | ---: |
| Fixed 5m | 3.47507 | 3.13392 | 0.19720 | 1.34367 |
| Mamba-R time-varying | 3.32482 | 2.75894 | 0.20181 | 1.41112 |
| Mamba mean sigma | 18.49064 | 18.04494 | 0.42898 | 3.80857 |
| Mamba permuted sigmas | 3.72117 | 3.18280 | 0.21608 | 1.55517 |
| No GPS | 307.82362 | 307.79977 | 1.57809 | 15.65267 |

**These outcomes do not yet prove that predicted temporal weights improve FGO.**
The 18.49m constant sigma result warrants objective/convergence checking.
Different sigma mappings can lead a nonlinear optimizer to different basins.
Constant mean sigma also does not preserve mean information weight,
because factor precision is proportional to 1/sigma^2.

New read-only audit script reuses EXACT original V7 graph builder.
It checks saved solution, sigma arrays, trajectory hashes, dataset hashes,
model provenance, factor counts, 680 h0 validation anchors, and evaluates
saved vs original LiDAR initialization under the **same graph**.
It optionally continues the same LM solver from the saved trajectory,
and/or restarts one graph from a different saved method's trajectory.
Both alternatives are calculated in memory; NO files are written.

**Only compare objective values within one fixed graph configuration.**
Cross-case graph objective values are not comparable because measurement
sigmas define different weighted least-squares objectives.
All reported trajectories are in already explored KITTI 0027 validation,
evaluated against non-independent OXTS-derived GPS, not paper-ready ATE.

Run on server:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
set +e; set +u; set +o pipefail
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
echo V7_TEST_RC=$?
python -u -m experiments.isj_v7.audit_fg_convergence \
  --seed 43 --cases fixed mamba_r mamba_mean_sigma mamba_perm_sigma no_gps
echo OBJECTIVE_AUDIT_RC=$?
python -u -m experiments.isj_v7.audit_fg_convergence \
  --seed 43 --cases mamba_mean_sigma --refine --cross-start mamba_r
echo MEAN_SIGMA_CONVERGENCE_RC=$?
~~~

If an LM continuation or alternative initialization materially reduces
the same graph objective and changes poses, the original 18.49m result
may be solver-start sensitive. The solver-matched protocol must be
validated before making a claim about learnable temporal weights.
Original stored results are never overwritten.

Next, complete GRU96, GRU106 and Mamba-RU pilot controls and independent
evaluation-reference methodology; do not tune this heuristic using
the same selected validation segment and present it as held-out evidence.

## Stage 9b — Evaluate alternative optimized trajectories (development-only)

Seed43 Mamba-mean audit outcome from the server:

- Saved-solution objective: 5778.349329.
- One extra LM pass from saved: objective 5731.352772, pose shift RMSE 3.6533m.
- Under the identical Mamba-mean graph, Mamba-R saved poses start
  at objective 1048.942071 and continue to 1034.647777.
- The ~82.1% reduction is **within the SAME graph objective**.
  This is strong evidence of sensitivity to initialization and/or
  local basin selection, NOT proof of a global optimum.
- Original mean-sigma 18.4906m OXTS-relative error is not
  a sound direct comparison to the original Mamba-R 3.3248m
  without convergence-control.

New --score-development-reference flag computes diagnostics only
AFTER all requested LM continuations finish in memory. It reads
the non-independent clean OXTS reference exclusively for scoring,
never to construct GPS sigma or to initialize/optimize the graph.
The original archived trajectories remain unmodified.

Read-only step 1: show the earlier 5-case objective audit:

~~~bash
cd /root/projects/DFG
conda activate dfg-isj-gpu
grep -E '"case":|"objective_from_saved_solution":|"saved_3d_rmse_to_non_independent_oxts_m":' \
  logs/isj_v7_fg_objective_audit.log
~~~

Read-only step 2: evaluate the alternative solution under SAME graph
and show its OXTS-relative (NOT independent ground truth) error:

~~~bash
git pull --ff-only
python -m pytest tests/test_isj_v7_*.py -q
python -u -m experiments.isj_v7.audit_fg_convergence \
  --seed 43 --cases mamba_mean_sigma --refine --cross-start mamba_r \
  --score-development-reference
~~~

Optional further diagnostic: under each of the other saved case
objectives, use the exact SAME saved Mamba-R poses as an alternate
initialization. Only report after sufficient within-case convergence.

~~~bash
python -u -m experiments.isj_v7.audit_fg_convergence \
  --seed 43 --cases fixed mamba_r mamba_mean_sigma mamba_perm_sigma \
  --refine --cross-start mamba_r --score-development-reference
~~~

This optional common start is itself Mamba-derived and thus may
favor a particular basin. Repeat from Fixed saved poses, or from
other independent but predeclared starting trajectories, before
presenting a solver-matched comparison as a scientific result.
Do not inspect held-out drives 0016/0020 to tune the method.
