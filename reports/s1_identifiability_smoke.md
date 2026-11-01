# S1 identifiability study (PLAN T5), smoke config

> **Smoke config.** These runs only check that the pipeline works end to end (one seed, one noise level, short training, small grids). Smoke numbers do not answer S1.

Built from 18 run records in `experiments/s1_smoke/` by `apexpinn.eval.s1_report`. Every number is `Observed`, read from the record named by its run id. Labels and criterion outcomes apply the rules declared in PLAN T5 to those numbers. Config: 12 stints, compounds ['C2', 'C3', 'C4'], seeds [0], noise [0.05] s, models ['M1', 'M2'].

## S1b recovery

Relative error (estimate - truth) / truth of the trained `M1` and `M2` estimates, lap time only, initialized from the seed's prior draw. Recovered means |relative error| within the `O6` tolerance (`k_c` 25%, `rho_c` 15%) for every gated parameter. Residual ratio is residual RMS on scored laps / noise_s. `Observed`.

| Run id | Model | Noise s | Seed | Status | rho_c[C2] | rho_c[C3] | rho_c[C4] | k_c[C2] | k_c[C3] | k_c[C4] | Recovered | Max abs err, ungated | Residual ratio | Collapse flags |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_fit_M1_well_specified_noise0.05_seed0 | M1 | 0.05 | 0 | ok | +309.5% | +76.5% | +239.6% | -82.8% | -44.3% | -50.1% | no | +40.0% | 210.183 | none |
| main_fit_M2_well_specified_noise0.05_seed0 | M2 | 0.05 | 0 | ok | +157.3% | +6.4% | +29.1% | -78.9% | +13.9% | -58.8% | no | +47.8% | 2.594 | none |

## S1c identifiability

Labels follow the PLAN T5 method declarations: identifiable if every seed recovers within the `O6` tolerance and every profile is sharp; non-identifiable if at least half of the profiles are flat; weakly identifiable otherwise. Parameters without an `O6` tolerance are labeled from the profiles only (basis `profile`). Profiles use the exact ODE likelihood with `b_s` concentrated out, so the shapes are shared by `M1` and `M2`; the recovery part of the label is per model. Shapes and recovery flags are listed per seed, in seed order.

| Parameter | Noise s | Basis | Label | Profile shapes | Recovered |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k_c[C2] | 0.05 | M1 | **weakly_identifiable** | sharp | no |
| k_c[C2] | 0.05 | M2 | **weakly_identifiable** | sharp | no |
| k_c[C3] | 0.05 | M1 | **weakly_identifiable** | sharp | no |
| k_c[C3] | 0.05 | M2 | **identifiable** | sharp | yes |
| k_c[C4] | 0.05 | M1 | **weakly_identifiable** | sharp | no |
| k_c[C4] | 0.05 | M2 | **weakly_identifiable** | sharp | no |
| rho_c[C2] | 0.05 | M1 | **weakly_identifiable** | sharp | no |
| rho_c[C2] | 0.05 | M2 | **weakly_identifiable** | sharp | no |
| rho_c[C3] | 0.05 | M1 | **weakly_identifiable** | sharp | no |
| rho_c[C3] | 0.05 | M2 | **identifiable** | sharp | yes |
| rho_c[C4] | 0.05 | M1 | **weakly_identifiable** | sharp | no |
| rho_c[C4] | 0.05 | M2 | **weakly_identifiable** | sharp | no |

### Gauss-Newton spectrum and MLE

Smallest relative eigenvalue (eigenvalue / largest) of J^T J at the ODE-likelihood MLE, and the three largest loadings of its eigenvector (the least resolved direction). MLE at box edge lists the MLE coordinates on an edge of the optimizer box (`S1C_BOUNDS`). `Observed`.

| Run id | Noise s | Seed | MLE SSE | SSE at truth | Smallest relative eigenvalue | Weakest direction | MLE at box edge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_profile_ode_well_specified_noise0.05_seed0 | 0.05 | 0 | 0.6944 | 0.7265 | 1.51e-04 | kappa -0.93, k_c[C4] -0.21, h_opt_c[C4] -0.18 | none |

### Profile intervals

95% profile-likelihood intervals (deviance n log(SSE / SSE_min) <= 3.841) in natural units. A side marked open is unbounded inside the grid and reports the grid edge; a side marked open at failed point met a failed point before the threshold and reports the last finite grid point inside. Unconverged points are grid points where L-BFGS-B reported no convergence; they still enter the deviance, and a shape marked `*` rests on one (inside the interval or at a bracketing grid point), so it may be sharper than the true profile. Failed points are grid points whose solve found no finite ODE-likelihood value after its start (not counted among the unconverged points); they have no deviance, and a side that reaches one before the threshold is open. At box edge counts grid points with a free coordinate on the box edge. `n/a`: the record does not store the count. `Observed`.

| Run id | Parameter | Truth | MLE | 95% interval | Shape | Covers truth | Unconverged points | Failed points | At box edge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C2] | 0.0450 | 0.0463 | [0.0456, 0.0466] | sharp | no | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C3] | 0.0600 | 0.0606 | [0.0598, 0.0610] | sharp* | yes | 2, interval rests on one | 0 | 1 |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C4] | 0.0800 | 0.0820 | [0.0808, 0.0824] | sharp* | no | 1, interval rests on one | 0 | 1 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C2] | 0.2800 | 0.2645 | [0.2395, 0.2907] | sharp | yes | 0 | 0 | 1 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C3] | 0.3500 | 0.3966 | [0.3213, 0.4807] | sharp | yes | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C4] | 0.4300 | 0.4461 | [0.3241, 0.5930] | sharp | yes | 0 | 0 | 0 |

Declared treatment of the optimizer box and failed ODE solves (PLAN T5 method declarations, verbatim):

- **Optimizer box for `S1c`** (added 2026-11-01 after the first smoke run stopped on `max_steps` in two profile runs, before any full-config result was read): the MLE and every profile solve keep each free coordinate inside a box: `k_c` [0.01, 10], `ρ_c` [0.002, 2], `γ_c` [0.1, 60], `h_opt,c` [0.02, 0.98], `κ` [0.03, 30] (`physics.identifiability.S1C_BOUNDS`: the prior ranges widened 10× on each side, and the `h_opt` grid clip). The profiled coordinate is held at its grid value even outside the box. A solution with a free coordinate on the box edge is recorded and counted in the report. Reason: the likely cause of the stops (`Inferred`, not confirmed) is the unbounded L-BFGS-B line search in log coordinates reaching very large `k_c`, where the explicit Tsit5 solve needs more than 4096 steps.
- **Failed ODE solves in `S1c`** (added with the optimizer box; the fallback to the lowest finite evaluation added after review v1 of this change, before any result was read): an ODE solve that stops without success gives a non-finite SSE instead of stopping the run. Inside an L-BFGS-B solve, such an evaluation returns a large finite penalty with a zero gradient, so the line search backtracks. If the final iterate of a solve is still non-finite, the solve falls back to its lowest finite evaluation and counts as unconverged. A profile grid point whose solve found no finite evaluation after its start is a failed point: it has no deviance, it is not counted as unconverged, and the next grid point starts from the last finite solution. The 95% interval is read outward from the minimum; if that walk reaches a failed point before the deviance crosses 3.841, that side counts as unbounded, so the profile is `flat`. The report counts failed points per profile next to the unconverged points. Reason: a failed point gives no evidence, and reading it as unbounded can only make a profile look flatter than it is, never sharper.

## S1d misspecification

Noise 0.05 s, lap time only, seeds [0], variants ['wrong_phi', 'student_t', 'piecewise_wear', 'drop_early', 'traffic']. Each variant is compared against the truth it records (`rho_target`, else `Θ_true`).

Pass criteria, declared before running (PLAN T5, verbatim):

1. Every parameter labeled identifiable in `S1c` stays within 1.5× its `O6` tolerance (`ρ_c` 22.5%, `k_c` 37.5%) on all 3 seeds, except where criterion 4 predicts a shift.
2. The 95% profile-likelihood interval of each identifiable parameter covers its variant truth on at least 2 of 3 seeds.
3. Diagnostics flag the misfit where it is visible: residual RMS on scored laps rises above `noise_s` by at least 20% for `piecewise_wear` and `traffic`, and the collapse detector does not fire on any run. A variant where the fit is wrong and no diagnostic flags it is recorded as a silent failure.
4. `wrong_phi`: the recovered `ρ_c` shift relative to the well-specified run with the same seed is within 30% of `φ_used − φ_true` for each compound.

Declared clarifications (PLAN T5 method declarations, verbatim):

- **Reading of the misspecification criteria**: in criterion 1, "except where criterion 4 predicts a shift" means `ρ_c` under `wrong_phi` is judged by criterion 4 only, while `k_c` under `wrong_phi` is judged by criterion 1. Criterion 4 must hold for every compound on all 3 seeds. The criterion 3 residual RMS is `sqrt(data MSE)` on scored laps with the training offsets.
- **Criteria per model**: criteria 1 to 4 are evaluated per variant and per fitted model (`M1`, plus `M2` when included).
- **"Labeled identifiable in `S1c`"** (criteria 1 and 2) means the main-study label at the misspecification noise level (0.05 s): the label of that model for `ρ_c` and `k_c`, and the profile-only label for `γ_c`, `h_opt,c` and `κ`. A parameter with another label gets `not_applicable` for criteria 1 and 2.
- **Criterion 1** covers the parameters with an `O6` tolerance (`ρ_c`, `k_c`), because the other parameters have no `O6` tolerance to scale: |relative error| against the variant truth must stay at or below 1.5 × tolerance.
- **Criterion 2** uses the profile runs of the variant (exact ODE likelihood). An interval side that is unbounded inside the grid counts as covering that side, because the 95% region then extends at least to the grid edge. The cost: a profile that turns flat under misspecification passes coverage, so the report prints each seed's misspecification profile shape next to its coverage. "2 of 3 seeds" is applied as at least ceil(2n / 3) of the n configured seeds (2 at n = 3).
- **Criterion 3** counts a residual ratio (residual RMS / `noise_s`) of at least 1.2 as a flag. Any collapse flag in a variant's fit runs fails criterion 3.
- **Silent failures**: a fit is wrong on a seed when it fails criterion 1 or criterion 4 there. If that seed's run has a residual ratio below 1.2 and no collapse flag, it is a silent failure. Reason: criterion 4 is the only test of `ρ_c` under `wrong_phi`, so a wrong shift without a flag is a wrong fit without a flag. A parameter that is not labeled identifiable has no criterion 1 check and so no silent-failure check, because the main study gives no reliable reference for a correct fit of it.
- **Missing runs**: a planned run without a finished record gives the outcome `missing`, never `pass`.
- **Unconverged profile points**: grid points where L-BFGS-B reports no convergence still enter the deviance, because scipy also reports failure on harmless line-search stops. The report counts them per profile and marks a shape that rests on one: an unconverged point inside the interval or at a grid point bracketing an endpoint. Such a point can only make a profile look sharper than it is.
- **`M2` in the misspecification study**: included if the slowest main-study `M2` fit took under 600 s.
- While any main-study `M2` fit record is missing or failed, `M2` is undecided: part misspec runs the `M1` and profile runs, and adds the `M2` runs once every main-study `M2` fit has finished. A failing `M2` fit never blocks the mandatory `M1` runs.

**M2 in the misspecification study**: included (slowest main M2 fit 11.5 s < 600 s, run main_fit_M2_well_specified_noise0.05_seed0, measured).

### Outcomes

| Variant | Model | Criterion 1 | Criterion 2 | Criterion 3 | Criterion 4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| wrong_phi | M1 | governed_by_criterion_4 | not_applicable | pass | fail |
| wrong_phi | M2 | pass | fail | pass | fail |
| student_t | M1 | not_applicable | not_applicable | pass | not_applicable |
| student_t | M2 | pass | fail | pass | not_applicable |
| piecewise_wear | M1 | not_applicable | not_applicable | pass | not_applicable |
| piecewise_wear | M2 | fail | fail | pass | not_applicable |
| drop_early | M1 | not_applicable | not_applicable | pass | not_applicable |
| drop_early | M2 | pass | pass | pass | not_applicable |
| traffic | M1 | not_applicable | not_applicable | pass | not_applicable |
| traffic | M2 | pass | fail | pass | not_applicable |

### Details

Criterion 2 lists each seed's misspecification profile shape next to coverage, because an open interval side counts as covering; `*` marks a shape that rests on an unconverged profile point.

- **wrong_phi / M1**: C1 rho_c[C2]: governed_by_criterion_4; C1 rho_c[C3]: governed_by_criterion_4; C1 rho_c[C4]: governed_by_criterion_4; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 216.051); C4 rho_c[C2]: fail (shifts -0.0036); C4 rho_c[C3]: fail (shifts -0.0050); C4 rho_c[C4]: fail (shifts 0.0255).
- **wrong_phi / M2**: C1 rho_c[C2]: governed_by_criterion_4; C1 rho_c[C3]: governed_by_criterion_4; C1 rho_c[C4]: governed_by_criterion_4; C1 k_c[C2]: not_applicable; C1 k_c[C3]: pass (errors +19.3%); C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: fail (covered on 0 of 1 seeds; shapes sharp*); C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: pass (covered on 1 of 1 seeds; shapes sharp); C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 2.453); C4 rho_c[C2]: fail (shifts 0.0058); C4 rho_c[C3]: pass (shifts 0.0083); C4 rho_c[C4]: fail (shifts 0.0055).
- **student_t / M1**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: not_applicable; C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 208.748).
- **student_t / M2**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: pass (errors +9.7%); C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: pass (errors +6.4%); C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: fail (covered on 0 of 1 seeds; shapes sharp*); C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: pass (covered on 1 of 1 seeds; shapes sharp); C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 2.764).
- **piecewise_wear / M1**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: not_applicable; C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 211.110).
- **piecewise_wear / M2**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: fail (errors +26.5%); C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: fail (errors -38.4%); C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: fail (covered on 0 of 1 seeds; shapes wide); C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: fail (covered on 0 of 1 seeds; shapes wide); C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 2.459).
- **drop_early / M1**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: not_applicable; C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 209.818).
- **drop_early / M2**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: pass (errors +4.9%); C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: pass (errors +14.5%); C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: pass (covered on 1 of 1 seeds; shapes sharp*); C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: pass (covered on 1 of 1 seeds; shapes sharp); C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 2.608).
- **traffic / M1**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: not_applicable; C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 211.264).
- **traffic / M2**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: pass (errors +4.2%); C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: pass (errors +6.2%); C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: fail (covered on 0 of 1 seeds; shapes sharp); C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: pass (covered on 1 of 1 seeds; shapes sharp); C2 k_c[C4]: not_applicable; C3: pass (residual ratio seed 0 2.947).

### Silent failures

- None found among the finished runs.

## Wall time

Measured wall time in seconds per run, read from the JSON records (`Observed`; machine and device in each record's `meta`). Runs without a record have no measured time, and no estimate is given for them.

| Part | Kind | Model | Runs | Median total s | Max total s | Sum total s | Max fit s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main | fit | M1 | 1 | 13.4 | 13.4 | 13.4 | 3.7 |
| main | fit | M2 | 1 | 11.5 | 11.5 | 11.5 | 11.5 |
| main | profile | ode | 1 | 39.0 | 39.0 | 39.0 | n/a |
| misspec | fit | M1 | 5 | 0.9 | 2.1 | 5.4 | 0.6 |
| misspec | fit | M2 | 5 | 1.1 | 1.3 | 5.4 | 1.3 |
| misspec | profile | ode | 5 | 29.4 | 41.4 | 153.3 | n/a |

Sum over all records: 228.2 s. Slowest main-study M2 fit: 11.5 s (run main_fit_M2_well_specified_noise0.05_seed0); the threshold for M2 in the misspecification study is 600 s.

## P1 trigger evidence

These are mechanical trigger checks against PRD section 8, not the decision. The P1 decision is made by the user in `experiments/pivots.md`. Realistic noise for R1 is 0.15 s (`Assumed`, PLAN T5 method declarations).

- **R0 (plan)**: the default rung. It stays unless the user moves to R1 or R2.
- **R1 trigger** (`k_c` or `h_opt,c` non-identifiable at realistic noise), mechanical trigger check, not the decision: `not_evaluable`.
  noise 0.15 s is not in this config.
- **R2 indicator** (`M1` stalls on at least 1 seeds at one noise level, with the `M2` records of those seeds present), mechanical trigger check, not the decision: `indicator_absent`. Declared rule (PLAN T5, verbatim): **R2 indicator for P1** (mechanical, not the decision): an `M1` fit stalls when its objective fell by less than 1% over the last 20% of logged steps while its residual ratio stayed at or above 1.2, so the data term did not reach the noise level. The indicator is present when `M1` stalls on at least 3 seeds (all seeds if fewer) at one noise level and the `M2` records of those seeds exist. `M1` and `M2` recovery and the final normalized loss terms are shown as context. The `R4` remedies and the tuning budget of `PRD §8` are not checked by code.

| Run id | Model | Noise s | Seed | Status | Stalled | Objective change, last 20% | Residual ratio | Final normalized terms | Not recovered |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_fit_M1_well_specified_noise0.05_seed0 | M1 | 0.05 | 0 | ok | no | -0.1811 | 210.183 | data 0.362, mono 0.000, reg 1.093, thermal 0.063, wear 0.027 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M2_well_specified_noise0.05_seed0 | M2 | 0.05 | 0 | ok | no | -0.2136 | 2.594 | data 0.046 | rho_c[C2], rho_c[C4], k_c[C2], k_c[C4] |

- **E5** (refuted if a relative Gauss-Newton eigenvalue is below 1e-06 at noise 0.05 s), mechanical trigger check, not the decision: `not_refuted`. Values: main_profile_ode_well_specified_noise0.05_seed0 1.51e-04.

## Missing runs

- None: every planned run has a finished record.

## How to run

From the repository root, after `uv sync` (uv installs Python 3.12 itself):

```
uv run python scripts/run_s1.py --config smoke                  # part main, then part misspec; finished runs are skipped
uv run python scripts/run_s1.py --config smoke --part main
uv run python scripts/run_s1.py --config smoke --part misspec   # M2 runs are added once every main-study M2 fit has finished
uv run python scripts/run_s1.py --config smoke --report-only    # rebuild this report from the JSON records
uv run python scripts/run_s1.py --config smoke --part main --shard 0/2   # machine A
uv run python scripts/run_s1.py --config smoke --part main --shard 1/2   # machine B
uv run python scripts/run_s1.py --config smoke --part main --only main_profile_ode_well_specified_noise0.05_seed0   # one run
```

With shards, copy every machine's `experiments/s1_smoke/` into one folder before `--part misspec`. No full-study wall time has been measured yet; the wall-time table above is the only source. Before an overnight run, time one profile run with `--only` as above: its record is kept and the full run skips it.
