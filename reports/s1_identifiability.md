# S1 identifiability study (PLAN T5), full config

Built from 31 run records in `experiments/s1/` by `apexpinn.eval.s1_report`. Every number is `Observed`, read from the record named by its run id. Labels and criterion outcomes apply the rules declared in PLAN T5 to those numbers. Config: 48 stints, compounds ['C2', 'C3', 'C4'], seeds [0, 1, 2, 3, 4], noise [0.05, 0.15] s, models ['M1', 'M2', 'M2v2'].

## S1b recovery

Relative error (estimate - truth) / truth of the trained `M1`, `M2` and `M2v2` estimates, lap time only. `M1` and `M2` (S1b v1) start from the seed's prior draw; `M2v2` (`M2` v2) keeps the best of several prior draws by final scored residual RMS. Recovered means |relative error| within the `O6` tolerance (`k_c` 25%, `rho_c` 15%) for every gated parameter. Residual ratio is residual RMS on scored laps / noise_s. `Observed`.

| Run id | Model | Noise s | Seed | Status | rho_c[C2] | rho_c[C3] | rho_c[C4] | k_c[C2] | k_c[C3] | k_c[C4] | Recovered | Max abs err, ungated | Residual ratio | Collapse flags |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_fit_M1_well_specified_noise0.05_seed0 | M1 | 0.05 | 0 | ok | +140.2% | +462.2% | +216.6% | -99.5% | -95.3% | -96.3% | no | +117.8% | 54.080 | none |
| main_fit_M1_well_specified_noise0.05_seed1 | M1 | 0.05 | 1 | ok | +1490.8% | +847.3% | +323.1% | -96.7% | -93.5% | -98.9% | no | +92.9% | 32.652 | none |
| main_fit_M1_well_specified_noise0.05_seed2 | M1 | 0.05 | 2 | ok | +623.5% | +644.7% | +273.2% | -89.6% | -93.1% | -94.4% | no | +277.9% | 5.642 | none |
| main_fit_M1_well_specified_noise0.05_seed3 | M1 | 0.05 | 3 | ok | +419.1% | +801.8% | +525.7% | -89.2% | -97.3% | -97.5% | no | +221.7% | 20.095 | none |
| main_fit_M1_well_specified_noise0.05_seed4 | M1 | 0.05 | 4 | ok | +1475.0% | +1018.5% | +3963.3% | -97.0% | -97.9% | -98.8% | no | +213.3% | 227.347 | none |
| main_fit_M1_well_specified_noise0.15_seed0 | M1 | 0.15 | 0 | ok | -75.2% | -73.8% | -85.4% | -99.1% | -94.8% | -95.7% | no | +107.3% | 26.343 | none |
| main_fit_M1_well_specified_noise0.15_seed1 | M1 | 0.15 | 1 | ok | +346.0% | +181.0% | +77.6% | -98.0% | -93.4% | -99.2% | no | +88.9% | 15.821 | none |
| main_fit_M1_well_specified_noise0.15_seed2 | M1 | 0.15 | 2 | ok | +668.3% | +689.5% | +307.3% | -89.4% | -92.9% | -93.4% | no | +277.3% | 3.270 | none |
| main_fit_M1_well_specified_noise0.15_seed3 | M1 | 0.15 | 3 | ok | +414.3% | +790.6% | +516.1% | -89.2% | -97.3% | -97.6% | no | +224.9% | 8.729 | none |
| main_fit_M1_well_specified_noise0.15_seed4 | M1 | 0.15 | 4 | ok | +2883.4% | +1812.3% | +3800.0% | -95.4% | -96.6% | -99.0% | no | +244.1% | 64.167 | none |
| main_fit_M2_well_specified_noise0.05_seed0 | M2 | 0.05 | 0 | ok | +95.1% | -0.2% | +0.6% | -88.1% | -2.3% | -7.7% | no | +28.7% | 1.711 | none |
| main_fit_M2_well_specified_noise0.05_seed1 | M2 | 0.05 | 1 | ok | +1.2% | +1.6% | -1.1% | +3.7% | +1.8% | -7.5% | yes | +9.5% | 0.957 | none |
| main_fit_M2_well_specified_noise0.05_seed2 | M2 | 0.05 | 2 | ok | +1.4% | -0.7% | +0.2% | +3.7% | -12.3% | +20.0% | yes | +48.5% | 0.996 | none |
| main_fit_M2_well_specified_noise0.05_seed3 | M2 | 0.05 | 3 | ok | +0.2% | +2.8% | -1.2% | -0.1% | -12.8% | -30.7% | no | +39.8% | 0.965 | none |
| main_fit_M2_well_specified_noise0.05_seed4 | M2 | 0.05 | 4 | ok | +2.1% | +0.8% | -1.7% | -7.4% | -6.4% | -15.7% | yes | +14.1% | 1.015 | none |
| main_fit_M2_well_specified_noise0.15_seed0 | M2 | 0.15 | 0 | ok | +94.2% | -0.1% | +3.4% | -88.7% | -8.6% | -3.5% | no | +28.2% | 1.091 | none |
| main_fit_M2_well_specified_noise0.15_seed1 | M2 | 0.15 | 1 | ok | +1.7% | +5.1% | -1.3% | +17.0% | +7.1% | -2.7% | yes | +11.7% | 0.953 | none |
| main_fit_M2_well_specified_noise0.15_seed2 | M2 | 0.15 | 2 | ok | +3.2% | -2.6% | +0.1% | +20.1% | -23.6% | +28.3% | no | +49.8% | 0.966 | none |
| main_fit_M2_well_specified_noise0.15_seed3 | M2 | 0.15 | 3 | ok | -0.3% | +7.2% | -0.6% | +9.3% | -6.9% | -37.5% | no | +37.0% | 0.956 | none |
| main_fit_M2_well_specified_noise0.15_seed4 | M2 | 0.15 | 4 | ok | +6.5% | +2.4% | -0.7% | -18.5% | -14.9% | -16.5% | yes | +28.9% | 1.013 | none |
| main_fit_M2v2_well_specified_noise0.05_seed0 | M2v2 | 0.05 | 0 | ok | +44.9% | +0.1% | +1.3% | -76.0% | -5.6% | -6.9% | no | +12.3% | 1.423 | none |
| main_fit_M2v2_well_specified_noise0.05_seed1 | M2v2 | 0.05 | 1 | ok | +1.2% | +1.6% | -1.1% | +3.7% | +1.8% | -7.5% | yes | +9.5% | 0.957 | none |
| main_fit_M2v2_well_specified_noise0.05_seed2 | M2v2 | 0.05 | 2 | ok | +1.2% | -3.2% | -0.2% | +6.4% | -13.8% | -6.5% | yes | +8.3% | 0.965 | none |
| main_fit_M2v2_well_specified_noise0.05_seed3 | M2v2 | 0.05 | 3 | ok | -0.4% | +2.0% | +0.9% | +6.0% | +3.9% | -0.5% | yes | +4.1% | 0.954 | none |
| main_fit_M2v2_well_specified_noise0.05_seed4 | M2v2 | 0.05 | 4 | ok | +2.1% | +0.8% | -1.7% | -7.4% | -6.4% | -15.7% | yes | +14.1% | 1.015 | none |
| main_fit_M2v2_well_specified_noise0.15_seed0 | M2v2 | 0.15 | 0 | ok | +44.5% | +0.0% | +3.5% | -76.5% | -9.2% | -0.5% | no | +19.7% | 1.044 | none |
| main_fit_M2v2_well_specified_noise0.15_seed1 | M2v2 | 0.15 | 1 | ok | +1.7% | +5.1% | -1.3% | +17.0% | +7.1% | -2.7% | yes | +11.7% | 0.953 | none |
| main_fit_M2v2_well_specified_noise0.15_seed2 | M2v2 | 0.15 | 2 | ok | +3.0% | -4.9% | -0.1% | +22.8% | -19.4% | -5.8% | yes | +18.1% | 0.962 | none |
| main_fit_M2v2_well_specified_noise0.15_seed3 | M2v2 | 0.15 | 3 | ok | -0.9% | +6.2% | +2.4% | +15.1% | +10.2% | -4.7% | yes | +12.2% | 0.954 | none |
| main_fit_M2v2_well_specified_noise0.15_seed4 | M2v2 | 0.15 | 4 | ok | +6.5% | +2.4% | -0.7% | -18.5% | -14.9% | -16.5% | yes | +28.9% | 1.013 | none |

### Recovered fits per model and noise level

Count of planned main fits that recover every gated parameter, per model and noise level, and over all noise levels. Missing counts planned fits without a finished record; they are never counted as recovered. The v1 rows (`M1`, `M2`) are the frozen S1b v1 outcome; `M2v2` is `M2` v2, reported alongside v1. `Observed`.

| Model | Method version | Noise s | Recovered | Missing |
| :--- | :--- | :--- | :--- | :--- |
| M1 | v1 | 0.05 | 0 of 5 | 0 |
| M1 | v1 | 0.15 | 0 of 5 | 0 |
| M1 | v1 | all | 0 of 10 | 0 |
| M2 | v1 | 0.05 | 3 of 5 | 0 |
| M2 | v1 | 0.15 | 2 of 5 | 0 |
| M2 | v1 | all | 5 of 10 | 0 |
| M2v2 | v2 | 0.05 | 4 of 5 | 0 |
| M2v2 | v2 | 0.15 | 4 of 5 | 0 |
| M2v2 | v2 | all | 8 of 10 | 0 |

### `M2` v1 vs `M2` v2

Per seed: `M2` v1 (one prior draw) and `M2` v2 (best start by final scored residual RMS; the truth is not used to select). Residual ratio per start lists start index: ratio. `missing`: no finished record. `Observed`.

Start 0 matches v1: start 0 repeats the v1 fit (same prior draw, train key and setup), so its residual ratio should equal the v1 ratio within a relative 1e-06; `no` points to nondeterminism or setup drift. A consistency check only; it changes no selection, recovery or label.

| Noise s | Seed | v1 recovered | v1 residual ratio | v1 not recovered | v2 recovered | v2 residual ratio | v2 not recovered | v2 selected start | v2 residual ratio per start | Start 0 matches v1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.05 | 0 | no | 1.711 | rho_c[C2], k_c[C2] | no | 1.423 | rho_c[C2], k_c[C2] | 1 | 0: 1.711, 1: 1.423, 2: 1.605 | yes |
| 0.05 | 1 | yes | 0.957 | none | yes | 0.957 | none | 0 | 0: 0.957, 1: 0.964, 2: 0.960 | yes |
| 0.05 | 2 | yes | 0.996 | none | yes | 0.965 | none | 2 | 0: 0.996, 1: 1.526, 2: 0.965 | yes |
| 0.05 | 3 | no | 0.965 | k_c[C4] | yes | 0.954 | none | 1 | 0: 0.965, 1: 0.954, 2: 1.057 | yes |
| 0.05 | 4 | yes | 1.015 | none | yes | 1.015 | none | 0 | 0: 1.015, 1: 1.962, 2: 1.021 | yes |
| 0.15 | 0 | no | 1.091 | rho_c[C2], k_c[C2] | no | 1.044 | rho_c[C2], k_c[C2] | 1 | 0: 1.091, 1: 1.044, 2: 1.074 | yes |
| 0.15 | 1 | yes | 0.953 | none | yes | 0.953 | none | 0 | 0: 0.953, 1: 0.954, 2: 0.953 | yes |
| 0.15 | 2 | no | 0.966 | k_c[C4] | yes | 0.962 | none | 2 | 0: 0.966, 1: 1.039, 2: 0.962 | yes |
| 0.15 | 3 | no | 0.956 | k_c[C4] | yes | 0.954 | none | 1 | 0: 0.956, 1: 0.954, 2: 0.967 | yes |
| 0.15 | 4 | yes | 1.013 | none | yes | 1.013 | none | 0 | 0: 1.013, 1: 1.139, 2: 1.014 | yes |

Declared S1b method versions (PLAN T5 method declarations, verbatim):

- **S1b v1 outcome** (declared 2026-11-01, after all 20 main fits under the v1 declarations were read): the S1b recovery outcome under the v1 method declarations is final as read: `M1` recovered on 0 of 10 main fits and `M2` on 5 of 10. The v1 run records in `experiments/s1/` are kept, never overwritten or relabeled, and the report keeps showing them. Later method versions are reported alongside v1 and never replace it.
- **`M2` v2, multi-start** (declared 2026-11-01, after the v1 `M2` results were read; a post-hoc method version reported alongside v1, never in place of it; the R1 trigger basis added after review v1 of this change, before any `M2` v2 fit was run): an `M2` v2 fit trains the v1 `M2` setup (Adam, learning rate 1e-2, 1000 steps, lap time only) from 3 independent prior draws per seed (2 in the smoke config): start 0 is the v1 prior draw `fold_in(key, 101)`, start i >= 1 is the profile MLE start `fold_in(key, 200 + i)`. The fit keeps the start with the lowest final scored residual RMS (the square root of its final data term with its training offsets); the truth is never used to select, and normalized objectives are not compared because each start normalizes by its own initial value. The step budget stays at 1000, so v2 changes only the starts: on seed 0 the v1 objective fell only 10.1% from step 500 to step 999. Reason: on seed 0 the v1 `M2` fit stopped at an SSE about 3.1 times the ODE-likelihood MLE SSE on the same data, while the MLE recovers C2, which points to an optimizer failure from that prior draw (`Inferred`). Scope: v2 targets optimizer failures only. An `M2` v2 miss while the fit sits at the noise floor (residual ratio below 1.2) remains evidence of weak identifiability, and no further optimizer change may be used to remove such misses. Run ids use the model tag `M2v2`. S1c labels are computed for `M2` and `M2v2` separately; the R1 trigger and the R2 indicator read only the v1 models (`M1` and `M2` v1), and the `M2` misspecification decision keeps using `M2` v1.

## S1c identifiability

Labels follow the PLAN T5 method declarations: identifiable if every seed recovers within the `O6` tolerance and every profile is sharp; non-identifiable if at least half of the profiles are flat; weakly identifiable otherwise. Parameters without an `O6` tolerance are labeled from the profiles only (basis `profile`). Profiles use the exact ODE likelihood with `b_s` concentrated out, so the shapes are shared by `M1` and `M2`; the recovery part of the label is per model. Shapes and recovery flags are listed per seed, in seed order.

| Parameter | Noise s | Basis | Label | Profile shapes | Recovered |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gamma_c[C2] | 0.05 | profile | **missing** | sharp | n/a |
| gamma_c[C3] | 0.05 | profile | **missing** | sharp | n/a |
| gamma_c[C4] | 0.05 | profile | **missing** | sharp | n/a |
| h_opt_c[C2] | 0.05 | profile | **missing** | sharp | n/a |
| h_opt_c[C3] | 0.05 | profile | **missing** | sharp | n/a |
| h_opt_c[C4] | 0.05 | profile | **missing** | sharp | n/a |
| k_c[C2] | 0.05 | M1 | **missing** | sharp | no, no, no, no, no |
| k_c[C2] | 0.05 | M2 | **missing** | sharp | no, yes, yes, yes, yes |
| k_c[C2] | 0.05 | M2v2 | **missing** | sharp | no, yes, yes, yes, yes |
| k_c[C3] | 0.05 | M1 | **missing** | sharp | no, no, no, no, no |
| k_c[C3] | 0.05 | M2 | **missing** | sharp | yes, yes, yes, yes, yes |
| k_c[C3] | 0.05 | M2v2 | **missing** | sharp | yes, yes, yes, yes, yes |
| k_c[C4] | 0.05 | M1 | **missing** | sharp | no, no, no, no, no |
| k_c[C4] | 0.05 | M2 | **missing** | sharp | yes, yes, yes, no, yes |
| k_c[C4] | 0.05 | M2v2 | **missing** | sharp | yes, yes, yes, yes, yes |
| kappa | 0.05 | profile | **missing** | sharp | n/a |
| rho_c[C2] | 0.05 | M1 | **missing** | sharp | no, no, no, no, no |
| rho_c[C2] | 0.05 | M2 | **missing** | sharp | no, yes, yes, yes, yes |
| rho_c[C2] | 0.05 | M2v2 | **missing** | sharp | no, yes, yes, yes, yes |
| rho_c[C3] | 0.05 | M1 | **missing** | sharp | no, no, no, no, no |
| rho_c[C3] | 0.05 | M2 | **missing** | sharp | yes, yes, yes, yes, yes |
| rho_c[C3] | 0.05 | M2v2 | **missing** | sharp | yes, yes, yes, yes, yes |
| rho_c[C4] | 0.05 | M1 | **missing** | sharp | no, no, no, no, no |
| rho_c[C4] | 0.05 | M2 | **missing** | sharp | yes, yes, yes, yes, yes |
| rho_c[C4] | 0.05 | M2v2 | **missing** | sharp | yes, yes, yes, yes, yes |
| gamma_c[C2] | 0.15 | profile | **missing** | n/a | n/a |
| gamma_c[C3] | 0.15 | profile | **missing** | n/a | n/a |
| gamma_c[C4] | 0.15 | profile | **missing** | n/a | n/a |
| h_opt_c[C2] | 0.15 | profile | **missing** | n/a | n/a |
| h_opt_c[C3] | 0.15 | profile | **missing** | n/a | n/a |
| h_opt_c[C4] | 0.15 | profile | **missing** | n/a | n/a |
| k_c[C2] | 0.15 | M1 | **missing** | n/a | no, no, no, no, no |
| k_c[C2] | 0.15 | M2 | **missing** | n/a | no, yes, yes, yes, yes |
| k_c[C2] | 0.15 | M2v2 | **missing** | n/a | no, yes, yes, yes, yes |
| k_c[C3] | 0.15 | M1 | **missing** | n/a | no, no, no, no, no |
| k_c[C3] | 0.15 | M2 | **missing** | n/a | yes, yes, yes, yes, yes |
| k_c[C3] | 0.15 | M2v2 | **missing** | n/a | yes, yes, yes, yes, yes |
| k_c[C4] | 0.15 | M1 | **missing** | n/a | no, no, no, no, no |
| k_c[C4] | 0.15 | M2 | **missing** | n/a | yes, yes, no, no, yes |
| k_c[C4] | 0.15 | M2v2 | **missing** | n/a | yes, yes, yes, yes, yes |
| kappa | 0.15 | profile | **missing** | n/a | n/a |
| rho_c[C2] | 0.15 | M1 | **missing** | n/a | no, no, no, no, no |
| rho_c[C2] | 0.15 | M2 | **missing** | n/a | no, yes, yes, yes, yes |
| rho_c[C2] | 0.15 | M2v2 | **missing** | n/a | no, yes, yes, yes, yes |
| rho_c[C3] | 0.15 | M1 | **missing** | n/a | no, no, no, no, no |
| rho_c[C3] | 0.15 | M2 | **missing** | n/a | yes, yes, yes, yes, yes |
| rho_c[C3] | 0.15 | M2v2 | **missing** | n/a | yes, yes, yes, yes, yes |
| rho_c[C4] | 0.15 | M1 | **missing** | n/a | no, no, no, no, no |
| rho_c[C4] | 0.15 | M2 | **missing** | n/a | yes, yes, yes, yes, yes |
| rho_c[C4] | 0.15 | M2v2 | **missing** | n/a | yes, yes, yes, yes, yes |

### Gauss-Newton spectrum and MLE

Smallest relative eigenvalue (eigenvalue / largest) of J^T J at the ODE-likelihood MLE, and the three largest loadings of its eigenvector (the least resolved direction). MLE at box edge lists the MLE coordinates on an edge of the optimizer box (`S1C_BOUNDS`). `Observed`.

| Run id | Noise s | Seed | MLE SSE | SSE at truth | Smallest relative eigenvalue | Weakest direction | MLE at box edge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_profile_ode_well_specified_noise0.05_seed0 | 0.05 | 0 | 2.9241 | 2.9524 | 0.001 | k_c[C4] -0.60, kappa -0.54, h_opt_c[C4] -0.38 | none |
| main_profile_ode_well_specified_noise0.05_seed1 | 0.05 | 1 | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | 0.05 | 2 | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | 0.05 | 3 | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | 0.05 | 4 | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | 0.15 | 0 | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | 0.15 | 1 | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | 0.15 | 2 | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | 0.15 | 3 | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | 0.15 | 4 | missing | n/a | n/a | n/a | n/a |

### Profile intervals

95% profile-likelihood intervals (deviance n log(SSE / SSE_min) <= 3.841) in natural units. A side marked open is unbounded inside the grid and reports the grid edge; a side marked open at failed point met a failed point before the threshold and reports the last finite grid point inside. Unconverged points are grid points where L-BFGS-B reported no convergence; they still enter the deviance, and a shape marked `*` rests on one (inside the interval or at a bracketing grid point), so it may be sharper than the true profile. Failed points are grid points whose solve found no finite ODE-likelihood value after its start (not counted among the unconverged points); they have no deviance, and a side that reaches one before the threshold is open. At box edge counts grid points with a free coordinate on the box edge. `n/a`: the record does not store the count. `Observed`.

Endpoints follow interval method v2 (PLAN T5 method declarations, verbatim):

- **Interval method v2** (declared 2026-11-01, after the seed 0, noise 0.05 s profile was read and before any other profile run was read; the re-solved bracket fallback added after review v1 of this change, before any v2 profile was run): each bounded endpoint of a 95% profile interval is found by Brent's method (`scipy.optimize.brentq`) on the coarse-grid bracket (the last grid point inside the 3.841 threshold and the first outside), where every evaluation is a real profile solve: L-BFGS-B with the profiled coordinate pinned, warm-started from the coarse solution at the inside bracket point, with the same box and `maxiter`, and a deviance against the same `SSE_min` as the coarse profile. The tolerance is `xtol` = 1e-4 in the grid coordinate (log, or logit for `h_opt_c`), with at most 30 iterations per side. If a refinement solve finds no finite SSE, the re-solved bracket ends no longer straddle the 3.841 threshold, or Brent's method does not converge, that side keeps the v1 linear endpoint and the report marks it. Refinement evaluations where L-BFGS-B reports no convergence are counted per side. Unbounded sides (grid edge, or a failed coarse point before the crossing) are unchanged. Shapes, coverage and labels use the v2 interval. Every profile record also stores the v1 linear interval, and the report prints it as `Superseded`. The seed 0, noise 0.05 s profile is rerun under v2, so every reported interval uses one method. The quadratic fit in `scripts/make_figures.py` stays a diagnostic only. Reason: on the coarse grid (factors 0.85, 1.0, 1.15 near the MLE, neighbour deviance 26 to 30) linear interpolation puts the crossing at about 3.841 / D of the grid step instead of about sqrt(3.841 / D), so the v1 intervals came out about 2.6 to 2.8 times too narrow (`Inferred`), and 6 of 13 seed 0 intervals missed the truth.

Interval method is `v2 refined` for records of profile method version 3 and `v1 linear` for older records. The v1 linear interval column is `Superseded` and shown for comparison only. Refinement lists the Brent evaluations per side (or why a side kept its v1 linear endpoint) and the refinement evaluations where L-BFGS-B reported no convergence.

Interval method per main profile record: v2 refined 1, v1 linear 0.

| Run id | Parameter | Truth | MLE | 95% interval | Shape | Covers truth | Interval method | v1 linear interval (Superseded) | Refinement | Unconverged points | Failed points | At box edge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C2] | 0.0450 | 0.0448 | [0.0438, 0.0458] | sharp* | yes | v2 refined | [0.0446, 0.0450] | lo 10 evals, hi 11 evals, 4 unconverged | 1, interval rests on one | 0 | 1 |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C3] | 0.0600 | 0.0600 | [0.0588, 0.0611] | sharp* | yes | v2 refined | [0.0597, 0.0601] | lo 9 evals, hi 11 evals, 4 unconverged | 1, interval rests on one | 0 | 4 |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C4] | 0.0800 | 0.0813 | [0.0796, 0.0826] | sharp* | yes | v2 refined | [0.0808, 0.0814] | lo 9 evals, hi 12 evals, 3 unconverged | 1, interval rests on one | 0 | 7 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C2] | 0.2800 | 0.2989 | [0.2752, 0.3249] | sharp | yes | v2 refined | [0.2864, 0.3146] | lo 8 evals, hi 8 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C3] | 0.3500 | 0.3356 | [0.3051, 0.3695] | sharp* | yes | v2 refined | [0.3172, 0.3587] | lo 8 evals, hi 8 evals, 3 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C4] | 0.4300 | 0.4595 | [0.4001, 0.5185] | sharp | yes | v2 refined | [0.4079, 0.5106] | lo 6 evals, hi 6 evals, 3 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | gamma_c[C2] | 3.2000 | 3.3209 | [3.1395, 3.5050] | sharp* | yes | v2 refined | [3.2532, 3.3903] | lo 10 evals, hi 9 evals, 3 unconverged | 1, interval rests on one | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed0 | gamma_c[C3] | 3.0000 | 3.0420 | [2.8387, 3.2541] | sharp* | yes | v2 refined | [2.9493, 3.1412] | lo 8 evals, hi 8 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | gamma_c[C4] | 2.8000 | 3.0553 | [2.7405, 3.3999] | sharp | yes | v2 refined | [2.8344, 3.3166] | lo 9 evals, hi 8 evals, 2 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | h_opt_c[C2] | 0.6400 | 0.6387 | [0.6217, 0.6566] | sharp* | yes | v2 refined | [0.6336, 0.6451] | lo 9 evals, hi 11 evals, 3 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | h_opt_c[C3] | 0.6000 | 0.5926 | [0.5752, 0.6098] | sharp* | yes | v2 refined | [0.5864, 0.5977] | lo 9 evals, hi 10 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | h_opt_c[C4] | 0.5700 | 0.5782 | [0.5550, 0.5971] | sharp* | yes | v2 refined | [0.5687, 0.5847] | lo 9 evals, hi 9 evals, 3 unconverged | 4, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | kappa | 1.0000 | 0.9181 | [0.8123, 1.0268] | sharp* | yes | v2 refined | [0.8345, 1.0028] | lo 7 evals, hi 8 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Declared treatment of the optimizer box and failed ODE solves (PLAN T5 method declarations, verbatim):

- **Optimizer box for `S1c`** (added 2026-11-01 after the first smoke run stopped on `max_steps` in two profile runs, before any full-config result was read): the MLE and every profile solve keep each free coordinate inside a box: `k_c` [0.01, 10], `ρ_c` [0.002, 2], `γ_c` [0.1, 60], `h_opt,c` [0.02, 0.98], `κ` [0.03, 30] (`physics.identifiability.S1C_BOUNDS`: the prior ranges widened 10× on each side, and the `h_opt` grid clip). The profiled coordinate is held at its grid value even outside the box. A solution with a free coordinate on the box edge is recorded and counted in the report. Reason: the likely cause of the stops (`Inferred`, not confirmed) is the unbounded L-BFGS-B line search in log coordinates reaching very large `k_c`, where the explicit Tsit5 solve needs more than 4096 steps.
- **Failed ODE solves in `S1c`** (added with the optimizer box; the fallback to the lowest finite evaluation added after review v1 of this change, before any result was read): an ODE solve that stops without success gives a non-finite SSE instead of stopping the run. Inside an L-BFGS-B solve, such an evaluation returns a large finite penalty with a zero gradient, so the line search backtracks. If the final iterate of a solve is still non-finite, the solve falls back to its lowest finite evaluation and counts as unconverged. A profile grid point whose solve found no finite evaluation after its start is a failed point: it has no deviance, it is not counted as unconverged, and the next grid point starts from the last finite solution. The 95% interval is read outward from the minimum; if that walk reaches a failed point before the deviance crosses 3.841, that side counts as unbounded, so the profile is `flat`. The report counts failed points per profile next to the unconverged points. Reason: a failed point gives no evidence, and reading it as unbounded can only make a profile look flatter than it is, never sharper.

## S1d misspecification

Noise 0.05 s, lap time only, seeds [0, 1, 2], variants ['wrong_phi', 'student_t', 'piecewise_wear', 'drop_early', 'traffic']. Each variant is compared against the truth it records (`rho_target`, else `Θ_true`).

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

**M2 in the misspecification study**: included (slowest main M2 fit 26.2 s < 600 s, run main_fit_M2_well_specified_noise0.05_seed1, measured).

### Outcomes

| Variant | Model | Criterion 1 | Criterion 2 | Criterion 3 | Criterion 4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| wrong_phi | M1 | missing | missing | missing | missing |
| wrong_phi | M2 | missing | missing | missing | missing |
| student_t | M1 | missing | missing | missing | not_applicable |
| student_t | M2 | missing | missing | missing | not_applicable |
| piecewise_wear | M1 | missing | missing | missing | not_applicable |
| piecewise_wear | M2 | missing | missing | missing | not_applicable |
| drop_early | M1 | missing | missing | missing | not_applicable |
| drop_early | M2 | missing | missing | missing | not_applicable |
| traffic | M1 | missing | missing | missing | not_applicable |
| traffic | M2 | missing | missing | missing | not_applicable |

### Details

Criterion 2 lists each seed's misspecification profile shape next to coverage, because an open interval side counts as covering; `*` marks a shape that rests on an unconverged profile point.

- **wrong_phi / M1**: C1 rho_c[C2]: governed_by_criterion_4; C1 rho_c[C3]: governed_by_criterion_4; C1 rho_c[C4]: governed_by_criterion_4; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing; C4 rho_c[C2]: missing; C4 rho_c[C3]: missing; C4 rho_c[C4]: missing.
- **wrong_phi / M2**: C1 rho_c[C2]: governed_by_criterion_4; C1 rho_c[C3]: governed_by_criterion_4; C1 rho_c[C4]: governed_by_criterion_4; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing; C4 rho_c[C2]: missing; C4 rho_c[C3]: missing; C4 rho_c[C4]: missing.
- **student_t / M1**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **student_t / M2**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **piecewise_wear / M1**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **piecewise_wear / M2**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **drop_early / M1**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **drop_early / M2**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **traffic / M1**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **traffic / M2**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.

### Silent failures

- None found among the finished runs.

## Wall time

Measured wall time in seconds per run, read from the JSON records (`Observed`; machine and device in each record's `meta`). Runs without a record have no measured time, and no estimate is given for them.

| Part | Kind | Model | Runs | Median total s | Max total s | Sum total s | Max fit s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main | fit | M1 | 10 | 28.0 | 49.7 | 323.1 | 39.0 |
| main | fit | M2 | 10 | 15.6 | 31.8 | 186.3 | 26.2 |
| main | fit | M2v2 | 10 | 54.3 | 68.0 | 556.0 | 65.3 |
| main | profile | ode | 1 | 538.1 | 538.1 | 538.1 | n/a |

Sum over all records: 1603.4 s. Slowest main-study M2 fit: 26.2 s (run main_fit_M2_well_specified_noise0.05_seed1); the threshold for M2 in the misspecification study is 600 s.

## P1 trigger evidence

These are mechanical trigger checks against PRD section 8, not the decision. The P1 decision is made by the user in `experiments/pivots.md`. Realistic noise for R1 is 0.15 s (`Assumed`, PLAN T5 method declarations).

- **R0 (plan)**: the default rung. It stays unless the user moves to R1 or R2.
- **R1 trigger** (`k_c` or `h_opt,c` non-identifiable at realistic noise), mechanical trigger check, not the decision: `incomplete`.
  Non-identifiable: none. Weakly identifiable: none.
- **R2 indicator** (`M1` stalls on at least 3 seeds at one noise level, with the `M2` records of those seeds present), mechanical trigger check, not the decision: `indicator_absent`. Declared rule (PLAN T5, verbatim): **R2 indicator for P1** (mechanical, not the decision): an `M1` fit stalls when its objective fell by less than 1% over the last 20% of logged steps while its residual ratio stayed at or above 1.2, so the data term did not reach the noise level. The indicator is present when `M1` stalls on at least 3 seeds (all seeds if fewer) at one noise level and the `M2` records of those seeds exist. `M1` and `M2` recovery and the final normalized loss terms are shown as context. The `R4` remedies and the tuning budget of `PRD §8` are not checked by code.

| Run id | Model | Noise s | Seed | Status | Stalled | Objective change, last 20% | Residual ratio | Final normalized terms | Not recovered |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_fit_M1_well_specified_noise0.05_seed0 | M1 | 0.05 | 0 | ok | no | -0.4861 | 54.080 | data 0.023, mono 0.000, reg 6.255, thermal 0.009, wear 2.21e-04 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.05_seed1 | M1 | 0.05 | 1 | ok | no | -0.4010 | 32.652 | data 0.007, mono 0.000, reg 6.613, thermal 0.008, wear 8.18e-05 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.05_seed2 | M1 | 0.05 | 2 | ok | no | -0.7503 | 5.642 | data 0.001, mono 0.000, reg 2.789, thermal 0.001, wear 5.13e-05 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.05_seed3 | M1 | 0.05 | 3 | ok | no | -0.3832 | 20.095 | data 0.012, mono 0.000, reg 2.693, thermal 0.006, wear 5.26e-05 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.05_seed4 | M1 | 0.05 | 4 | ok | no | -0.1655 | 227.347 | data 0.052, mono 0.000, reg 5.894, thermal 0.010, wear 1.22e-04 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M2_well_specified_noise0.05_seed0 | M2 | 0.05 | 0 | ok | no | -0.0318 | 1.711 | data 0.017 | rho_c[C2], k_c[C2] |
| main_fit_M2_well_specified_noise0.05_seed1 | M2 | 0.05 | 1 | ok | no | -0.0197 | 0.957 | data 0.011 | none |
| main_fit_M2_well_specified_noise0.05_seed2 | M2 | 0.05 | 2 | ok | no | -0.0249 | 0.996 | data 0.020 | none |
| main_fit_M2_well_specified_noise0.05_seed3 | M2 | 0.05 | 3 | ok | no | -0.0108 | 0.965 | data 0.018 | k_c[C4] |
| main_fit_M2_well_specified_noise0.05_seed4 | M2 | 0.05 | 4 | ok | no | -0.0013 | 1.015 | data 0.020 | none |
| main_fit_M1_well_specified_noise0.15_seed0 | M1 | 0.15 | 0 | ok | no | -0.4080 | 26.343 | data 0.044, mono 0.000, reg 4.965, thermal 0.010, wear 9.07e-05 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.15_seed1 | M1 | 0.15 | 1 | ok | no | -0.5512 | 15.821 | data 0.013, mono 0.000, reg 6.524, thermal 0.009, wear 8.11e-05 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.15_seed2 | M1 | 0.15 | 2 | ok | no | -0.4875 | 3.270 | data 0.002, mono 0.000, reg 2.665, thermal 0.001, wear 8.83e-05 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.15_seed3 | M1 | 0.15 | 3 | ok | no | -0.4271 | 8.729 | data 0.010, mono 0.000, reg 2.623, thermal 0.005, wear 4.47e-05 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.15_seed4 | M1 | 0.15 | 4 | ok | no | -0.2196 | 64.167 | data 0.039, mono 0.000, reg 7.297, thermal 0.009, wear 4.47e-05 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M2_well_specified_noise0.15_seed0 | M2 | 0.15 | 0 | ok | no | -0.0099 | 1.091 | data 0.060 | rho_c[C2], k_c[C2] |
| main_fit_M2_well_specified_noise0.15_seed1 | M2 | 0.15 | 1 | ok | no | -0.0039 | 0.953 | data 0.089 | none |
| main_fit_M2_well_specified_noise0.15_seed2 | M2 | 0.15 | 2 | ok | no | -0.0040 | 0.966 | data 0.145 | k_c[C4] |
| main_fit_M2_well_specified_noise0.15_seed3 | M2 | 0.15 | 3 | ok | no | -0.0013 | 0.956 | data 0.143 | k_c[C4] |
| main_fit_M2_well_specified_noise0.15_seed4 | M2 | 0.15 | 4 | ok | no | -0.0001 | 1.013 | data 0.153 | none |

- **E5** (refuted if a relative Gauss-Newton eigenvalue is below 1e-06 at noise 0.05 s), mechanical trigger check, not the decision: `incomplete`. Values: main_profile_ode_well_specified_noise0.05_seed0 0.001, main_profile_ode_well_specified_noise0.05_seed1 n/a, main_profile_ode_well_specified_noise0.05_seed2 n/a, main_profile_ode_well_specified_noise0.05_seed3 n/a, main_profile_ode_well_specified_noise0.05_seed4 n/a.
- **`M1` exploratory diagnostics** (PLAN T5, verbatim): **`M1` exploratory diagnostics** (declared 2026-11-01, after the v1 `M1` results were read): runs in `experiments/s1_m1_diag/` separate three candidate causes of the v1 `M1` failure (`k_c` collapsed on 10 of 10 main fits): (a) a longer budget (20000 steps), (b) the physics weight (thermal, wear and mono weights 0; or thermal and wear weights 10), (c) a bounded latent `h = tanh(tau * n_h)`. They use seed 0 (seed 1 optional), noise 0.05 s, and otherwise the v1 `M1` setup. They are exploratory: they are not S1b evidence, they change no v1 or v2 result or label, and no `M1` method change is declared from them. Any later `M1` method version must be declared here before its runs are read. The P1 decision stays with the user in `experiments/pivots.md`. Their runs and mechanical checks are in `reports/s1_m1_diag.md`, built by `scripts/run_m1_diag.py`.

## Missing runs

- main_profile_ode_well_specified_noise0.05_seed1: no record
- main_profile_ode_well_specified_noise0.05_seed2: no record
- main_profile_ode_well_specified_noise0.05_seed3: no record
- main_profile_ode_well_specified_noise0.05_seed4: no record
- main_profile_ode_well_specified_noise0.15_seed0: no record
- main_profile_ode_well_specified_noise0.15_seed1: no record
- main_profile_ode_well_specified_noise0.15_seed2: no record
- main_profile_ode_well_specified_noise0.15_seed3: no record
- main_profile_ode_well_specified_noise0.15_seed4: no record
- misspec_fit_M1_drop_early_noise0.05_seed0: no record
- misspec_fit_M1_drop_early_noise0.05_seed1: no record
- misspec_fit_M1_drop_early_noise0.05_seed2: no record
- misspec_fit_M1_piecewise_wear_noise0.05_seed0: no record
- misspec_fit_M1_piecewise_wear_noise0.05_seed1: no record
- misspec_fit_M1_piecewise_wear_noise0.05_seed2: no record
- misspec_fit_M1_student_t_noise0.05_seed0: no record
- misspec_fit_M1_student_t_noise0.05_seed1: no record
- misspec_fit_M1_student_t_noise0.05_seed2: no record
- misspec_fit_M1_traffic_noise0.05_seed0: no record
- misspec_fit_M1_traffic_noise0.05_seed1: no record
- misspec_fit_M1_traffic_noise0.05_seed2: no record
- misspec_fit_M1_wrong_phi_noise0.05_seed0: no record
- misspec_fit_M1_wrong_phi_noise0.05_seed1: no record
- misspec_fit_M1_wrong_phi_noise0.05_seed2: no record
- misspec_fit_M2_drop_early_noise0.05_seed0: no record
- misspec_fit_M2_drop_early_noise0.05_seed1: no record
- misspec_fit_M2_drop_early_noise0.05_seed2: no record
- misspec_fit_M2_piecewise_wear_noise0.05_seed0: no record
- misspec_fit_M2_piecewise_wear_noise0.05_seed1: no record
- misspec_fit_M2_piecewise_wear_noise0.05_seed2: no record
- misspec_fit_M2_student_t_noise0.05_seed0: no record
- misspec_fit_M2_student_t_noise0.05_seed1: no record
- misspec_fit_M2_student_t_noise0.05_seed2: no record
- misspec_fit_M2_traffic_noise0.05_seed0: no record
- misspec_fit_M2_traffic_noise0.05_seed1: no record
- misspec_fit_M2_traffic_noise0.05_seed2: no record
- misspec_fit_M2_wrong_phi_noise0.05_seed0: no record
- misspec_fit_M2_wrong_phi_noise0.05_seed1: no record
- misspec_fit_M2_wrong_phi_noise0.05_seed2: no record
- misspec_profile_ode_drop_early_noise0.05_seed0: no record
- misspec_profile_ode_drop_early_noise0.05_seed1: no record
- misspec_profile_ode_drop_early_noise0.05_seed2: no record
- misspec_profile_ode_piecewise_wear_noise0.05_seed0: no record
- misspec_profile_ode_piecewise_wear_noise0.05_seed1: no record
- misspec_profile_ode_piecewise_wear_noise0.05_seed2: no record
- misspec_profile_ode_student_t_noise0.05_seed0: no record
- misspec_profile_ode_student_t_noise0.05_seed1: no record
- misspec_profile_ode_student_t_noise0.05_seed2: no record
- misspec_profile_ode_traffic_noise0.05_seed0: no record
- misspec_profile_ode_traffic_noise0.05_seed1: no record
- misspec_profile_ode_traffic_noise0.05_seed2: no record
- misspec_profile_ode_wrong_phi_noise0.05_seed0: no record
- misspec_profile_ode_wrong_phi_noise0.05_seed1: no record
- misspec_profile_ode_wrong_phi_noise0.05_seed2: no record

## How to run

From the repository root, after `uv sync` (uv installs Python 3.12 itself):

```
uv run python scripts/run_s1.py --config full                  # part main, then part misspec; finished runs are skipped
uv run python scripts/run_s1.py --config full --part main
uv run python scripts/run_s1.py --config full --part misspec   # M2 runs are added once every main-study M2 fit has finished
uv run python scripts/run_s1.py --config full --report-only    # rebuild this report from the JSON records
uv run python scripts/run_s1.py --config full --part main --shard 0/2   # machine A
uv run python scripts/run_s1.py --config full --part main --shard 1/2   # machine B
uv run python scripts/run_s1.py --config full --part main --only main_profile_ode_well_specified_noise0.05_seed0   # one run
```

With shards, copy every machine's `experiments/s1/` into one folder before `--part misspec`. No full-study wall time has been measured yet; the wall-time table above is the only source. Before an overnight run, time one profile run with `--only` as above: its record is kept and the full run skips it.
