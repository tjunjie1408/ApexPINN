# S1 identifiability study (PLAN T5), full config

Built from 40 run records in `experiments/s1/` by `apexpinn.eval.s1_report`. Every number is `Observed`, read from the record named by its run id. Labels and criterion outcomes apply the rules declared in PLAN T5 to those numbers. Config: 48 stints, compounds ['C2', 'C3', 'C4'], seeds [0, 1, 2, 3, 4], noise [0.05, 0.15] s, models ['M1', 'M2', 'M2v2'].

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
| gamma_c[C2] | 0.05 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| gamma_c[C3] | 0.05 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| gamma_c[C4] | 0.05 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| h_opt_c[C2] | 0.05 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| h_opt_c[C3] | 0.05 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| h_opt_c[C4] | 0.05 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| k_c[C2] | 0.05 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| k_c[C2] | 0.05 | M2 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, yes, yes, yes, yes |
| k_c[C2] | 0.05 | M2v2 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, yes, yes, yes, yes |
| k_c[C3] | 0.05 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| k_c[C3] | 0.05 | M2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| k_c[C3] | 0.05 | M2v2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| k_c[C4] | 0.05 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| k_c[C4] | 0.05 | M2 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, no, yes |
| k_c[C4] | 0.05 | M2v2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| kappa | 0.05 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| rho_c[C2] | 0.05 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| rho_c[C2] | 0.05 | M2 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, yes, yes, yes, yes |
| rho_c[C2] | 0.05 | M2v2 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, yes, yes, yes, yes |
| rho_c[C3] | 0.05 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| rho_c[C3] | 0.05 | M2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| rho_c[C3] | 0.05 | M2v2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| rho_c[C4] | 0.05 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| rho_c[C4] | 0.05 | M2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| rho_c[C4] | 0.05 | M2v2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| gamma_c[C2] | 0.15 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| gamma_c[C3] | 0.15 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| gamma_c[C4] | 0.15 | profile | **weakly_identifiable** | wide, sharp, sharp, sharp, sharp | n/a |
| h_opt_c[C2] | 0.15 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| h_opt_c[C3] | 0.15 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| h_opt_c[C4] | 0.15 | profile | **identifiable** | sharp, sharp, sharp, sharp, sharp | n/a |
| k_c[C2] | 0.15 | M1 | **weakly_identifiable** | wide, wide, wide, sharp, sharp | no, no, no, no, no |
| k_c[C2] | 0.15 | M2 | **weakly_identifiable** | wide, wide, wide, sharp, sharp | no, yes, yes, yes, yes |
| k_c[C2] | 0.15 | M2v2 | **weakly_identifiable** | wide, wide, wide, sharp, sharp | no, yes, yes, yes, yes |
| k_c[C3] | 0.15 | M1 | **weakly_identifiable** | sharp, wide, sharp, wide, sharp | no, no, no, no, no |
| k_c[C3] | 0.15 | M2 | **weakly_identifiable** | sharp, wide, sharp, wide, sharp | yes, yes, yes, yes, yes |
| k_c[C3] | 0.15 | M2v2 | **weakly_identifiable** | sharp, wide, sharp, wide, sharp | yes, yes, yes, yes, yes |
| k_c[C4] | 0.15 | M1 | **weakly_identifiable** | wide, wide, wide, sharp, sharp | no, no, no, no, no |
| k_c[C4] | 0.15 | M2 | **weakly_identifiable** | wide, wide, wide, sharp, sharp | yes, yes, no, no, yes |
| k_c[C4] | 0.15 | M2v2 | **weakly_identifiable** | wide, wide, wide, sharp, sharp | yes, yes, yes, yes, yes |
| kappa | 0.15 | profile | **weakly_identifiable** | wide, wide, sharp, wide, wide | n/a |
| rho_c[C2] | 0.15 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| rho_c[C2] | 0.15 | M2 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, yes, yes, yes, yes |
| rho_c[C2] | 0.15 | M2v2 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, yes, yes, yes, yes |
| rho_c[C3] | 0.15 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| rho_c[C3] | 0.15 | M2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| rho_c[C3] | 0.15 | M2v2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| rho_c[C4] | 0.15 | M1 | **weakly_identifiable** | sharp, sharp, sharp, sharp, sharp | no, no, no, no, no |
| rho_c[C4] | 0.15 | M2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |
| rho_c[C4] | 0.15 | M2v2 | **identifiable** | sharp, sharp, sharp, sharp, sharp | yes, yes, yes, yes, yes |

### Gauss-Newton spectrum and MLE

Smallest relative eigenvalue (eigenvalue / largest) of J^T J at the ODE-likelihood MLE, and the three largest loadings of its eigenvector (the least resolved direction). MLE at box edge lists the MLE coordinates on an edge of the optimizer box (`S1C_BOUNDS`). `Observed`.

| Run id | Noise s | Seed | MLE SSE | SSE at truth | Smallest relative eigenvalue | Weakest direction | MLE at box edge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_profile_ode_well_specified_noise0.05_seed0 | 0.05 | 0 | 2.9241 | 2.9524 | 0.001 | k_c[C4] -0.60, kappa -0.54, h_opt_c[C4] -0.38 | none |
| main_profile_ode_well_specified_noise0.05_seed1 | 0.05 | 1 | 2.6638 | 2.6971 | 0.002 | kappa -0.56, k_c[C4] -0.55, h_opt_c[C4] -0.38 | none |
| main_profile_ode_well_specified_noise0.05_seed2 | 0.05 | 2 | 2.7456 | 2.7686 | 0.002 | kappa 0.56, k_c[C4] 0.49, k_c[C3] 0.41 | none |
| main_profile_ode_well_specified_noise0.05_seed3 | 0.05 | 3 | 2.5520 | 2.5954 | 7.73e-04 | kappa 0.84, k_c[C3] 0.25, k_c[C4] 0.25 | none |
| main_profile_ode_well_specified_noise0.05_seed4 | 0.05 | 4 | 3.1692 | 3.2220 | 0.001 | k_c[C4] -0.55, kappa -0.53, h_opt_c[C4] -0.39 | none |
| main_profile_ode_well_specified_noise0.15_seed0 | 0.15 | 0 | 26.3053 | 26.5718 | 0.001 | kappa 0.66, k_c[C4] 0.52, h_opt_c[C4] 0.32 | none |
| main_profile_ode_well_specified_noise0.15_seed1 | 0.15 | 1 | 23.9519 | 24.2735 | 0.002 | k_c[C4] -0.60, kappa -0.48, h_opt_c[C4] -0.38 | none |
| main_profile_ode_well_specified_noise0.15_seed2 | 0.15 | 2 | 24.7190 | 24.9175 | 0.002 | k_c[C3] -0.52, h_opt_c[C3] -0.48, kappa -0.46 | none |
| main_profile_ode_well_specified_noise0.15_seed3 | 0.15 | 3 | 22.9722 | 23.3582 | 8.51e-04 | kappa 0.85, k_c[C3] 0.25, h_opt_c[C3] 0.23 | none |
| main_profile_ode_well_specified_noise0.15_seed4 | 0.15 | 4 | 28.5420 | 28.9981 | 9.82e-04 | kappa 0.63, k_c[C4] 0.43, k_c[C3] 0.40 | none |

### Profile intervals

95% profile-likelihood intervals (deviance n log(SSE / SSE_min) <= 3.841) in natural units. A side marked open is unbounded inside the grid and reports the grid edge; a side marked open at failed point met a failed point before the threshold and reports the last finite grid point inside. Unconverged points are grid points where L-BFGS-B reported no convergence; they still enter the deviance, and a shape marked `*` rests on one (inside the interval or at a bracketing grid point), so it may be sharper than the true profile. Failed points are grid points whose solve found no finite ODE-likelihood value after its start (not counted among the unconverged points); they have no deviance, and a side that reaches one before the threshold is open. At box edge counts grid points with a free coordinate on the box edge. `n/a`: the record does not store the count. `Observed`.

Endpoints follow interval method v2 (PLAN T5 method declarations, verbatim):

- **Interval method v2** (declared 2026-11-01, after the seed 0, noise 0.05 s profile was read and before any other profile run was read; the re-solved bracket fallback added after review v1 of this change, before any v2 profile was run): each bounded endpoint of a 95% profile interval is found by Brent's method (`scipy.optimize.brentq`) on the coarse-grid bracket (the last grid point inside the 3.841 threshold and the first outside), where every evaluation is a real profile solve: L-BFGS-B with the profiled coordinate pinned, warm-started from the coarse solution at the inside bracket point, with the same box and `maxiter`, and a deviance against the same `SSE_min` as the coarse profile. The tolerance is `xtol` = 1e-4 in the grid coordinate (log, or logit for `h_opt_c`), with at most 30 iterations per side. If a refinement solve finds no finite SSE, the re-solved bracket ends no longer straddle the 3.841 threshold, or Brent's method does not converge, that side keeps the v1 linear endpoint and the report marks it. Refinement evaluations where L-BFGS-B reports no convergence are counted per side. Unbounded sides (grid edge, or a failed coarse point before the crossing) are unchanged. Shapes, coverage and labels use the v2 interval. Every profile record also stores the v1 linear interval, and the report prints it as `Superseded`. The seed 0, noise 0.05 s profile is rerun under v2, so every reported interval uses one method. The quadratic fit in `scripts/make_figures.py` stays a diagnostic only. Reason: on the coarse grid (factors 0.85, 1.0, 1.15 near the MLE, neighbour deviance 26 to 30) linear interpolation puts the crossing at about 3.841 / D of the grid step instead of about sqrt(3.841 / D), so the v1 intervals came out about 2.6 to 2.8 times too narrow (`Inferred`), and 6 of 13 seed 0 intervals missed the truth.

Interval method is `v2 refined` for records of profile method version 3 and `v1 linear` for older records. The v1 linear interval column is `Superseded` and shown for comparison only. Refinement lists the Brent evaluations per side (or why a side kept its v1 linear endpoint) and the refinement evaluations where L-BFGS-B reported no convergence.

Interval method per main profile record: v2 refined 10, v1 linear 0.

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
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C2] | 0.0450 | 0.0453 | [0.0443, 0.0463] | sharp | yes | v2 refined | [0.0451, 0.0455] | lo 11 evals, hi 11 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C3] | 0.0600 | 0.0609 | [0.0594, 0.0624] | sharp | yes | v2 refined | [0.0606, 0.0611] | lo 10 evals, hi 11 evals, 1 unconverged | 1 | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C4] | 0.0800 | 0.0799 | [0.0786, 0.0811] | sharp | yes | v2 refined | [0.0795, 0.0800] | lo 9 evals, hi 12 evals, 0 unconverged | 0 | 0 | 8 |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C2] | 0.2800 | 0.3104 | [0.2852, 0.3371] | sharp | no | v2 refined | [0.2980, 0.3256] | lo 8 evals, hi 8 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C3] | 0.3500 | 0.3578 | [0.3224, 0.3977] | sharp | yes | v2 refined | [0.3346, 0.3876] | lo 8 evals, hi 6 evals, 1 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C4] | 0.4300 | 0.4426 | [0.3963, 0.4965] | sharp | yes | v2 refined | [0.4103, 0.4857] | lo 8 evals, hi 6 evals, 1 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C2] | 3.2000 | 3.1890 | [3.0215, 3.3620] | sharp | yes | v2 refined | [3.1309, 3.2501] | lo 9 evals, hi 9 evals, 1 unconverged | 0 | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C3] | 3.0000 | 2.8847 | [2.6839, 3.0979] | sharp | yes | v2 refined | [2.7902, 2.9913] | lo 9 evals, hi 8 evals, 1 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C4] | 2.8000 | 2.8179 | [2.5487, 3.1039] | sharp | yes | v2 refined | [2.6423, 3.0112] | lo 12 evals, hi 8 evals, 0 unconverged | 2 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C2] | 0.6400 | 0.6581 | [0.6401, 0.6750] | sharp | no | v2 refined | [0.6533, 0.6629] | lo 12 evals, hi 11 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C3] | 0.6000 | 0.6130 | [0.5927, 0.6330] | sharp | yes | v2 refined | [0.6055, 0.6199] | lo 9 evals, hi 12 evals, 0 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C4] | 0.5700 | 0.5678 | [0.5492, 0.5866] | sharp | yes | v2 refined | [0.5610, 0.5729] | lo 9 evals, hi 10 evals, 0 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | kappa | 1.0000 | 1.0308 | [0.9158, 1.1564] | sharp | yes | v2 refined | [0.9451, 1.1305] | lo 8 evals, hi 6 evals, 0 unconverged | 2 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C2] | 0.0450 | 0.0455 | [0.0446, 0.0464] | sharp | yes | v2 refined | [0.0453, 0.0456] | lo 11 evals, hi 12 evals, 2 unconverged | 1 | 0 | 1 |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C3] | 0.0600 | 0.0595 | [0.0578, 0.0611] | sharp | yes | v2 refined | [0.0590, 0.0597] | lo 9 evals, hi 11 evals, 1 unconverged | 0 | 0 | 3 |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C4] | 0.0800 | 0.0802 | [0.0787, 0.0815] | sharp* | yes | v2 refined | [0.0796, 0.0803] | lo 9 evals, hi 12 evals, 1 unconverged | 1, interval rests on one | 0 | 6 |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C2] | 0.2800 | 0.3017 | [0.2785, 0.3261] | sharp | yes | v2 refined | [0.2901, 0.3153] | lo 8 evals, hi 8 evals, 3 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C3] | 0.3500 | 0.3302 | [0.2929, 0.3693] | sharp | yes | v2 refined | [0.3023, 0.3615] | lo 6 evals, hi 7 evals, 1 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C4] | 0.4300 | 0.4462 | [0.3936, 0.5047] | sharp* | yes | v2 refined | [0.4038, 0.4971] | lo 6 evals, hi 6 evals, 3 unconverged | 3, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C2] | 3.2000 | 3.1569 | [2.9880, 3.3332] | sharp | yes | v2 refined | [3.0961, 3.2236] | lo 9 evals, hi 9 evals, 0 unconverged | 1 | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C3] | 3.0000 | 2.8130 | [2.6014, 3.0337] | sharp | yes | v2 refined | [2.7039, 2.9276] | lo 8 evals, hi 8 evals, 1 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C4] | 2.8000 | 2.7088 | [2.4362, 3.0080] | sharp | yes | v2 refined | [2.5229, 2.9287] | lo 8 evals, hi 7 evals, 2 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C2] | 0.6400 | 0.6560 | [0.6401, 0.6711] | sharp | no | v2 refined | [0.6516, 0.6606] | lo 10 evals, hi 9 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C3] | 0.6000 | 0.5945 | [0.5687, 0.6176] | sharp | yes | v2 refined | [0.5821, 0.6038] | lo 8 evals, hi 10 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C4] | 0.5700 | 0.5761 | [0.5548, 0.5946] | sharp* | yes | v2 refined | [0.5633, 0.5831] | lo 7 evals, hi 9 evals, 2 unconverged | 2, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed2 | kappa | 1.0000 | 1.0229 | [0.8992, 1.1430] | sharp | yes | v2 refined | [0.9228, 1.1148] | lo 7 evals, hi 7 evals, 2 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C2] | 0.0450 | 0.0448 | [0.0439, 0.0457] | sharp | yes | v2 refined | [0.0447, 0.0450] | lo 10 evals, hi 11 evals, 0 unconverged | 0 | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C3] | 0.0600 | 0.0612 | [0.0601, 0.0624] | sharp* | no | v2 refined | [0.0610, 0.0614] | lo 10 evals, hi 11 evals, 2 unconverged | 1, interval rests on one | 0 | 3 |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C4] | 0.0800 | 0.0807 | [0.0794, 0.0819] | sharp | yes | v2 refined | [0.0802, 0.0808] | lo 9 evals, hi 12 evals, 2 unconverged | 0 | 0 | 6 |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C2] | 0.2800 | 0.2967 | [0.2729, 0.3223] | sharp | yes | v2 refined | [0.2841, 0.3116] | lo 10 evals, hi 9 evals, 3 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C3] | 0.3500 | 0.3638 | [0.3261, 0.4050] | sharp | yes | v2 refined | [0.3376, 0.3950] | lo 8 evals, hi 7 evals, 2 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C4] | 0.4300 | 0.4283 | [0.3740, 0.4879] | sharp | yes | v2 refined | [0.3822, 0.4837] | lo 7 evals, hi 6 evals, 3 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C2] | 3.2000 | 3.3082 | [3.1333, 3.4888] | sharp | yes | v2 refined | [3.2457, 3.3753] | lo 9 evals, hi 9 evals, 2 unconverged | 0 | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C3] | 3.0000 | 3.0850 | [2.8657, 3.3158] | sharp | yes | v2 refined | [2.9800, 3.2030] | lo 8 evals, hi 7 evals, 2 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C4] | 2.8000 | 2.6884 | [2.3969, 3.0113] | sharp | yes | v2 refined | [2.4779, 2.9493] | lo 7 evals, hi 6 evals, 0 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C2] | 0.6400 | 0.6374 | [0.6194, 0.6548] | sharp* | yes | v2 refined | [0.6320, 0.6431] | lo 9 evals, hi 9 evals, 3 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C3] | 0.6000 | 0.6066 | [0.5870, 0.6249] | sharp | yes | v2 refined | [0.5994, 0.6118] | lo 9 evals, hi 10 evals, 5 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C4] | 0.5700 | 0.5754 | [0.5541, 0.5938] | sharp* | yes | v2 refined | [0.5638, 0.5810] | lo 8 evals, hi 10 evals, 4 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed3 | kappa | 1.0000 | 1.0399 | [0.8240, 1.2526] | sharp* | yes | v2 refined | [0.8337, 1.2394] | lo 6 evals, hi 8 evals, 1 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C2] | 0.0450 | 0.0460 | [0.0448, 0.0470] | sharp | yes | v2 refined | [0.0457, 0.0461] | lo 10 evals, hi 11 evals, 0 unconverged | 1 | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C3] | 0.0600 | 0.0605 | [0.0591, 0.0617] | sharp | yes | v2 refined | [0.0601, 0.0606] | lo 9 evals, hi 11 evals, 0 unconverged | 0 | 0 | 5 |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C4] | 0.0800 | 0.0801 | [0.0781, 0.0819] | sharp | yes | v2 refined | [0.0795, 0.0803] | lo 10 evals, hi 11 evals, 2 unconverged | 0 | 0 | 8 |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C2] | 0.2800 | 0.2601 | [0.2385, 0.2834] | sharp | yes | v2 refined | [0.2485, 0.2746] | lo 10 evals, hi 10 evals, 1 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C3] | 0.3500 | 0.3302 | [0.2960, 0.3661] | sharp* | yes | v2 refined | [0.3059, 0.3565] | lo 8 evals, hi 7 evals, 3 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C4] | 0.4300 | 0.4155 | [0.3667, 0.4703] | sharp | yes | v2 refined | [0.3774, 0.4638] | lo 7 evals, hi 7 evals, 3 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C2] | 3.2000 | 3.2086 | [3.0286, 3.3992] | sharp | yes | v2 refined | [3.1345, 3.2828] | lo 9 evals, hi 9 evals, 2 unconverged | 1 | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C3] | 3.0000 | 2.8557 | [2.6418, 3.0813] | sharp | yes | v2 refined | [2.7462, 2.9737] | lo 10 evals, hi 8 evals, 1 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C4] | 2.8000 | 2.7591 | [2.4668, 3.0731] | sharp | yes | v2 refined | [2.5500, 3.0005] | lo 9 evals, hi 7 evals, 1 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C2] | 0.6400 | 0.6324 | [0.6121, 0.6521] | sharp | yes | v2 refined | [0.6254, 0.6406] | lo 9 evals, hi 8 evals, 2 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C3] | 0.6000 | 0.5916 | [0.5690, 0.6108] | sharp | yes | v2 refined | [0.5815, 0.5979] | lo 8 evals, hi 9 evals, 2 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C4] | 0.5700 | 0.5588 | [0.5357, 0.5811] | sharp | yes | v2 refined | [0.5488, 0.5676] | lo 9 evals, hi 9 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed4 | kappa | 1.0000 | 0.8908 | [0.7820, 1.0000] | sharp | yes | v2 refined | [0.7998, 0.9786] | lo 9 evals, hi 6 evals, 1 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C2] | 0.0450 | 0.0447 | [0.0419, 0.0474] | sharp* | yes | v2 refined | [0.0433, 0.0457] | lo 8 evals, hi 9 evals, 3 unconverged | 1, interval rests on one | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C3] | 0.0600 | 0.0601 | [0.0562, 0.0635] | sharp | yes | v2 refined | [0.0576, 0.0612] | lo 7 evals, hi 9 evals, 1 unconverged | 0 | 0 | 4 |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C4] | 0.0800 | 0.0839 | [0.0795, 0.0874] | sharp | yes | v2 refined | [0.0812, 0.0846] | lo 8 evals, hi 10 evals, 1 unconverged | 1 | 0 | 7 |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C2] | 0.2800 | 0.3429 | [0.2647, 0.4461] | wide* | yes | v2 refined | [0.2695, 0.4460] | lo 7 evals, hi 5 evals, 0 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C3] | 0.3500 | 0.3125 | [0.2382, 0.4129] | sharp* | yes | v2 refined | [0.2413, 0.4117] | lo 6 evals, hi 5 evals, 3 unconverged | 3, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C4] | 0.4300 | 0.5428 | [0.3768, 0.7577] | wide* | yes | v2 refined | [0.3775, 0.7492] | lo 5 evals, hi 6 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C2] | 3.2000 | 3.6187 | [3.0324, 4.2884] | sharp* | yes | v2 refined | [3.0457, 4.2597] | lo 6 evals, hi 6 evals, 1 unconverged | 2, interval rests on one | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C3] | 3.0000 | 3.1382 | [2.5650, 3.7779] | sharp | yes | v2 refined | [2.5894, 3.7435] | lo 6 evals, hi 6 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C4] | 2.8000 | 3.7864 | [2.6784, 5.2153] | wide* | yes | v2 refined | [2.6870, 5.1699] | lo 5 evals, hi 6 evals, 1 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C2] | 0.6400 | 0.6379 | [0.5867, 0.6884] | sharp* | yes | v2 refined | [0.5932, 0.6840] | lo 7 evals, hi 6 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C3] | 0.6000 | 0.5806 | [0.5273, 0.6304] | sharp | yes | v2 refined | [0.5304, 0.6232] | lo 6 evals, hi 6 evals, 4 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C4] | 0.5700 | 0.5979 | [0.5425, 0.6516] | sharp | yes | v2 refined | [0.5446, 0.6458] | lo 8 evals, hi 7 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed0 | kappa | 1.0000 | 0.7610 | [0.4872, 1.1176] | wide* | yes | v2 refined | [0.4899, 1.1118] | lo 5 evals, hi 6 evals, 0 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C2] | 0.0450 | 0.0456 | [0.0431, 0.0481] | sharp* | yes | v2 refined | [0.0444, 0.0465] | lo 8 evals, hi 9 evals, 1 unconverged | 1, interval rests on one | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C3] | 0.0600 | 0.0630 | [0.0584, 0.0672] | sharp | yes | v2 refined | [0.0602, 0.0645] | lo 8 evals, hi 8 evals, 1 unconverged | 0 | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C4] | 0.0800 | 0.0797 | [0.0755, 0.0833] | sharp | yes | v2 refined | [0.0765, 0.0806] | lo 8 evals, hi 10 evals, 1 unconverged | 0 | 0 | 8 |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C2] | 0.2800 | 0.3895 | [0.3060, 0.5065] | wide | no | v2 refined | [0.3106, 0.5064] | lo 7 evals, hi 4 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C3] | 0.3500 | 0.3789 | [0.2762, 0.5262] | wide | yes | v2 refined | [0.2793, 0.5225] | lo 6 evals, hi 5 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C4] | 0.4300 | 0.4776 | [0.3529, 0.6511] | wide | yes | v2 refined | [0.3562, 0.6462] | lo 8 evals, hi 5 evals, 0 unconverged | 2 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C2] | 3.2000 | 3.2617 | [2.7151, 3.9082] | sharp | yes | v2 refined | [2.7330, 3.8794] | lo 6 evals, hi 6 evals, 0 unconverged | 1 | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C3] | 3.0000 | 2.6507 | [2.0946, 3.3697] | sharp* | yes | v2 refined | [2.1276, 3.3547] | lo 6 evals, hi 6 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C4] | 2.8000 | 2.8518 | [2.0173, 3.8453] | sharp* | yes | v2 refined | [2.0224, 3.8204] | lo 6 evals, hi 7 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C2] | 0.6400 | 0.6943 | [0.6476, 0.7425] | sharp | no | v2 refined | [0.6555, 0.7348] | lo 6 evals, hi 6 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C3] | 0.6000 | 0.6450 | [0.5803, 0.6983] | sharp* | yes | v2 refined | [0.5819, 0.6927] | lo 6 evals, hi 6 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C4] | 0.5700 | 0.5675 | [0.5167, 0.6163] | sharp | yes | v2 refined | [0.5214, 0.6080] | lo 6 evals, hi 7 evals, 2 unconverged | 2 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed1 | kappa | 1.0000 | 1.1296 | [0.8109, 1.5120] | wide* | yes | v2 refined | [0.8152, 1.5007] | lo 6 evals, hi 6 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C2] | 0.0450 | 0.0463 | [0.0438, 0.0488] | sharp | yes | v2 refined | [0.0453, 0.0472] | lo 9 evals, hi 9 evals, 2 unconverged | 3 | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C3] | 0.0600 | 0.0587 | [0.0515, 0.0640] | sharp | yes | v2 refined | [0.0525, 0.0610] | lo 7 evals, hi 7 evals, 4 unconverged | 0 | 0 | 3 |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C4] | 0.0800 | 0.0804 | [0.0746, 0.0837] | sharp | yes | v2 refined | [0.0753, 0.0812] | lo 6 evals, hi 10 evals, 2 unconverged | 2 | 0 | 7 |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C2] | 0.2800 | 0.3480 | [0.2728, 0.4463] | wide* | yes | v2 refined | [0.2776, 0.4449] | lo 6 evals, hi 5 evals, 1 unconverged | 2, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C3] | 0.3500 | 0.2943 | [0.2030, 0.4184] | sharp | yes | v2 refined | [0.2037, 0.4159] | lo 5 evals, hi 6 evals, 0 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C4] | 0.4300 | 0.4852 | [0.2901, 0.7230] | wide* | yes | v2 refined | [0.2938, 0.7218] | lo 6 evals, hi 5 evals, 1 unconverged | 3, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C2] | 3.2000 | 3.1327 | [2.6096, 3.7461] | sharp* | yes | v2 refined | [2.6242, 3.7176] | lo 6 evals, hi 6 evals, 2 unconverged | 1, interval rests on one | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C3] | 3.0000 | 2.5104 | [1.9307, 3.1535] | sharp* | yes | v2 refined | [1.9602, 3.1260] | lo 6 evals, hi 8 evals, 2 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C4] | 2.8000 | 2.6058 | [1.7428, 3.8149] | sharp* | yes | v2 refined | [1.7549, 3.7973] | lo 5 evals, hi 6 evals, 3 unconverged | 3, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C2] | 0.6400 | 0.6844 | [0.6387, 0.7359] | sharp | yes | v2 refined | [0.6490, 0.7321] | lo 7 evals, hi 11 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C3] | 0.6000 | 0.5818 | [0.4874, 0.6608] | sharp | yes | v2 refined | [0.4925, 0.6530] | lo 6 evals, hi 9 evals, 3 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C4] | 0.5700 | 0.5854 | [0.4992, 0.6359] | sharp | yes | v2 refined | [0.5010, 0.6313] | lo 6 evals, hi 7 evals, 1 unconverged | 2 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed2 | kappa | 1.0000 | 1.0354 | [0.6886, 1.4074] | sharp* | yes | v2 refined | [0.6874, 1.4029] | lo 9 evals, hi 10 evals, 1 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C2] | 0.0450 | 0.0446 | [0.0418, 0.0473] | sharp | yes | v2 refined | [0.0432, 0.0457] | lo 9 evals, hi 10 evals, 0 unconverged | 0 | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C3] | 0.0600 | 0.0637 | [0.0603, 0.0672] | sharp | no | v2 refined | [0.0621, 0.0649] | lo 9 evals, hi 9 evals, 0 unconverged | 0 | 0 | 3 |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C4] | 0.0800 | 0.0819 | [0.0782, 0.0853] | sharp | yes | v2 refined | [0.0780, 0.0828] | lo 6 evals, hi 10 evals, 0 unconverged | 0 | 0 | 7 |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C2] | 0.2800 | 0.3214 | [0.2546, 0.4126] | sharp | yes | v2 refined | [0.2588, 0.4111] | lo 6 evals, hi 6 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C3] | 0.3500 | 0.3861 | [0.2825, 0.5258] | wide* | yes | v2 refined | [0.2858, 0.5226] | lo 6 evals, hi 5 evals, 1 unconverged | 2, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C4] | 0.4300 | 0.4125 | [0.2549, 0.6365] | sharp | yes | v2 refined | [0.2562, 0.6351] | lo 6 evals, hi 5 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C2] | 3.2000 | 3.5936 | [3.0253, 4.1918] | sharp | yes | v2 refined | [3.0346, 4.1745] | lo 5 evals, hi 5 evals, 0 unconverged | 1 | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C3] | 3.0000 | 3.3015 | [2.6408, 4.0975] | sharp | yes | v2 refined | [2.6780, 4.0674] | lo 9 evals, hi 8 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C4] | 2.8000 | 2.5011 | [1.7454, 3.6446] | sharp | yes | v2 refined | [1.7465, 3.6355] | lo 5 evals, hi 6 evals, 2 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C2] | 0.6400 | 0.6248 | [0.5769, 0.6777] | sharp | yes | v2 refined | [0.5860, 0.6730] | lo 9 evals, hi 6 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C3] | 0.6000 | 0.6144 | [0.5621, 0.6622] | sharp | yes | v2 refined | [0.5672, 0.6533] | lo 7 evals, hi 7 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C4] | 0.5700 | 0.5806 | [0.4906, 0.6283] | sharp | yes | v2 refined | [0.4971, 0.6225] | lo 7 evals, hi 6 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed3 | kappa | 1.0000 | 1.0353 | [0.4888, 1.6465] | wide* | yes | v2 refined | [0.4880, 1.6324] | lo 5 evals, hi 9 evals, 2 unconverged | 2, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C2] | 0.0450 | 0.0479 | [0.0443, 0.0512] | sharp* | yes | v2 refined | [0.0458, 0.0493] | lo 8 evals, hi 8 evals, 2 unconverged | 4, interval rests on one | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C3] | 0.0600 | 0.0615 | [0.0564, 0.0652] | sharp* | yes | v2 refined | [0.0581, 0.0627] | lo 8 evals, hi 9 evals, 3 unconverged | 4, interval rests on one | 0 | 5 |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C4] | 0.0800 | 0.0806 | [0.0738, 0.0864] | sharp* | yes | v2 refined | [0.0753, 0.0829] | lo 7 evals, hi 8 evals, 3 unconverged | 2, interval rests on one | 0 | 7 |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C2] | 0.2800 | 0.2279 | [0.1784, 0.2953] | sharp* | yes | v2 refined | [0.1813, 0.2953] | lo 6 evals, hi 8 evals, 4 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C3] | 0.3500 | 0.3013 | [0.2113, 0.4097] | sharp* | yes | v2 refined | [0.2115, 0.4074] | lo 5 evals, hi 6 evals, 0 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C4] | 0.4300 | 0.4005 | [0.2630, 0.5997] | sharp* | yes | v2 refined | [0.2648, 0.5996] | lo 5 evals, hi 5 evals, 1 unconverged | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C2] | 3.2000 | 3.2585 | [2.7174, 3.8253] | sharp | yes | v2 refined | [2.7305, 3.8030] | lo 5 evals, hi 6 evals, 1 unconverged | 0 | 0 | 2 |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C3] | 3.0000 | 2.6107 | [2.0010, 3.2590] | sharp | yes | v2 refined | [2.0322, 3.2285] | lo 6 evals, hi 6 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C4] | 2.8000 | 2.6774 | [1.7822, 3.8532] | sharp | yes | v2 refined | [1.7938, 3.8289] | lo 6 evals, hi 6 evals, 1 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C2] | 0.6400 | 0.6186 | [0.5587, 0.6857] | sharp | yes | v2 refined | [0.5592, 0.6850] | lo 5 evals, hi 9 evals, 1 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C3] | 0.6000 | 0.5776 | [0.4990, 0.6391] | sharp | yes | v2 refined | [0.5033, 0.6383] | lo 6 evals, hi 5 evals, 0 unconverged | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C4] | 0.5700 | 0.5435 | [0.4603, 0.6145] | sharp | yes | v2 refined | [0.4610, 0.6178] | lo 5 evals, hi 6 evals, 0 unconverged | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.15_seed4 | kappa | 1.0000 | 0.7129 | [0.4213, 1.0308] | wide* | yes | v2 refined | [0.4230, 1.0219] | lo 7 evals, hi 7 evals, 1 unconverged | 1, interval rests on one | 0 | 0 |

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
| wrong_phi | M1 | governed_by_criterion_4 | missing | missing | missing |
| wrong_phi | M2 | missing | missing | missing | missing |
| student_t | M1 | not_applicable | missing | missing | not_applicable |
| student_t | M2 | missing | missing | missing | not_applicable |
| piecewise_wear | M1 | not_applicable | missing | missing | not_applicable |
| piecewise_wear | M2 | missing | missing | missing | not_applicable |
| drop_early | M1 | not_applicable | missing | missing | not_applicable |
| drop_early | M2 | missing | missing | missing | not_applicable |
| traffic | M1 | not_applicable | missing | missing | not_applicable |
| traffic | M2 | missing | missing | missing | not_applicable |

### Details

Criterion 2 lists each seed's misspecification profile shape next to coverage, because an open interval side counts as covering; `*` marks a shape that rests on an unconverged profile point.

- **wrong_phi / M1**: C1 rho_c[C2]: governed_by_criterion_4; C1 rho_c[C3]: governed_by_criterion_4; C1 rho_c[C4]: governed_by_criterion_4; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing; C4 rho_c[C2]: missing; C4 rho_c[C3]: missing; C4 rho_c[C4]: missing.
- **wrong_phi / M2**: C1 rho_c[C2]: governed_by_criterion_4; C1 rho_c[C3]: governed_by_criterion_4; C1 rho_c[C4]: governed_by_criterion_4; C1 k_c[C2]: not_applicable; C1 k_c[C3]: missing; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: not_applicable; C2 k_c[C3]: missing; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing; C4 rho_c[C2]: missing; C4 rho_c[C3]: missing; C4 rho_c[C4]: missing.
- **student_t / M1**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: not_applicable; C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **student_t / M2**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: not_applicable; C1 k_c[C3]: missing; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: not_applicable; C2 k_c[C3]: missing; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **piecewise_wear / M1**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: not_applicable; C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **piecewise_wear / M2**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: not_applicable; C1 k_c[C3]: missing; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: not_applicable; C2 k_c[C3]: missing; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **drop_early / M1**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: not_applicable; C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **drop_early / M2**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: not_applicable; C1 k_c[C3]: missing; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: not_applicable; C2 k_c[C3]: missing; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **traffic / M1**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: not_applicable; C1 rho_c[C4]: not_applicable; C1 k_c[C2]: not_applicable; C1 k_c[C3]: not_applicable; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: not_applicable; C2 rho_c[C4]: not_applicable; C2 k_c[C2]: not_applicable; C2 k_c[C3]: not_applicable; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **traffic / M2**: C1 rho_c[C2]: not_applicable; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: not_applicable; C1 k_c[C3]: missing; C1 k_c[C4]: not_applicable; C2 rho_c[C2]: not_applicable; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: not_applicable; C2 k_c[C3]: missing; C2 k_c[C4]: not_applicable; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.

### Silent failures

- None found among the finished runs.

## Wall time

Measured wall time in seconds per run, read from the JSON records (`Observed`; machine and device in each record's `meta`). Runs without a record have no measured time, and no estimate is given for them.

| Part | Kind | Model | Runs | Median total s | Max total s | Sum total s | Max fit s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main | fit | M1 | 10 | 28.0 | 49.7 | 323.1 | 39.0 |
| main | fit | M2 | 10 | 15.6 | 31.8 | 186.3 | 26.2 |
| main | fit | M2v2 | 10 | 54.3 | 68.0 | 556.0 | 65.3 |
| main | profile | ode | 10 | 675.7 | 882.8 | 6899.5 | n/a |

Sum over all records: 7964.8 s. Slowest main-study M2 fit: 26.2 s (run main_fit_M2_well_specified_noise0.05_seed1); the threshold for M2 in the misspecification study is 600 s.

## P1 trigger evidence

These are mechanical trigger checks against PRD section 8, not the decision. The P1 decision is made by the user in `experiments/pivots.md`. Realistic noise for R1 is 0.15 s (`Assumed`, PLAN T5 method declarations).

- **R0 (plan)**: the default rung. It stays unless the user moves to R1 or R2.
- **R1 trigger** (`k_c` or `h_opt,c` non-identifiable at realistic noise), mechanical trigger check, not the decision: `trigger_condition_not_met`.
  Non-identifiable: none. Weakly identifiable: k_c[C2] (M1), k_c[C2] (M2), k_c[C3] (M1), k_c[C3] (M2), k_c[C4] (M1), k_c[C4] (M2).
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

- **E5** (refuted if a relative Gauss-Newton eigenvalue is below 1e-06 at noise 0.05 s), mechanical trigger check, not the decision: `not_refuted`. Values: main_profile_ode_well_specified_noise0.05_seed0 0.001, main_profile_ode_well_specified_noise0.05_seed1 0.002, main_profile_ode_well_specified_noise0.05_seed2 0.002, main_profile_ode_well_specified_noise0.05_seed3 7.73e-04, main_profile_ode_well_specified_noise0.05_seed4 0.001.
- **`M1` exploratory diagnostics** (PLAN T5, verbatim): **`M1` exploratory diagnostics** (declared 2026-11-01, after the v1 `M1` results were read; causes (d) and (e) added after the S1a slow test failure, before any diagnostic was run): runs in `experiments/s1_m1_diag/` test five candidate causes of the v1 `M1` failure (`k_c` collapsed on 10 of 10 main fits, and biased low in S1a with true states observed): (a) a longer budget (20000 steps), (b) the physics weight (thermal, wear and mono weights 0; or thermal and wear weights 10), (c) a bounded latent `h = tanh(tau * n_h)`, (d) a no-network least-squares check that fits `k_c` and `rho_c` directly from the true ODE states and rates, and (e) network smoothing and collocation resolution (`n_colloc` 128, or `m1_width` 128). They use seed 0 (seed 1 optional), noise 0.05 s, and otherwise the v1 `M1` setup. They are exploratory: they are not S1b evidence, they change no v1 or v2 result or label, and no `M1` method change is declared from them. Any later `M1` method version must be declared here before its runs are read. The P1 decision stays with the user in `experiments/pivots.md`. Their runs and mechanical checks are in `reports/s1_m1_diag.md`, built by `scripts/run_m1_diag.py`.

## Missing runs

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
