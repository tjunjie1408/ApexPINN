# S1 identifiability study (PLAN T5), full config

Built from 2 run records in `experiments/s1/` by `apexpinn.eval.s1_report`. Every number is `Observed`, read from the record named by its run id. Labels and criterion outcomes apply the rules declared in PLAN T5 to those numbers. Config: 48 stints, compounds ['C2', 'C3', 'C4'], seeds [0, 1, 2, 3, 4], noise [0.05, 0.15] s, models ['M1', 'M2'].

## S1b recovery

Relative error (estimate - truth) / truth of the trained `M1` and `M2` estimates, lap time only, initialized from the seed's prior draw. Recovered means |relative error| within the `O6` tolerance (`k_c` 25%, `rho_c` 15%) for every gated parameter. Residual ratio is residual RMS on scored laps / noise_s. `Observed`.

| Run id | Model | Noise s | Seed | Status | rho_c[C2] | rho_c[C3] | rho_c[C4] | k_c[C2] | k_c[C3] | k_c[C4] | Recovered | Max abs err, ungated | Residual ratio | Collapse flags |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_fit_M1_well_specified_noise0.05_seed0 | M1 | 0.05 | 0 | ok | +140.2% | +462.2% | +216.6% | -99.5% | -95.3% | -96.3% | no | +117.8% | 54.080 | none |
| main_fit_M1_well_specified_noise0.05_seed1 | M1 | 0.05 | 1 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.05_seed2 | M1 | 0.05 | 2 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.05_seed3 | M1 | 0.05 | 3 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.05_seed4 | M1 | 0.05 | 4 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed0 | M1 | 0.15 | 0 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed1 | M1 | 0.15 | 1 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed2 | M1 | 0.15 | 2 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed3 | M1 | 0.15 | 3 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed4 | M1 | 0.15 | 4 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed0 | M2 | 0.05 | 0 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed1 | M2 | 0.05 | 1 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed2 | M2 | 0.05 | 2 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed3 | M2 | 0.05 | 3 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed4 | M2 | 0.05 | 4 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed0 | M2 | 0.15 | 0 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed1 | M2 | 0.15 | 1 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed2 | M2 | 0.15 | 2 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed3 | M2 | 0.15 | 3 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed4 | M2 | 0.15 | 4 | missing | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

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
| k_c[C2] | 0.05 | M1 | **missing** | sharp | no |
| k_c[C2] | 0.05 | M2 | **missing** | sharp |  |
| k_c[C3] | 0.05 | M1 | **missing** | sharp | no |
| k_c[C3] | 0.05 | M2 | **missing** | sharp |  |
| k_c[C4] | 0.05 | M1 | **missing** | sharp | no |
| k_c[C4] | 0.05 | M2 | **missing** | sharp |  |
| kappa | 0.05 | profile | **missing** | sharp | n/a |
| rho_c[C2] | 0.05 | M1 | **missing** | sharp | no |
| rho_c[C2] | 0.05 | M2 | **missing** | sharp |  |
| rho_c[C3] | 0.05 | M1 | **missing** | sharp | no |
| rho_c[C3] | 0.05 | M2 | **missing** | sharp |  |
| rho_c[C4] | 0.05 | M1 | **missing** | sharp | no |
| rho_c[C4] | 0.05 | M2 | **missing** | sharp |  |
| gamma_c[C2] | 0.15 | profile | **missing** | n/a | n/a |
| gamma_c[C3] | 0.15 | profile | **missing** | n/a | n/a |
| gamma_c[C4] | 0.15 | profile | **missing** | n/a | n/a |
| h_opt_c[C2] | 0.15 | profile | **missing** | n/a | n/a |
| h_opt_c[C3] | 0.15 | profile | **missing** | n/a | n/a |
| h_opt_c[C4] | 0.15 | profile | **missing** | n/a | n/a |
| k_c[C2] | 0.15 | M1 | **missing** | n/a |  |
| k_c[C2] | 0.15 | M2 | **missing** | n/a |  |
| k_c[C3] | 0.15 | M1 | **missing** | n/a |  |
| k_c[C3] | 0.15 | M2 | **missing** | n/a |  |
| k_c[C4] | 0.15 | M1 | **missing** | n/a |  |
| k_c[C4] | 0.15 | M2 | **missing** | n/a |  |
| kappa | 0.15 | profile | **missing** | n/a | n/a |
| rho_c[C2] | 0.15 | M1 | **missing** | n/a |  |
| rho_c[C2] | 0.15 | M2 | **missing** | n/a |  |
| rho_c[C3] | 0.15 | M1 | **missing** | n/a |  |
| rho_c[C3] | 0.15 | M2 | **missing** | n/a |  |
| rho_c[C4] | 0.15 | M1 | **missing** | n/a |  |
| rho_c[C4] | 0.15 | M2 | **missing** | n/a |  |

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

| Run id | Parameter | Truth | MLE | 95% interval | Shape | Covers truth | Unconverged points | Failed points | At box edge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C2] | 0.0450 | 0.0448 | [0.0446, 0.0450] | sharp* | no | 1, interval rests on one | 0 | 1 |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C3] | 0.0600 | 0.0600 | [0.0597, 0.0601] | sharp* | yes | 1, interval rests on one | 0 | 4 |
| main_profile_ode_well_specified_noise0.05_seed0 | rho_c[C4] | 0.0800 | 0.0813 | [0.0808, 0.0814] | sharp* | no | 1, interval rests on one | 0 | 7 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C2] | 0.2800 | 0.2989 | [0.2864, 0.3146] | sharp | no | 0 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C3] | 0.3500 | 0.3356 | [0.3172, 0.3587] | sharp* | yes | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | k_c[C4] | 0.4300 | 0.4595 | [0.4079, 0.5106] | sharp | yes | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | gamma_c[C2] | 3.2000 | 3.3209 | [3.2532, 3.3903] | sharp* | no | 1, interval rests on one | 0 | 2 |
| main_profile_ode_well_specified_noise0.05_seed0 | gamma_c[C3] | 3.0000 | 3.0420 | [2.9493, 3.1412] | sharp* | yes | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | gamma_c[C4] | 2.8000 | 3.0553 | [2.8344, 3.3166] | sharp | no | 1 | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | h_opt_c[C2] | 0.6400 | 0.6387 | [0.6336, 0.6451] | sharp* | yes | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | h_opt_c[C3] | 0.6000 | 0.5926 | [0.5864, 0.5977] | sharp* | no | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | h_opt_c[C4] | 0.5700 | 0.5782 | [0.5687, 0.5847] | sharp* | yes | 4, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed0 | kappa | 1.0000 | 0.9181 | [0.8345, 1.0028] | sharp* | yes | 1, interval rests on one | 0 | 0 |
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed1 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed2 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed3 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.05_seed4 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed0 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed1 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed2 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed3 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | rho_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | k_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | gamma_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C2] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C3] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | h_opt_c[C4] | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |
| main_profile_ode_well_specified_noise0.15_seed4 | kappa | n/a | n/a | n/a | missing | n/a | n/a | n/a | n/a |

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

**M2 in the misspecification study**: undecided (10 of 10 main M2 fit records are missing or failed); the M1 runs proceed, and the M2 misspecification runs are planned once every main-study M2 fit has finished.

### Outcomes

| Variant | Model | Criterion 1 | Criterion 2 | Criterion 3 | Criterion 4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| wrong_phi | M1 | missing | missing | missing | missing |
| student_t | M1 | missing | missing | missing | not_applicable |
| piecewise_wear | M1 | missing | missing | missing | not_applicable |
| drop_early | M1 | missing | missing | missing | not_applicable |
| traffic | M1 | missing | missing | missing | not_applicable |

### Details

Criterion 2 lists each seed's misspecification profile shape next to coverage, because an open interval side counts as covering; `*` marks a shape that rests on an unconverged profile point.

- **wrong_phi / M1**: C1 rho_c[C2]: governed_by_criterion_4; C1 rho_c[C3]: governed_by_criterion_4; C1 rho_c[C4]: governed_by_criterion_4; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing; C4 rho_c[C2]: missing; C4 rho_c[C3]: missing; C4 rho_c[C4]: missing.
- **student_t / M1**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **piecewise_wear / M1**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **drop_early / M1**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.
- **traffic / M1**: C1 rho_c[C2]: missing; C1 rho_c[C3]: missing; C1 rho_c[C4]: missing; C1 k_c[C2]: missing; C1 k_c[C3]: missing; C1 k_c[C4]: missing; C2 rho_c[C2]: missing; C2 rho_c[C3]: missing; C2 rho_c[C4]: missing; C2 k_c[C2]: missing; C2 k_c[C3]: missing; C2 k_c[C4]: missing; C2 gamma_c[C2]: missing; C2 gamma_c[C3]: missing; C2 gamma_c[C4]: missing; C2 h_opt_c[C2]: missing; C2 h_opt_c[C3]: missing; C2 h_opt_c[C4]: missing; C2 kappa: missing; C3: missing.

### Silent failures

- None found among the finished runs.

## Wall time

Measured wall time in seconds per run, read from the JSON records (`Observed`; machine and device in each record's `meta`). Runs without a record have no measured time, and no estimate is given for them.

| Part | Kind | Model | Runs | Median total s | Max total s | Sum total s | Max fit s |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main | fit | M1 | 1 | 40.4 | 40.4 | 40.4 | 30.7 |
| main | profile | ode | 1 | 255.2 | 255.2 | 255.2 | n/a |

Sum over all records: 295.6 s. Slowest main-study M2 fit: n/a (no measured main-study M2 fit); the threshold for M2 in the misspecification study is 600 s.

## P1 trigger evidence

These are mechanical trigger checks against PRD section 8, not the decision. The P1 decision is made by the user in `experiments/pivots.md`. Realistic noise for R1 is 0.15 s (`Assumed`, PLAN T5 method declarations).

- **R0 (plan)**: the default rung. It stays unless the user moves to R1 or R2.
- **R1 trigger** (`k_c` or `h_opt,c` non-identifiable at realistic noise), mechanical trigger check, not the decision: `incomplete`.
  Non-identifiable: none. Weakly identifiable: none.
- **R2 indicator** (`M1` stalls on at least 3 seeds at one noise level, with the `M2` records of those seeds present), mechanical trigger check, not the decision: `incomplete`. Declared rule (PLAN T5, verbatim): **R2 indicator for P1** (mechanical, not the decision): an `M1` fit stalls when its objective fell by less than 1% over the last 20% of logged steps while its residual ratio stayed at or above 1.2, so the data term did not reach the noise level. The indicator is present when `M1` stalls on at least 3 seeds (all seeds if fewer) at one noise level and the `M2` records of those seeds exist. `M1` and `M2` recovery and the final normalized loss terms are shown as context. The `R4` remedies and the tuning budget of `PRD §8` are not checked by code.

| Run id | Model | Noise s | Seed | Status | Stalled | Objective change, last 20% | Residual ratio | Final normalized terms | Not recovered |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| main_fit_M1_well_specified_noise0.05_seed0 | M1 | 0.05 | 0 | ok | no | -0.4861 | 54.080 | data 0.023, mono 0.000, reg 6.255, thermal 0.009, wear 2.21e-04 | rho_c[C2], rho_c[C3], rho_c[C4], k_c[C2], k_c[C3], k_c[C4] |
| main_fit_M1_well_specified_noise0.05_seed1 | M1 | 0.05 | 1 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.05_seed2 | M1 | 0.05 | 2 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.05_seed3 | M1 | 0.05 | 3 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.05_seed4 | M1 | 0.05 | 4 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed0 | M2 | 0.05 | 0 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed1 | M2 | 0.05 | 1 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed2 | M2 | 0.05 | 2 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed3 | M2 | 0.05 | 3 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.05_seed4 | M2 | 0.05 | 4 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed0 | M1 | 0.15 | 0 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed1 | M1 | 0.15 | 1 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed2 | M1 | 0.15 | 2 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed3 | M1 | 0.15 | 3 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M1_well_specified_noise0.15_seed4 | M1 | 0.15 | 4 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed0 | M2 | 0.15 | 0 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed1 | M2 | 0.15 | 1 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed2 | M2 | 0.15 | 2 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed3 | M2 | 0.15 | 3 | missing | n/a | n/a | n/a | n/a | n/a |
| main_fit_M2_well_specified_noise0.15_seed4 | M2 | 0.15 | 4 | missing | n/a | n/a | n/a | n/a | n/a |

- **E5** (refuted if a relative Gauss-Newton eigenvalue is below 1e-06 at noise 0.05 s), mechanical trigger check, not the decision: `incomplete`. Values: main_profile_ode_well_specified_noise0.05_seed0 0.001, main_profile_ode_well_specified_noise0.05_seed1 n/a, main_profile_ode_well_specified_noise0.05_seed2 n/a, main_profile_ode_well_specified_noise0.05_seed3 n/a, main_profile_ode_well_specified_noise0.05_seed4 n/a.

## Missing runs

M2 misspecification runs are listed once every main-study M2 fit has a record.

- main_fit_M1_well_specified_noise0.05_seed1: no record
- main_fit_M1_well_specified_noise0.05_seed2: no record
- main_fit_M1_well_specified_noise0.05_seed3: no record
- main_fit_M1_well_specified_noise0.05_seed4: no record
- main_fit_M1_well_specified_noise0.15_seed0: no record
- main_fit_M1_well_specified_noise0.15_seed1: no record
- main_fit_M1_well_specified_noise0.15_seed2: no record
- main_fit_M1_well_specified_noise0.15_seed3: no record
- main_fit_M1_well_specified_noise0.15_seed4: no record
- main_fit_M2_well_specified_noise0.05_seed0: no record
- main_fit_M2_well_specified_noise0.05_seed1: no record
- main_fit_M2_well_specified_noise0.05_seed2: no record
- main_fit_M2_well_specified_noise0.05_seed3: no record
- main_fit_M2_well_specified_noise0.05_seed4: no record
- main_fit_M2_well_specified_noise0.15_seed0: no record
- main_fit_M2_well_specified_noise0.15_seed1: no record
- main_fit_M2_well_specified_noise0.15_seed2: no record
- main_fit_M2_well_specified_noise0.15_seed3: no record
- main_fit_M2_well_specified_noise0.15_seed4: no record
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
