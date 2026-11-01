# Pivot log

Rule (PRD section 8): every pivot is logged here with the triggering evidence, the date, and the rung. Checkpoints are evaluated with recorded evidence. Each rung keeps JAX, the physics-informed model, F1 data, and the simulator.

This file is written by hand. `scripts/run_s1.py` never writes it.

## P1 (status: decided, R2)

Checkpoint after milestone A1 (PLAN T5). The decision is the user's.

**Evidence**

- Report: [reports/s1_identifiability.md](../reports/s1_identifiability.md), sections "S1b recovery", "S1c identifiability", "S1d misspecification", and "P1 trigger evidence" (built from all 85 S1 study records: 40 main + 45 misspecification).
- Exploratory `M1` diagnostics report: [reports/s1_m1_diag.md](../reports/s1_m1_diag.md) (built from all 7 diagnostic variants in [experiments/s1_m1_diag/](s1_m1_diag/)).
- Run records: [experiments/s1/](s1/) (one JSON record per run, with config hash and run metadata) and [RECORD.md](../RECORD.md) sections 7, 7a, 8, and 9.

**Triggers (PRD section 8)**

| Rung | Model inside the simulator | Physics | Enter when | Observed evidence | Fires? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R0 (plan) | `M1` PINN | Thermal + wear (`PROJECT §4.4`) | Default | `M1` recovers 0 of 10 main fits in `S1b` (`k_c` collapses -89.2% to -99.5% on all 30 compound estimates; residual ratios 3.270 to 227.347), fails `S1a` even with true states observed (`test_s1a_recovers_k_and_rho_with_states_observed`, `@pytest.mark.xfail(strict=True)`), and recovers 0 of 6 gated parameters across all 5 network-with-physics diagnostic variants in `reports/s1_m1_diag.md`. | Replaced by R2 |
| R1 | `M1` PINN | Wear-only latent state. Warm-up becomes an exponential `γ_c · e^(−τ/τ_w)` term with one global `τ_w` fitted on training folds. | **P1**: `S1c` marks `k_c` or `h_opt,c` non-identifiable at realistic noise; or **P2**: `k_c` profile likelihood on real data is flat (limit L1). | `reports/s1_identifiability.md`: mechanical R1 check is `trigger_condition_not_met`. Across all 10 main profiles (`v2 refined`), 0 of 13 parameters are `non-identifiable` at either 0.05 s or 0.15 s (0 `flat` profiles out of 130; 123 of 130 intervals cover the truth, 94.6%); at 0.15 s all three `h_opt_c` are `identifiable` (`sharp` on all 5 seeds) and all three `k_c` are `weakly_identifiable`; E5 is `not_refuted` (smallest relative Gauss-Newton eigenvalues `7.73e-4` to `0.002` at 0.05 s). | No |
| R2 | `M2` ODE as forecasting engine, `M1` reported as comparator | Same as current rung (`PROJECT §4.4`) | **P1/P2**: `M1` fails to converge across seeds (loss-term ratio stalls over 3 seeds after the tuning budget is spent), *after* trying the remedies in risk `R4`. | All `R4` remedies (init loss normalization, gradient-norm logging, hard ICs, curriculum warm-up and ramp, float64) and 5 exploratory `M1` variants (`budget_20k`, `physics_x10`, `bounded_h`, `colloc_128`, `width_128`) leave `M1` at 0 of 6 recovered (`k_c` -87.3% to -99.3%, with `width_128` triggering the mechanical stall flag), while closed-form least squares on the true ODE states (`true_state_ls`) recovers 6 of 6 (`0.0%` error, ratio `1.000`). On the same data, `M2` v1 recovers 5 of 10 fits in 15.6 s median and `M2v2` (3 prior starts) recovers 8 of 10 fits (48 of 48 gated parameters on seeds 1 to 4) in 54.3 s median. | **Yes (selected)** |

**Mechanical R2 reading**: the report's R2 indicator reads "loss-term ratio stalls" as declared in PLAN T5 (method declarations, "R2 indicator for P1"): an `M1` fit stalls when its objective fell by less than 1% over the last 20% of logged steps while its residual ratio stayed at or above 1.2. On the 10 v1 `M1` fits that mechanical indicator reads `indicator_absent` because the physics and regularization terms were still shifting (`-16.6%` to `-75.0%` over the last 20%) while the fits remained far above the noise floor (residual ratios `3.270` to `227.347`); on the `width_128` diagnostic variant the mechanical stall indicator fires (`+2.28%` over the last 20% at residual ratio `14.202`). Recovery failures and the `R4` remedies / diagnostic variants are evaluated here by user judgment.

**Context note**: `reports/data_probe.md` (warm-up visibility table, 10 development races) found a median all-stint gain of 0.000 s of the fastest scored lap over the first scored lap on 5 of the 10 races (2023_barcelona, 2024_bahrain, 2024_suzuka, 2024_spielberg, 2023_monza); the first scored lap is usually stint lap 2 because the out-lap is excluded (limit L1). Status `Inferred`: this informs how plausible R1 is on real data at checkpoint P2, whereas P1 is decided on the S1 evidence above.

**Decision** (check one)

- [ ] R0: stay on the plan.
- [ ] R1: wear-only latent state with an exponential warm-up term.
- [x] R2: `M2` as the forecasting engine, `M1` as comparator.

**Rationale**: `S1c` proves the thermal + wear ODE (`PROJECT §4.4`) is identifiable from lap times (`0` non-identifiable parameters at both `0.05` s and `0.15` s, `E5 not_refuted`, `R1 trigger_condition_not_met`, and `M2v2` recovers 8 of 10 fits and 48 of 48 gated parameters across seeds 1 to 4), so simplifying the physics (`R1`) is unwarranted at P1. Conversely, `M1` (soft-constraint PINN collocation) fails on 10 of 10 main fits in `S1b`, fails `S1a` even when true states `(h, w)` are observed, and fails on all 5 network-with-physics diagnostic variants (`budget_20k`, `physics_x10`, `bounded_h`, `colloc_128`, `width_128`), while direct least squares on the true states (`true_state_ls`) recovers all 6 gated parameters to `0.0%` error. Entering **R2** keeps the full thermal + wear ODE physics, promotes `M2` (differentiable Diffrax ODE model) to the primary forecasting engine for the simulator, and retains `M1` as a comparator (`--part misspec` and real-data folds still evaluate `M1` as comparator).

**Date**: 2026-11-01

**Reviewed by**: User (confirmed in session)

## P2 (T13, not started)

Checkpoint after milestone A4 (real folds, PLAN T13). Not evaluated yet.
