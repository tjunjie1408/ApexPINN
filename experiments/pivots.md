# Pivot log

Rule (PRD section 8): every pivot is logged here with the triggering evidence, the date, and the rung. Checkpoints are evaluated with recorded evidence. Each rung keeps JAX, the physics-informed model, F1 data, and the simulator.

This file is written by hand. `scripts/run_s1.py` never writes it.

## P1 (status: pending user review)

Checkpoint after milestone A1 (PLAN T5). The decision is the user's.

**Evidence**

- Report: [reports/s1_identifiability.md](../reports/s1_identifiability.md), section "P1 trigger evidence" (mechanical trigger checks, not the decision).
- Run records: [experiments/s1/](s1/) (one JSON record per run, with config hash and run metadata).

**Triggers (PRD section 8)**

| Rung | Model inside the simulator | Physics | Enter when | Observed evidence | Fires? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R0 (plan) | `M1` PINN | Thermal + wear (`PROJECT §4.4`) | Default | | |
| R1 | `M1` PINN | Wear-only latent state. Warm-up becomes an exponential `γ_c · e^(−τ/τ_w)` term with one global `τ_w` fitted on training folds. | **P1**: `S1c` marks `k_c` or `h_opt,c` non-identifiable at realistic noise; or **P2**: `k_c` profile likelihood on real data is flat (limit L1). | | |
| R2 | `M2` ODE as forecasting engine, `M1` reported as comparator | Same as current rung | **P1/P2**: `M1` fails to converge across seeds (loss-term ratio stalls over 3 seeds after the tuning budget is spent), *after* trying the remedies in risk `R4`. | | |

**Mechanical R2 reading**: the report's R2 indicator reads "loss-term ratio stalls" as declared in PLAN T5 (method declarations, "R2 indicator for P1"): an `M1` fit stalls when its objective fell by less than 1% over the last 20% of logged steps while its residual ratio stayed at or above 1.2. Recovery failures are shown as context only. The `R4` remedies and the tuning budget are for the user to confirm here.

**Context note**: `reports/data_probe.md` (warm-up visibility table, 10 development races) found a median all-stint gain of 0.000 s of the fastest scored lap over the first scored lap on 5 of the 10 races (2023_barcelona, 2024_bahrain, 2024_suzuka, 2024_spielberg, 2023_monza); the first scored lap is usually stint lap 2 because the out-lap is excluded (limit L1). Status `Inferred`: this informs how plausible R1 is on real data, but P1 is decided on the S1 evidence above.

**Decision** (check one)

- [ ] R0: stay on the plan.
- [ ] R1: wear-only latent state with an exponential warm-up term.
- [ ] R2: `M2` as the forecasting engine, `M1` as comparator.

**Rationale**:

**Date**:

**Reviewed by**:

## P2 (T13, not started)

Checkpoint after milestone A4 (real folds, PLAN T13). Not evaluated yet.
