# Exploratory M1 diagnostics (PLAN T5)

> Exploratory M1 diagnostics. Not S1b evidence; no S1b result, label or method changes from these runs.

Declared in PLAN T5 (verbatim):

- **`M1` exploratory diagnostics** (declared 2026-11-01, after the v1 `M1` results were read): runs in `experiments/s1_m1_diag/` separate three candidate causes of the v1 `M1` failure (`k_c` collapsed on 10 of 10 main fits): (a) a longer budget (20000 steps), (b) the physics weight (thermal, wear and mono weights 0; or thermal and wear weights 10), (c) a bounded latent `h = tanh(tau * n_h)`. They use seed 0 (seed 1 optional), noise 0.05 s, and otherwise the v1 `M1` setup. They are exploratory: they are not S1b evidence, they change no v1 or v2 result or label, and no `M1` method change is declared from them. Any later `M1` method version must be declared here before its runs are read. The P1 decision stays with the user in `experiments/pivots.md`.

Built by `apexpinn.eval.m1_diag_report` from the variant records in `experiments/s1_m1_diag/<variant>/` and the v1 `M1` records in `experiments/s1/` (baseline row). Every number is `Observed`, read from the record named in the Source column. Relative errors are (estimate - truth) / truth. The R2 stall flag applies the PLAN T5 R2 rule: objective fell by less than 1% over the last 20% of logged steps while the residual ratio stayed at or above 1.2. Gated recovered counts `rho_c` and `k_c` within their `O6` tolerance. `missing` marks a run without a record.

Variants:

- `budget_20k`: cause (a): 20000 training steps instead of 5000 (log_every 1000; n_warm 200 and n_ramp 800 unchanged).
- `physics_off`: cause (b): thermal, wear and mono weights 0, so only the data and reg terms train; k_c, rho_c and kappa then get no gradient and stay at the prior draw.
- `physics_x10`: cause (b): thermal and wear weights 10 instead of 1.
- `bounded_h`: cause (c): bounded latent h = tanh(tau * n_h) (PINN latent_mode 'tanh').

## Seed 0, noise 0.05 s

| Run | Steps | Weights | Latent | Status | Residual ratio | Gated recovered | `k_c` rel. error | `rho_c` rel. error | Objective change, last 20% | R2 stall flag | Final normalized data | Final normalized thermal | Thermal residual RMS | Fit wall time s | Any `k_c` within tolerance | Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| baseline v1 | 5000 | thermal 1, wear 1, mono 0.1 | linear | ok | 54.080 | 0 of 6 | C2 -99.5%, C3 -95.3%, C4 -96.3% | C2 +140.2%, C3 +462.2%, C4 +216.6% | -0.4861 | no | 0.023 | 0.009 | 0.083 | 30.7 | no | `experiments/s1/main_fit_M1_well_specified_noise0.05_seed0.json` |
| budget_20k | 20000 | thermal 1, wear 1, mono 0.1 | linear | ok | 7.818 | 0 of 6 | C2 -95.0%, C3 -93.5%, C4 -95.3% | C2 +1104.7%, C3 +323.3%, C4 +154.7% | -0.4965 | no | 5.69e-04 | 4.48e-04 | 0.019 | 115.4 | no | `experiments/s1_m1_diag/budget_20k/main_fit_M1_well_specified_noise0.05_seed0.json` |
| physics_off | 5000 | thermal 0, wear 0, mono 0 | linear | ok | 55.911 | 2 of 6 | C2 -60.4%, C3 +21.6%, C4 -8.7% | C2 +314.7%, C3 -23.5%, C4 +88.5% | -0.3941 | no | 0.021 | n/a | 0.908 | 16.7 | yes | `experiments/s1_m1_diag/physics_off/main_fit_M1_well_specified_noise0.05_seed0.json` |
| physics_x10 | 5000 | thermal 10, wear 10, mono 0.1 | linear | ok | 59.064 | 0 of 6 | C2 -99.0%, C3 -92.4%, C4 -92.5% | C2 +818.5%, C3 +683.8%, C4 +347.1% | -0.4259 | no | 0.026 | 0.001 | 0.032 | 30.3 | no | `experiments/s1_m1_diag/physics_x10/main_fit_M1_well_specified_noise0.05_seed0.json` |
| bounded_h | 5000 | thermal 1, wear 1, mono 0.1 | tanh | ok | 4.311 | 0 of 6 | C2 -90.6%, C3 -87.3%, C4 -89.7% | C2 +365.9%, C3 +274.3%, C4 +201.0% | -0.7469 | no | 0.002 | 0.001 | 0.006 | 27.2 | no | `experiments/s1_m1_diag/bounded_h/main_fit_M1_well_specified_noise0.05_seed0.json` |

## Mechanical reading

Mechanical checks per run, not an interpretation: whether the residual ratio fell below 1.2, and whether any `k_c` is within 25% of its truth.

- Seed 0, `baseline v1`: residual ratio below 1.2: no; any `k_c` within 25%: no; R2 stall flag: no.
- Seed 0, `budget_20k`: residual ratio below 1.2: no; any `k_c` within 25%: no; R2 stall flag: no.
- Seed 0, `physics_off`: residual ratio below 1.2: no; any `k_c` within 25%: yes; R2 stall flag: no.
- Seed 0, `physics_x10`: residual ratio below 1.2: no; any `k_c` within 25%: no; R2 stall flag: no.
- Seed 0, `bounded_h`: residual ratio below 1.2: no; any `k_c` within 25%: no; R2 stall flag: no.

The P1 decision is made by the user in `experiments/pivots.md`.

## How to run

```
uv run python scripts/run_m1_diag.py                       # every variant, seed 0
uv run python scripts/run_m1_diag.py --variant bounded_h   # one variant (repeatable)
uv run python scripts/run_m1_diag.py --seed 0 --seed 1     # seed 1 is optional
uv run python scripts/run_m1_diag.py --report-only         # rebuild this report
```
