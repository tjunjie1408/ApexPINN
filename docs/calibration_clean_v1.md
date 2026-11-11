# Clean calibration pilot v1

Declared 2026-11-06 after inspecting T13 v1 F1/F2 results. This is a post-hoc extension,
not a retrospective preregistration or a change to the running F3 experiment. Canonical
settings are in `configs/calibration_clean_v1.yaml`. No pilot result has been read.

For outer group i, reserve i+1 for tuning validation and i+2 for calibration, cyclically.
The default F3 pilot uses F4 for tuning validation and F5 for calibration. Tuning fits
and normalization use F1/F2/F6 only. After selecting one of the same 20 M2 configurations,
fit five final members on F1/F2/F4/F6, excluding both calibration F5 and outer-test F3.
The same five members predict calibration and test data. Never refit them on F5.
There are 25 new fits, 1000 steps each, in separate checkpoint folders. Failures retain
their slots. This does not extend any completed T13 v1 trial budget.

The raw distribution uses each final member's training-residual noise. It does not
estimate noise from calibration labels before calibrating. For every horizon, obtain
the raw mixture's central 90% bounds l,u and unchanged mean mu. Score a calibration
endpoint y as (mu-y)/(mu-l) if y < mu, otherwise (y-mu)/(u-mu). Give each event equal
total weight, then each eligible origin within that event equal weight. The scale is
the weighted empirical 0.90 quantile, floored at 1e-6 only to avoid a degenerate scale.
This is empirical recalibration, without a finite-sample coverage guarantee for dependent
laps or unseen circuits. No density-ratio estimation or conformal guarantee is claimed.

Apply scale a_h to the entire mixture: member means become mu+a_h*(mu_e-mu), and member
standard deviations become a_h*sigma_e. The mean stays fixed and exact mixture variance
becomes a_h^2 times its raw value. Report exact-mixture CRPS and 50%/90% intervals, widths
and coverage, alongside unchanged RMSE/MAE, by horizon and outer-test circuit. Scaling
can shrink or widen intervals. It is not selected using outer-test outcomes.

Acceptance of implementation means disjoint roles, no label leakage, fixed fit budgets,
hash-bound checkpoints/recipe/forecasts, unchanged paired means and exact mixture moments.
Scientific success is not guaranteed and has no new numerical pass threshold. Always
report unfavorable outcomes. Additional context-dependent calibration is not in this pilot.

Raw-versus-calibrated comparisons isolate postprocessing within this pilot. Comparisons
against T13 v1 also change the training population and raw-noise estimator, so they are
not a calibration-only treatment effect. F1/F2 are already inspected development results;
2025 remains untouched. H2, G2 and P2 remain separate.

Execution is user-launched on hosted Colab L4/float64/Python 3.12 only. Wait for the current
F3 run to finish before starting the pilot in that runtime. Preparing or uploading this
notebook does not launch training. Resource use and duration remain Unknown until measured.
