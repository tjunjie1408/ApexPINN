"""Write the research-record figures and derived numbers from the existing JSON records.

Reads experiments/data_probe/probe_results.json, the S1 records in experiments/s1/ (full config) and
experiments/s1_smoke/ (smoke config), and configs/decisions_v1.yaml, and builds the full-config synthetic
dataset make_synthetic_dataset(jax.random.key(0)). It never changes a record or a report. Run from the
repository root:

    uv run python scripts/make_figures.py                      # every figure, into reports/figures/
    uv run python scripts/make_figures.py --only s1_profiles   # one figure (repeatable)

Outputs in reports/figures/: probe_warmup.png, probe_pace_pit.png, synthetic_stints.png,
s1_fit_convergence.png, s1_recovery.png, s1_recovery_seeds.png, s1_profiles.png, s1_intervals.png,
s1_gn_spectrum.png, s1_misspec_smoke.png, and derived_numbers.json (per-parameter interval numbers with
the interval method, the deviance at truth, source run ids and config hashes, generated_utc).

Fit figures take M1 and M2 from one folder: experiments/s1/ (full config) if it holds any finished
main fit, else experiments/s1_smoke/. A model with no record in that folder is left out rather than filled
in from the other folder, so a full-config fit is never paired with a smoke fit. s1_recovery_seeds.png
shows every seed and noise level of M1, M2 and M2v2 from one folder, chosen by the same rule. The profile
record (the MLE bars in s1_recovery.png) is picked on its own, full before smoke. Every title and recovery
legend entry names its config, and smoke figures say 'smoke'. The reported S1c interval is interval
method v2 (refined) for profile records with interval_method 'v2_refined' and the v1 linear interpolation
for older records; for v2 records the v1 interval is drawn and written as Superseded. The quadratic
interval in s1_profiles.png, s1_intervals.png and derived_numbers.json is a post-hoc diagnostic.

matplotlib must be a direct dependency: run `uv add matplotlib` once. A figure whose input is missing is
skipped with a reason; a figure whose build raises any other exception is reported as failed. The exit
code is 1 if any artifact failed or none was written, else 0.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # before any apexpinn module creates arrays

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from apexpinn.eval.figures import ARTIFACT_NAMES, build_figures  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root (default: .)")
    parser.add_argument(
        "--out", type=Path, default=Path("reports/figures"), help="output folder, relative to --root"
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=ARTIFACT_NAMES,
        metavar="NAME",
        help="build only NAME (repeatable)",
    )
    args = parser.parse_args(argv)
    results = build_figures(args.root, args.out, only=args.only, log=print)
    outcomes = list(results.values())
    if any(r.startswith("failed") for r in outcomes):
        return 1
    return 0 if any(r.startswith("written") for r in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
