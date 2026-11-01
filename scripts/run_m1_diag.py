"""Run the exploratory M1 diagnostics (PLAN T5 method declarations, `M1` exploratory diagnostics).

Each variant is the v1 M1 setup with one change (budget_20k, physics_off, physics_x10, bounded_h), fitted
at noise 0.05 s on seed 0 (seed 1 optional). Records go to experiments/s1_m1_diag/<variant>/ and the
report to reports/s1_m1_diag.md, rebuilt at the end of every call. These runs are exploratory: they are
not S1b evidence and change no S1b result, label or method. Run from the repository root:

    uv run python scripts/run_m1_diag.py                        # every variant, seed 0
    uv run python scripts/run_m1_diag.py --variant bounded_h    # one variant (repeatable)
    uv run python scripts/run_m1_diag.py --seed 0 --seed 1      # add the optional seed 1
    uv run python scripts/run_m1_diag.py --report-only          # rebuild the report only

A finished record is skipped on a rerun; a record whose config_hash differs stops its variant unless --force,
and the other variants still run. Exit code 2 when any variant was refused (or the arguments are invalid),
1 when any run errored, else 0.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # before any apexpinn module creates arrays

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from apexpinn.eval.m1_diag_report import build_m1_diag_report  # noqa: E402
from apexpinn.train.m1_diag import DEFAULT_SEEDS, VARIANTS, run_diag, summary_line  # noqa: E402
from apexpinn.train.s1_study import DEFAULT_DECISIONS  # noqa: E402


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"seed must be an integer, got {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"seed must be >= 0, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant", action="append", choices=tuple(VARIANTS), help="variant to run (repeatable; default all)"
    )
    parser.add_argument("--seed", action="append", type=_seed, help="seed to run (repeatable; default 0)")
    parser.add_argument("--force", action="store_true", help="rerun runs whose recorded config_hash differs")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from existing JSON")
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root (default: .)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    local = root / "configs" / "decisions_v1.yaml"
    decisions = local if local.is_file() else DEFAULT_DECISIONS
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    code = 0
    if not args.report_only:
        try:
            summaries = run_diag(
                args.variant, seeds, root=root, force=args.force, log=print, decisions=decisions
            )
        except ValueError as exc:  # invalid variants or seeds, before anything runs
            print(f"stopped: {exc}", file=sys.stderr)
            code = 2
        else:  # run_diag has printed each variant's summary as soon as it finished
            refused = [name for name, s in summaries.items() if "refused" in s]
            for name in refused:
                print(summary_line(name, summaries[name]), file=sys.stderr)
            if refused:
                code = 2
            else:
                code = 1 if any(s["errors"] for s in summaries.values()) else 0
    report = build_m1_diag_report(root, seeds, decisions)
    print(f"report: {report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
