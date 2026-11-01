"""Run the S1 identifiability study (PLAN T5: S1b recovery, S1c profiles, S1d misspecification).

Each run writes one JSON record; a rerun skips finished runs, so the study can be stopped and resumed.
The report is rebuilt from the records at the end of every call. Run from the repository root:

    uv run python scripts/run_s1.py --config smoke     # pipeline check, writes experiments/s1_smoke/
    uv run python scripts/run_s1.py --config full      # the study, writes experiments/s1/

Local overnight run: start the full command and leave it; rerun the same command to resume after an
interruption.

Kaggle or AWS CPU: clone the repository, install uv, run `uv sync`, then the same command. uv installs
Python 3.12 itself, so the Python version of the image does not matter. To resume on another machine,
copy experiments/s1/ back into the checkout before rerunning; finished runs are skipped.

Sharding across machines: run `--part main --shard i/n` on n machines (i = 0..n-1), copy every
machine's experiments/s1/*.json into one folder, then run `--part misspec` (sharded the same way if
wanted). M2 joins the misspecification study only if the slowest main M2 fit took under 600 s, so while
any main M2 fit record is missing or failed, part misspec runs only the M1 and profile runs; rerun it
after every main M2 fit has finished to add the M2 runs. `--report-only` rebuilds the report from the
records.

Changed configs: every record stores the config_hash of the settings and code it ran with. A finished
record whose config_hash differs (for example after a code change that alters the profile method, the
synthetic generator or the residual equations) stops the run unless --force reruns it. Records with
status error never block and are rerun.

No full-study wall time has been measured yet. The report's wall-time table shows measured times only.
Before committing a machine overnight, time one full profile run (the record is kept and reused):

    uv run python scripts/run_s1.py --config full --part main --only <run_id>

with run_id main_profile_ode_well_specified_noise0.05_seed0, then read wall_time_s in its JSON record.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # before any apexpinn module creates arrays

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from apexpinn.eval.s1_report import load_records, render_report, write_report  # noqa: E402
from apexpinn.train.s1_study import full_config, parse_shard, run_study, smoke_config  # noqa: E402


def _shard(text: str) -> tuple[int, int]:
    try:
        return parse_shard(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--part", choices=("main", "misspec", "all"), default="all", help="'all' runs main, then misspec"
    )
    parser.add_argument("--shard", type=_shard, default=(0, 1), help="i/n: run every n-th planned run from i")
    parser.add_argument("--force", action="store_true", help="rerun runs whose recorded config_hash differs")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from existing JSON")
    parser.add_argument(
        "--only", action="append", metavar="RUN_ID", help="run only this run id (repeatable)"
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root (default: .)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    decisions = root / "configs" / "decisions_v1.yaml"
    cfg = smoke_config(decisions) if args.config == "smoke" else full_config(decisions)
    code = 0
    if not args.report_only:
        try:
            summary = run_study(
                cfg,
                part=args.part,
                shard=args.shard,
                force=args.force,
                root=root,
                log=print,
                only=args.only,
            )
        except ValueError as exc:
            print(f"stopped: {exc}", file=sys.stderr)
            code = 2
        else:
            print(
                f"planned {summary['planned']}, run {summary['run']}, skipped {summary['skipped']}, "
                f"errors {summary['errors']}; total measured wall time {summary['wall_time_s']:.1f} s"
            )
            code = 1 if summary["errors"] else 0
    records = load_records(root / cfg.out_dir)
    report = write_report(root / cfg.report_path, render_report(records, cfg))
    print(f"report: {report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
