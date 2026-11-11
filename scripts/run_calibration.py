"""Run the separately declared clean M2 calibration pilot on user-launched Colab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=Path, default=Path("configs/calibration_clean_v1.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/decisions_v1.yaml"))
    parser.add_argument("--gold-root", type=Path, default=Path("data/gold_development_frozen_v1"))
    parser.add_argument("--fold", default="F3")
    parser.add_argument("--phi", choices=("low", "mid", "high"), default="mid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan and args.report_only:
        parser.error("choose either plan or report-only")
    if args.report_only:
        from apexpinn.eval.calibration_report import render_report

        render_report(args.out)
        return 0
    from apexpinn.contracts import load_decisions
    from apexpinn.eval.calibration import split_plan
    from apexpinn.train.calibration_study import read_spec

    method, decisions = read_spec(args.method), load_decisions(args.config)
    if args.plan:
        print(
            json.dumps(
                {
                    "method": method["method_id"],
                    "split": split_plan(decisions, args.fold),
                    "tuning_fits": 20,
                    "final_fits": 5,
                    "steps_per_fit": 1000,
                    "calibration_factors": 4,
                    "refit_after_calibration": False,
                    "T13_v1_modified": False,
                    "compute_status": "NOT RUN",
                },
                indent=2,
            )
        )
        return 0
    import jax

    jax.config.update("jax_enable_x64", True)
    from apexpinn.train.calibration_study import run_clean

    run_clean(args.method, args.config, args.gold_root, args.out, args.fold, args.phi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
