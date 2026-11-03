"""Run the declared baseline study, or an explicitly crafted CPU fixture with --fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import pandas as pd  # noqa: E402

from apexpinn.contracts import load_decisions  # noqa: E402
from apexpinn.data import manifest as mf  # noqa: E402
from apexpinn.train.baseline_study import MODEL_NAMES, TrialRecord, fixture_silver, run_study  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/decisions_v1.yaml"))
    parser.add_argument("--gold-root", type=Path, default=Path("data/gold_t10"))
    parser.add_argument("--phi", choices=("low", "mid", "high"), default="mid")
    parser.add_argument("--out", type=Path, default=Path("experiments/baselines_v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/eval_baselines.md"))
    parser.add_argument("--model", action="append", choices=MODEL_NAMES)
    parser.add_argument("--fold", action="append")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fixture-mlp-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-trials-from", type=Path)
    parser.add_argument("--resume-declaration", type=Path)
    parser.add_argument("--compiled-kernels", action="store_true")
    args = parser.parse_args(argv)
    decisions = load_decisions(args.config)
    phi = getattr(decisions, f"phi_{args.phi}")
    completed, resume = [], None
    if bool(args.resume_trials_from) != bool(args.resume_declaration):
        parser.error("resume requires both trial directory and original execution declaration")
    if args.resume_trials_from:
        declaration = json.loads(args.resume_declaration.read_text())
        handoff_path = args.resume_declaration.with_name("checkpoint_handoff.json")
        if not handoff_path.exists():
            parser.error("resume requires the confirmed stopped-process checkpoint")
        if (
            args.fixture
            or args.fold
            or args.model
            or declaration["conditions"]["seed"] != args.seed
            or declaration["instrument"]["source_sha256"]["configs/decisions_v1.yaml"]
            != mf.sha256_file(args.config)
            or declaration["instrument"]["source_sha256"]["src/apexpinn/models/baselines.py"]
            != mf.sha256_file(Path(__file__).parents[1] / "src/apexpinn/models/baselines.py")
            or declaration["instrument"]["source_sha256"]["src/apexpinn/eval/metrics.py"]
            != mf.sha256_file(Path(__file__).parents[1] / "src/apexpinn/eval/metrics.py")
            or args.resume_trials_from.resolve() != Path(declaration["outputs"][args.phi]).resolve()
        ):
            parser.error(
                "resume declaration requires identical config/model definitions, full models/folds and seed"
            )
        files = sorted((args.resume_trials_from / "trials").glob("*.json"))
        if not files:
            parser.error("no completed trial records to resume")
        resume = {
            "declaration": str(args.resume_declaration),
            "declaration_sha256": mf.sha256_file(args.resume_declaration),
            "original_records": {str(p): mf.sha256_file(p) for p in files},
            "handoff": json.loads(handoff_path.read_text(encoding="utf-8-sig")),
            "note": "Completed trials reused exactly; previous final work discarded, not extra trials. "
            "Legacy trial clocks include all-horizon construction/scoring and differ from execution v2.",
        }
        for path in files:
            record = json.loads(path.read_text())
            record["instrument"] = {
                **record["instrument"],
                "resumed_from": str(path),
                "original_record_sha256": mf.sha256_file(path),
            }
            completed.append(TrialRecord(**record))
    if args.fixture:
        frame = fixture_silver(decisions, phi)
        coverage = False
        source = "crafted_fixture"
        steps = args.fixture_mlp_steps
    else:
        directory = args.gold_root / f"phi_{args.phi}"
        source_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        coverage = bool(
            source_manifest["coverage"]["configured_dev_events_complete"]
            and source_manifest["coverage"]["configured_folds_complete"]
        )
        if decisions.status != "frozen" or not coverage:
            parser.error("G0/full development coverage and frozen config required; no model fit was started")
        fold = next(iter(source_manifest["folds"]))
        path = directory / fold / "laps.parquet"
        expected = source_manifest["folds"][fold]["files"]["laps.parquet"]["sha256"]
        if mf.sha256_file(path) != expected:
            raise ValueError("Gold SHA-256 does not match its manifest")
        frame = pd.read_parquet(path)
        source = "development"
        steps = 500
        if float(source_manifest["conditions"]["phi"]) != phi:
            raise ValueError("Gold phi differs from declared run phi")
    receipt = run_study(
        frame,
        decisions,
        out=args.out,
        report=args.report,
        models=tuple(args.model or MODEL_NAMES),
        folds=args.fold,
        seed=args.seed,
        validation_only=args.fixture,
        source_kind=source,
        coverage_complete=coverage,
        mlp_steps=steps,
        resume_trials=completed,
        resume_provenance=resume,
        compiled_kernels=args.compiled_kernels,
    )
    print(
        json.dumps(
            {
                "G1": receipt["G1"],
                "trials": len(receipt["trials"]),
                "final_member_fits": len(receipt["final_member_fits"]),
                "failures": receipt["failures"],
            },
            sort_keys=True,
        )
    )
    return int(bool(receipt["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
