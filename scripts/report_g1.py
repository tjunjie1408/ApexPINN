"""Validate completed real studies and generate the G1 review without rerunning any model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from apexpinn.data.manifest import sha256_file
from apexpinn.eval.g1_report import (
    build_g1_report,
    read_study_for_review,
    review_complete,
    summarize_study,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="experiments/baselines_development_exec_v3_phi_")
    parser.add_argument("--review", type=Path, default=Path("experiments/g1_real_20261103/review.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/eval_baselines.md"))
    args = parser.parse_args()
    if args.review.exists():
        raise ValueError("review already exists; choose a separately versioned output")
    if args.report.exists() and args.report.read_text(encoding="utf-8").splitlines()[:1] != [
        "# Baseline evaluation report"
    ]:
        raise ValueError("refusing to replace an unrelated report")
    declaration = Path("experiments/g1_real_20261103/execution_v3_declaration.json")
    declared = json.loads(declaration.read_text())
    original = json.loads(declaration.with_name("declaration.json").read_text())
    g0_path = Path("experiments/g0_development_20261103/acceptance.json")
    assert sha256_file(g0_path) == original["instrument"]["G0_acceptance_sha256"]
    g0 = json.loads(g0_path.read_text())
    gold_payloads = {}
    for settings in g0["phi"].values():
        manifest_path = Path(settings["gold_manifest"])
        assert sha256_file(manifest_path) == settings["gold_manifest_sha256"]
        manifest = json.loads(manifest_path.read_text())
        for fold, metadata in manifest["folds"].items():
            for name, properties in metadata["files"].items():
                payload = manifest_path.parent / fold / name
                assert sha256_file(payload) == properties["sha256"], payload
                gold_payloads[str(payload)] = properties["sha256"]
    metric_sources = {}
    for name in ("src/apexpinn/eval/metrics.py", "src/apexpinn/eval/bootstrap.py"):
        digest = sha256_file(name)
        assert digest == original["instrument"]["source_sha256"][name], name
        metric_sources[name] = digest
    assert (
        sha256_file("configs/decisions_v1.yaml")
        == declared["instrument"]["source_sha256"]["configs/decisions_v1.yaml"]
    )
    review = {
        "schema": "apexpinn.g1_review/v1",
        "evidence": "Observed",
        "quantity": "completed development baseline benchmark and gate evidence",
        "population": "all configured development folds and phi settings, no temporal holdout",
        "conditions": {
            "primary_horizon": 5,
            "reference_selection": "inner scores only",
            "L2": "refuted, unchanged limitation",
        },
        "instrument": {
            "declaration_sha256": sha256_file(declaration),
            "G0_acceptance_sha256": sha256_file(g0_path),
            "accepted_gold_payload_sha256": gold_payloads,
            "unchanged_metric_source_sha256": metric_sources,
            "review_builder_sha256": sha256_file(
                Path(__file__).parents[1] / "src/apexpinn/eval/g1_report.py"
            ),
            "review_cli_sha256": sha256_file(__file__),
        },
        "studies": {},
    }
    for label in ("low", "mid", "high"):
        path = Path(args.prefix + label) / "study.json"
        receipt = read_study_for_review(path)
        for name, digest in receipt["instrument"]["source_sha256"].items():
            assert sha256_file(path.parent / "source" / name) == digest, name
            expected = declared["instrument"]["source_sha256"].get("src/apexpinn/" + name)
            if expected is not None:
                assert digest == expected, name
        for trial in receipt["trials"]:
            trial_path = (
                path.parent / "trials" / (f"{trial['outer_fold']}_{trial['model']}_{trial['trial']:02d}.json")
            )
            assert json.loads(trial_path.read_text()) == trial, trial_path
            previous = trial
            seen = set()
            while original_path := previous["instrument"].get("resumed_from"):
                assert original_path not in seen, "cyclic trial provenance"
                seen.add(original_path)
                assert sha256_file(original_path) == previous["instrument"]["original_record_sha256"]
                original_trial = json.loads(Path(original_path).read_text())
                assert {k: v for k, v in previous.items() if k != "instrument"} == {
                    k: v for k, v in original_trial.items() if k != "instrument"
                }, original_path
                previous = original_trial
        for fit in receipt["final_member_fits"]:
            fit_path = path.parent / "final_fits" / f"{fit['fold']}_{fit['model']}_{fit['member']}.json"
            assert json.loads(fit_path.read_text()) == fit, fit_path
        gold = pd.read_parquet(Path("data/gold_development_frozen_v1") / f"phi_{label}" / "F1/laps.parquet")
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(gold, index=False).to_numpy().tobytes()
        ).hexdigest()
        assert fingerprint == receipt["instrument"]["input_frame_sha256"]
        del gold
        summary = summarize_study(receipt, source=str(path), source_hash=sha256_file(path))
        errors = {}
        for row in receipt["forecasts"]:
            errors.setdefault((row["model"], row["horizon"]), []).append((row["y_pred"] - row["y_true"]) ** 2)
        for row in summary["scores"]:
            squared = errors.get((row["model"], row["horizon"]), [])
            if squared and row["rmse"] is not None:
                expected = math.sqrt(math.fsum(squared) / len(squared))
                assert math.isclose(expected, row["rmse"], rel_tol=1e-12, abs_tol=1e-12)
        review["studies"][label] = summary
        del receipt, errors
        gc.collect()
        print(
            json.dumps({"phi": label, "complete": summary["complete"], "gaps": summary["errors"]}), flush=True
        )
    diagnostic_path = declaration.with_name("low_source_flag_diagnostic.json")
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        error_source = Path(diagnostic["instrument"]["error_diagnostic"])
        assert sha256_file(error_source) == diagnostic["instrument"]["error_diagnostic_sha256"]
        error_diagnostic = json.loads(error_source.read_text())
        assert error_diagnostic["instrument"]["source_sha256"] == review["studies"]["low"]["source_sha256"]
        for case in diagnostic["cases"]:
            identity = case["instrument"]
            assert identity["gold_sha256"] == gold_payloads[identity["gold_source"]]
        review["posthoc_source_flag_diagnostic"] = {
            "source": str(diagnostic_path),
            "sha256": sha256_file(diagnostic_path),
            "cases": diagnostic["cases"],
        }
    review["G1"] = "PASS" if review_complete(review["studies"]) else "NOT ESTABLISHED"
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.report.exists():
        previous = args.review.parent / "eval_baselines_before_real.md"
        if not previous.exists():
            shutil.copyfile(args.report, previous)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_g1_report(review, source=str(args.review)), encoding="utf-8")
    return int(review["G1"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
