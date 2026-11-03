"""Compact, source-backed review of completed real baseline studies across the declared phi sweep."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_study_for_review(path: Path, *, chunk_size: int = 1 << 20) -> dict[str, Any]:
    """Read the study incrementally, checking but discarding its redundant per-origin audit rows."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer, position, eof = "", 0, False

        def refill() -> bool:
            nonlocal buffer, position, eof
            extra = stream.read(chunk_size)
            buffer, position = buffer[position:] + extra, 0
            eof = not extra
            return not eof

        def peek() -> str:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof or not refill():
                    return ""

        def take(token: str) -> None:
            nonlocal position
            if peek() != token:
                raise ValueError(f"expected {token!r} in study JSON")
            position += 1

        def value() -> Any:
            nonlocal position
            peek()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
                    continue
                if end == len(buffer) and not eof:
                    refill()
                    continue
                if end < len(buffer) and buffer[end] not in " \t\r\n,]}:":
                    if eof:
                        raise ValueError("invalid JSON token boundary")
                    refill()
                    continue
                position = end
                return result

        take("{")
        study = {}
        if peek() != "}":
            while True:
                name = value()
                if not isinstance(name, str) or name in study:
                    raise ValueError("study keys must be unique strings")
                take(":")
                if peek() == "[":
                    take("[")
                    rows = []
                    if peek() != "]":
                        while True:
                            row = value()
                            if name != "origin_audit":
                                rows.append(row)
                            if peek() == "]":
                                break
                            take(",")
                    take("]")
                    study[name] = rows
                else:
                    study[name] = value()
                if peek() == "}":
                    break
                take(",")
        take("}")
        if peek():
            raise ValueError("trailing content after study JSON")
        return study


def summarize_study(receipt: dict[str, Any], *, source: str, source_hash: str) -> dict[str, Any]:
    """Check completed accounting/populations and retain a small review record, not the large raw rows."""
    errors = []
    models = ("B0", "B1", "B2", "B3")
    folds = ("F1", "F2", "F3", "F4", "F5", "F6")
    horizons = (1, 3, 5, 10)
    if receipt["population"]["kind"] != "development" or receipt["conditions"]["validation_only"]:
        errors.append("not real development evaluation")
    if any(event.split("_", 1)[0] not in ("2023", "2024") for event in receipt["population"]["events"]):
        errors.append("event outside declared development seasons")
    if set(receipt["population"]["folds"]) != set(folds) or not receipt["conditions"]["coverage_complete"]:
        errors.append("incomplete configured fold/data coverage")
    if receipt["conditions"]["ensemble_members"] != 5 or receipt["conditions"]["mlp_steps"] != 500:
        errors.append("ensemble/training budget differs from declaration")
    errors.extend(f"model run failure: {name}: {reason}" for name, reason in receipt["failures"].items())
    budgets = []
    for model in models:
        for fold in folds:
            trials = [r for r in receipt["trials"] if r["model"] == model and r["outer_fold"] == fold]
            fits = [r for r in receipt["final_member_fits"] if r["model"] == model and r["fold"] == fold]
            count = 1 if model in ("B0", "B2") else 20
            if len(trials) != count or {r["trial"] for r in trials} != set(range(count)):
                errors.append(f"{fold}/{model}: trial count/IDs differ")
            if len({json.dumps(r["parameters"], sort_keys=True) for r in trials}) != count:
                errors.append(f"{fold}/{model}: trial parameters are not unique")
            if any(
                r["conditions"]["trial_budget"] != 20
                or r["conditions"]["declared_unique_configurations"] != count
                or r["conditions"]["primary_horizon"] != 5
                or r["conditions"]["mlp_steps"] != 500
                or r["conditions"]["validation_only"]
                or r["phi"] != receipt["conditions"]["phi"]
                for r in trials
            ):
                errors.append(f"{fold}/{model}: trial protocol differs")
            if len(fits) != 5 or {r["member"] for r in fits} != set(range(5)):
                errors.append(f"{fold}/{model}: final member count/IDs differ")
            if any(r["status"] != "finished" for r in fits):
                errors.append(f"{fold}/{model}: final member failure")
            if len({r["seed"] for r in fits}) != 5 or any(not r["state_sha256"] for r in fits):
                errors.append(f"{fold}/{model}: missing independent final fits/state identity")
            if any(not math.isfinite(r["elapsed_s"]) or r["elapsed_s"] < 0 for r in [*trials, *fits]):
                errors.append(f"{fold}/{model}: invalid wall-clock telemetry")
            if any(r["status"] not in ("finished", "failed") for r in trials):
                errors.append(f"{fold}/{model}: invalid trial status")
            if any(
                r["status"] == "finished"
                and (r["rmse_primary_s"] is None or not math.isfinite(r["rmse_primary_s"]))
                for r in trials
            ):
                errors.append(f"{fold}/{model}: invalid completed trial score")
            successful = [r for r in trials if r["status"] == "finished"]
            if successful:
                best_trial = min(successful, key=lambda r: (r["rmse_primary_s"], r["trial"]))
                if receipt["selections"].get(f"{fold}/{model}") != best_trial["parameters"]:
                    errors.append(f"{fold}/{model}: selection differs from inner-only minimum")
            budgets.append(
                {
                    "model": model,
                    "fold": fold,
                    "trials": len(trials),
                    "failed": sum(r["status"] == "failed" for r in trials),
                    "trial_s": sum(r["elapsed_s"] for r in trials),
                    "members": len(fits),
                    "fit_s": sum(r["elapsed_s"] for r in fits),
                    "resumed": sum("resumed_from" in r["instrument"] for r in trials),
                }
            )
    indexed = {}
    for row in receipt["metrics"]:
        key = (
            row["conditions"]["model"],
            row["conditions"]["horizon_actual_laps"],
            row["quantity"]["metric"],
        )
        if key in indexed:
            errors.append(f"duplicate metric {key}")
        indexed[key] = row["value"]
        if row["value"] is None or not math.isfinite(row["value"]):
            errors.append(f"unknown/nonfinite metric {key}")
    scores = []
    for model in models:
        for horizon in horizons:
            row = {
                name: indexed.get((model, horizon, name))
                for name in ("rmse", "mae", "crps", "coverage_50", "width_50", "coverage_90", "width_90")
            }
            if any(v is None for v in row.values()):
                errors.append(f"missing metric group {model}/{horizon}")
            scores.append({"model": model, "horizon": horizon, **row})
    keys = {model: {} for model in models}
    counts = Counter()
    fold_counts = Counter()
    for row in receipt["forecasts"]:
        model = row["model"]
        if model not in keys:
            errors.append(f"unknown forecast model {model}")
            continue
        key = (row["fold"], row["event_id"], row["stint_id"], row["t0"], row["horizon"])
        target = (
            row["y_true"],
            row["instrument"]["gold_hash"],
            row["instrument"]["normalization_hash"],
            tuple(sorted((k, v) for k, v in row["conditions"].items() if k != "model")),
        )
        if key in keys[model]:
            errors.append(f"duplicate forecast {model}/{key}")
        keys[model][key] = target
        counts[(model, row["horizon"])] += 1
        fold_counts[(model, row["horizon"], row["fold"])] += 1
        if (
            row["phi"] != receipt["conditions"]["phi"]
            or row["target_stint_lap"] != row["t0"] + row["horizon"]
        ):
            errors.append(f"inconsistent phi/endpoint {model}/{key}")
        scales = row["member_sigma_s"]
        if (
            row["sigma_s"] <= 0
            or not scales
            or len(scales) != 5
            or len(row["member_means_s"]) != 5
            or any(not math.isfinite(v) or v <= 0 for v in scales or [])
            or not all(
                math.isfinite(v)
                for v in [row["y_true"], row["y_pred"], row["sigma_s"], *row["member_means_s"]]
            )
        ):
            errors.append(f"invalid ensemble scale {model}/{key}")
        if row["conditions"]["validation_only"] or row["conditions"]["config_status"] != "frozen":
            errors.append(f"fixture or draft forecast {model}/{key}")
    for model in models:
        if keys[model] != keys["B0"] or not keys[model]:
            errors.append(f"{model}: mismatched or empty paired population")
        if any(fold_counts[(model, h, fold)] == 0 for h in horizons for fold in folds):
            errors.append(f"{model}: missing evaluated fold/horizon population")
    comparisons = receipt["paired_comparisons"]
    if len(comparisons) != 12 or any(c.get("status") == "Unknown" for c in comparisons):
        errors.append("missing paired confidence intervals")
    if any(
        not all(math.isfinite(c.get(k, math.nan)) for k in ("estimate", "ci_low", "ci_high"))
        for c in comparisons
    ):
        errors.append("nonfinite confidence interval")
    reference = receipt["selected_reference_by_inner_validation"]
    if {(c["model"], c["horizon"]) for c in comparisons} != {
        (m, h) for m in models if m != reference for h in horizons
    }:
        errors.append("confidence interval model/horizon set differs")
    if any(c.get("reference") != reference or c.get("n_resamples") != 2000 for c in comparisons):
        errors.append("confidence interval reference/resample count differs")
    origins = {h: {} for h in horizons}
    origin_counts = {}
    for row in receipt["origin_metrics"]:
        model = row["conditions"]["model"]
        h, name = row["conditions"]["horizon_actual_laps"], row["quantity"]["metric"]
        fold = row["population"]["fold"]
        origin_counts[(model, fold, h, name)] = row["value"]
        if model == "B0" and name != "history_rejection_share":
            origins[h][name] = origins[h].get(name, 0) + int(row["value"])
    count_names = (
        "candidate_origins",
        "eligible_origins",
        "history_rejected_origins",
        "target_rejected_origins",
        "input_quality_rejected_origins",
        "history_rejection_share",
    )
    origin_by_fold = []
    for fold in folds:
        for h in horizons:
            values = {name: origin_counts.get(("B0", fold, h, name)) for name in count_names}
            if any(v is None or not math.isfinite(v) or v < 0 for v in values.values()):
                errors.append(f"{fold}/h{h}: missing/invalid origin denominator")
                continue
            if any(
                origin_counts.get((model, fold, h, name)) != values[name]
                for model in models
                for name in count_names
            ):
                errors.append(f"{fold}/h{h}: model-dependent eligibility audit")
            if values["candidate_origins"] != sum(values[name] for name in count_names[1:5]):
                errors.append(f"{fold}/h{h}: origin counts do not partition candidates")
            if any(values["eligible_origins"] != fold_counts[(m, h, fold)] for m in models):
                errors.append(f"{fold}/h{h}: eligible count differs from forecasts")
            expected_share = (
                values["history_rejected_origins"] / values["candidate_origins"]
                if values["candidate_origins"]
                else math.nan
            )
            if not math.isclose(values["history_rejection_share"], expected_share, abs_tol=1e-15):
                errors.append(f"{fold}/h{h}: history rejection share differs")
            origin_by_fold.append({"fold": fold, "horizon": h, **values})
    best = min(models, key=lambda m: (indexed[(m, 5, "rmse")], m)) if not errors else None
    return {
        "source": source,
        "source_sha256": source_hash,
        "phi": receipt["conditions"]["phi"],
        "errors": errors,
        "complete": not errors,
        "budgets": budgets,
        "scores": scores,
        "inner_reference": receipt["selected_reference_by_inner_validation"],
        "inner_classic_reference": receipt["selected_B0_B2_reference_by_inner_validation"],
        "best_observed_primary": best,
        "paired_comparisons": comparisons,
        "origin_metrics": receipt["origin_metrics"],
        "origins_by_horizon": origins,
        "origin_counts_by_fold": origin_by_fold,
        "population_keys_sha256": hashlib.sha256(json.dumps(sorted(keys["B0"])).encode()).hexdigest(),
        "forecast_counts": {f"{m}/h{h}": n for (m, h), n in sorted(counts.items())},
        "shared_input_preparation_s": receipt["instrument"].get("shared_input_preparation_s", {}),
        "resume_provenance": receipt["instrument"].get("resume_provenance"),
    }


def review_complete(studies: dict[str, Any]) -> bool:
    return (
        set(studies) == {"low", "mid", "high"}
        and all(studies[label]["phi"] == phi for label, phi in (("low", 0.03), ("mid", 0.06), ("high", 0.09)))
        and all(s["complete"] for s in studies.values())
        and len({s["population_keys_sha256"] for s in studies.values()}) == 1
    )


def build_g1_report(review: dict[str, Any], *, source: str) -> str:
    """Render reviewed full studies without selecting hyperparameters or changing any result."""
    studies = review["studies"]
    status = "PASS" if review_complete(studies) else "NOT ESTABLISHED"
    budgets = [b for study in studies.values() for b in study["budgets"]]
    trials, members = sum(b["trials"] for b in budgets), sum(b["members"] for b in budgets)
    resumed = sum(b["resumed"] for b in budgets)
    lines = [
        "# Baseline evaluation report",
        "",
        "Generated by apexpinn.eval.g1_report; do not edit by hand.",
        "",
        f"Review source: `{source}`. Counts, scores and wall times are Observed in its linked studies.",
        f"G1: **{status}**. Real 2023-2024 development evaluation; no temporal-holdout model results.",
        f"Observed accounting: {trials} trial records ({resumed} resumed, {trials - resumed} new in v3); "
        f"{members} selected final-member fits. Earlier archive copies are not extra tuning attempts.",
        "",
        "Primary horizon: 5 actual laps. Hyperparameters and comparator references use inner circuits only.",
        "Lowest outer RMSE is descriptive; it does not change hyperparameters or the inner reference.",
        "",
        "## Primary benchmark and sensitivity",
        "",
        "| Phi | Inner-selected reference | B0-B2 inner reference | Lowest observed outer h5 RMSE |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for study in studies.values():
        best = study["best_observed_primary"]
        primary = next((r["rmse"] for r in study["scores"] if r["model"] == best and r["horizon"] == 5), None)
        best_label = "NOT ESTABLISHED" if primary is None else f"{best}, {primary:.6g} s"
        lines.append(
            f"| {study['phi']} | {study['inner_reference']} | {study['inner_classic_reference']} | "
            f"{best_label} |"
        )
    for label, study in studies.items():
        lines += [
            "",
            f"## Phi {study['phi']} ({label})",
            "",
            f"Source: `{study['source']}`.",
            "",
            "| Model | Horizon | RMSE s | MAE s | CRPS s | Cover 50 | Width 50 s | Cover 90 | Width 90 s |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in study["scores"]:
            values = [
                "Unknown" if row[k] is None else f"{row[k]:.6g}"
                for k in ("rmse", "mae", "crps", "coverage_50", "width_50", "coverage_90", "width_90")
            ]
            lines.append(f"| {row['model']} | {row['horizon']} | " + " | ".join(values) + " |")
        lines += [
            "",
            "| Horizon | Candidates | Eligible | History rejected | Target rejected | Input rejected |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for h, values in study["origins_by_horizon"].items():
            counts = [
                str(values.get(k, "Unknown"))
                for k in (
                    "candidate_origins",
                    "eligible_origins",
                    "history_rejected_origins",
                    "target_rejected_origins",
                    "input_quality_rejected_origins",
                )
            ]
            lines.append(f"| {h} | " + " | ".join(counts) + " |")
        lines += [
            "",
            "History rejection uses every original candidate as its denominator, before target filtering.",
            "",
            "| Fold | Horizon | Candidates | Eligible | History rejected | History rejection share |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in study["origin_counts_by_fold"]:
            lines.append(
                f"| {row['fold']} | {row['horizon']} | {int(row['candidate_origins'])} | "
                f"{int(row['eligible_origins'])} | {int(row['history_rejected_origins'])} | "
                f"{row['history_rejection_share']:.6g} |"
            )
        lines += [
            "",
            "| Model | Reference | Horizon | Relative RMSE reduction | 95% CI |",
            "| :--- | :--- | ---: | ---: | :--- |",
        ]
        for item in study["paired_comparisons"]:
            if item.get("status") == "Unknown":
                value, interval = "Unknown", item["reason"]
            else:
                value = f"{item['estimate']:.6g}"
                interval = f"[{item['ci_low']:.6g}, {item['ci_high']:.6g}]"
            lines.append(
                f"| {item['model']} | {item['reference']} | {item['horizon']} | {value} | {interval} |"
            )
        lines += [
            "",
            "| Fold | Model | Trials | Failed | Resumed | Trial seconds | Final members | Fit seconds |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for b in study["budgets"]:
            lines.append(
                f"| {b['fold']} | {b['model']} | {b['trials']} | {b['failed']} | {b['resumed']} | "
                f"{b['trial_s']:.6g} | {b['members']} | {b['fit_s']:.6g} |"
            )
        lines.extend(f"- Review gap: {error}" for error in study["errors"])
    if diagnostic := review.get("posthoc_source_flag_diagnostic"):
        lines += [
            "",
            "## Descriptive error cases",
            "",
            f"Observed post-hoc source check: `{diagnostic['source']}`. Phi=0.03, horizon=5.",
            "The cases were inspected after scoring and remain in every model's evaluation population.",
            "",
            "| Stint | Origin | First scored lap s | Status / accurate | B2 forecast s | Target s |",
            "| :--- | ---: | ---: | :--- | ---: | ---: |",
        ]
        for case in diagnostic["cases"]:
            forecast = case["forecast"]
            first = next(row for row in case["historical_rows"] if row["is_scored_lap"])
            lines.append(
                f"| {forecast['stint']} | {forecast['t0']} | {first['obs_lap_time_s']:.6g} | "
                f"{first['obs_track_status']} / {first['obs_is_accurate']} | "
                f"{forecast['y_pred_s']:.6g} | {forecast['y_true_s']:.6g} |"
            )
        lines += [
            "",
            "Forecast/target columns are trend-corrected lap times. Very slow green/accurate laps can "
            "pass the specified metadata filters and enter anchoring; B2 can then extrapolate negative "
            "corrected times. Source flags do not guarantee steady racing pace.",
            "No causal restart explanation is established by this check. No retrospective outlier "
            "removal or interval inflation was applied. Future data-quality changes require a separate "
            "method declaration and outputs.",
        ]
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "CIs use 2,000 paired race-then-stint resamples. Intervals have no multiplicity correction.",
        "Models use identical keyed populations and targets; no favorable intersections are taken.",
        "B0/B2 have one unique configuration each; B1/B3 have 20 per outer fold/phi. Failures count.",
        "Five final-member fits are timed separately. Deterministic B0/B2 member states may coincide.",
        "Replay-only trials exclude shared input preparation, which is recorded separately in each study.",
        "Resumed legacy times include all-horizon construction/scoring and concurrent diagnostic work; "
        "they are not directly comparable with replay-only times. Abandoned legacy final-work overhead "
        "is disclosed in the handoff and is outside selected final-member totals.",
        "Fit and tuning time are not inference latency or evidence for a simulator speedup.",
        "These shared-machine wall clocks include uncontrolled local checks and other workloads.",
        "Compiled float64 execution preserves model equations and optimizer settings; roundoff can "
        "change the Kalman optimizer path. Cross-kernel fits are not claimed bitwise identical.",
        "L2 remains REFUTED; the future mean-load approximation is a modeling limitation. Residual clipping "
        "and conservative 2023 Montreal compound exclusion remain as recorded in the G0 report.",
        "No physical-parameter identification, universal optimality, P2 decision or strategy result follows.",
        "",
    ]
    return "\n".join(lines)
