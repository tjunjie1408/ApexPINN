"""Source-backed T13 diagnostics and collection. Reading completed results never fits a model."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.data.guards import slice_at_origin
from apexpinn.eval.protocol import ForecastRecord, paired_comparison, score_forecasts
from apexpinn.models.stints import Scenario, forecast_q_knots
from apexpinn.physics.residuals import states_and_rates, thermal_residual, wear_residual


def job_diagnostics(ensemble, gold, records, method) -> dict:
    """Report original-population S3 strata and past-only representative S4/residual cases."""
    lookup = {stint: group.sort_values("stint_lap") for stint, group in gold.groupby("stint_id")}
    strata = {}
    representatives = {}
    bands = method["diagnostics"]["temperature_bands_c"]
    for row in records:
        first = lookup[row.stint_id].iloc[0]
        for dimension, value in (
            ("circuit", first.circuit_id),
            ("event", row.event_id),
            ("compound", first.compound_c),
            (
                "temperature",
                "cold"
                if first.obs_track_temp_c < bands[0]
                else "warm"
                if first.obs_track_temp_c < bands[1]
                else "hot",
            ),
        ):
            strata.setdefault((dimension, str(value), row.horizon), []).append(row.y_pred - row.y_true)
        if row.horizon == 5:
            representatives.setdefault(row.stint_id, row)
    summaries = [
        {
            "dimension": dim,
            "stratum": value,
            "horizon": h,
            "n": len(errors),
            "rmse_s": float(np.sqrt(np.mean(np.square(errors)))),
            "mae_s": float(np.mean(np.abs(errors))),
        }
        for (dim, value, h), errors in sorted(strata.items())
    ]
    checks = []
    for stint, origin in representatives.items():
        group = lookup[stint]
        history, _, _ = slice_at_origin(group, origin.t0, 5)
        grid = np.linspace(0, origin.t0 + 5, method["diagnostics"]["grid_points"])
        clean = group[group.is_scored_lap.eq(True)].sort_values("stint_lap")
        minimum = clean.loc[clean.proxy_y.idxmin()]
        actual_dip = bool(
            minimum.stint_lap > clean.stint_lap.iloc[0]
            and minimum.stint_lap <= method["diagnostics"]["dip_max_lap"]
            and clean.proxy_y.iloc[0] - minimum.proxy_y > method["diagnostics"]["dip_margin_s"]
            and clean.proxy_y.iloc[-1] > minimum.proxy_y
        )
        for index, member in enumerate(ensemble.members):
            h = history
            if member.constant_q is not None:
                h = replace(
                    h,
                    q=np.full_like(h.q, member.constant_q),
                    ctx=replace(h.ctx, circuit_q_mean=member.constant_q),
                )
            scenario = Scenario.from_history(h)
            knots = forecast_q_knots(h, 5, scenario)
            model = member.model
            if model.uses_collocation:

                def latent(t, fitted=model, context=h.ctx):
                    return fitted.latent(t, context.features())

                shape = np.asarray(model.shape(h.ctx.compound_idx, h.ctx.features(), grid)[0])
            else:
                latent = model.dense_latent(h.ctx.compound_idx, knots)
                shape = np.asarray(model.shape(h.ctx.compound_idx, knots, grid)[0])
            thermal, _, dh, dw = states_and_rates(latent, grid)
            q = np.interp(grid, np.arange(len(knots)), knots)
            rt = np.asarray(thermal_residual(model.physics, h.ctx.compound_idx, thermal, dh, q))
            rw = np.asarray(wear_residual(model.physics, h.ctx.compound_idx, thermal, dw))
            observed_grid = grid >= clean.stint_lap.iloc[0]
            predicted = shape[observed_grid]
            predicted_grid = grid[observed_grid]
            argmin = int(np.argmin(predicted))
            predicted_dip = bool(
                argmin > 0
                and predicted_grid[argmin] <= method["diagnostics"]["dip_max_lap"]
                and predicted[0] - predicted[argmin] > method["diagnostics"]["dip_margin_s"]
                and predicted[-1] > predicted[argmin]
            )
            checks.append(
                {
                    "stint": stint,
                    "member": index,
                    "origin": origin.t0,
                    "thermal_residual_rms": float(np.sqrt(np.mean(rt**2))),
                    "wear_residual_rms": float(np.sqrt(np.mean(rw**2))),
                    "wear_violation_share": float(
                        np.mean(np.asarray(dw) < -method["diagnostics"]["violation_tolerance"])
                    ),
                    "actual_full_stint_dip": actual_dip,
                    "predicted_past_scenario_dip": predicted_dip,
                }
            )
    parameters = [
        {
            name: np.asarray(getattr(m.model.physics, name)).tolist()
            for name in ("k", "rho", "gamma", "h_opt", "kappa")
        }
        for m in ensemble.members
    ]
    return {
        "evidence": "Observed",
        "quantity": "S3 point-error strata and representative S4/physics diagnostics",
        "population": "one earliest eligible h5 origin per held-out stint; all scored origins for S3",
        "conditions": {
            "future_q": "past-history scenario only",
            "S4_labels": "post-hoc descriptive whole-stint target labels; never inputs",
        },
        "instrument": {"method": "autodiff residuals on fixed continuous grid; original source units"},
        "strata": summaries,
        "representative_stints": checks,
        "member_parameters": parameters,
    }


def collect_results(root: Path) -> int:
    declaration = json.loads((root / "declaration.json").read_text())
    jobs = [json.loads(p.read_text()) for p in sorted((root / "jobs").glob("*/job.json"))]
    required = {
        (m, f"F{fold}", p, v)
        for m in ("M1", "M2")
        for fold in range(1, 7)
        for p in ("low", "mid", "high")
        for v in ("main", "no_thermal", "no_wear", "constant_q", "no_window")
        if m == "M1" or v == "main"
    }
    actual = {
        (
            j["conditions"]["model"],
            j["population"]["fold"],
            j["conditions"]["phi_label"],
            j["conditions"]["variant"],
        )
        for j in jobs
    }
    gaps = [str(k) for k in sorted(required - actual)]
    for job in jobs:
        if job["run_identity"] != declaration["identity"]:
            raise ValueError("job/declaration identity mismatch")
        name = "_".join(
            (
                job["conditions"]["model"],
                job["population"]["fold"],
                job["conditions"]["phi_label"],
                job["conditions"]["variant"],
            )
        )
        for file, sha in job["files"].items():
            if digest(root / "jobs" / name / file) != sha:
                raise ValueError("job forecast/audit hash mismatch")
    metrics, comparisons = [], []
    if not gaps:
        baseline_review = json.loads(Path("experiments/g1_real_20261103/review.json").read_text())
        for label in ("low", "mid", "high"):
            forecasts = []
            for path in sorted((root / "jobs").glob(f"*_{label}_*/forecasts.jsonl")):
                forecasts.extend(ForecastRecord(**json.loads(line)) for line in path.open())
            baseline = Path("data/cloud_baselines") / f"phi_{label}.parquet"
            if not baseline.exists():
                gaps.append(f"missing baseline forecasts: {baseline}")
                continue
            baseline_meta = json.loads(baseline.with_suffix(".json").read_text())
            if (
                digest(baseline) != baseline_meta["sha256"]
                or baseline_meta["source_sha256"] != baseline_review["studies"][label]["source_sha256"]
            ):
                raise ValueError("compact G1 baseline provenance mismatch")
            baseline_rows = [ForecastRecord(**row) for row in pd.read_parquet(baseline).to_dict("records")]
            metrics.extend(asdict(v) for v in score_forecasts(forecasts))
            model_names = sorted({r.model for r in forecasts})
            for model in model_names:
                reference_names = (
                    ("B1", "B3", baseline_review["studies"][label]["inner_classic_reference"])
                    if "/" not in model
                    else ("M1",)
                )
                if model == "M2":
                    reference_names = (*reference_names, "M1")
                for reference in reference_names:
                    reference_rows = baseline_rows if reference.startswith("B") else forecasts
                    for h in (1, 3, 5, 10):
                        try:
                            ci = paired_comparison(
                                [r for r in forecasts if r.model == model],
                                [r for r in reference_rows if r.model == reference],
                                horizon=h,
                                phi={"low": 0.03, "mid": 0.06, "high": 0.09}[label],
                                seed=0,
                            )
                            comparisons.append(
                                {
                                    "model": model,
                                    "reference": reference,
                                    "horizon": h,
                                    "phi_label": label,
                                    **asdict(ci),
                                }
                            )
                        except ValueError as exc:
                            gaps.append(f"incomparable {label}/{model}/{reference}/h{h}: {exc}")
    review = {
        "schema": "apexpinn.t13_review/v1",
        "evidence": "Observed",
        "quantity": "physics evaluation collection",
        "conditions": {
            "forecast_sweep": "COMPLETE" if not gaps else "INCOMPLETE",
            "G2": "NOT ESTABLISHED: remote restart evidence, H2 and scientific review remain separate",
            "P2": "USER DECISION PENDING",
        },
        "population": {"completed_jobs": len(jobs), "required_jobs": len(required)},
        "instrument": {"declaration_sha256": digest(root / "declaration.json")},
        "gaps": gaps,
        "metrics": metrics,
        "comparisons": comparisons,
        "not_established": [
            "H2 within-known-circuit validation-stint experiment is not implemented by this runner",
            "Cloud restart oracle and complete result review must be checked before G2 acceptance",
        ],
    }
    write_json(root / "review.json", review)
    lines = [
        "# Physics evaluation report",
        "",
        "Generated by apexpinn.eval.physics_report; do not edit by hand.",
        "",
        f"Source: `{root / 'review.json'}`. Observed completed jobs: {len(jobs)}/{len(required)}.",
        f"G2: {review['conditions']['G2']}. P2 remains the user's decision.",
        "",
        "| Model | Fold | Phi | Variant | Trials | Calibration fits | Final fits |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: |",
    ]
    for j in jobs:
        c = j["conditions"]
        lines.append(
            f"| {c['model']} | {j['population']['fold']} | {c['phi']} | {c['variant']} | "
            f"{len(j['trials'])} | {len(j['calibration_fits'])} | {len(j['final_fits'])} |"
        )
    lines += ["", "| Model | Phi | Horizon | Metric | Value |", "| :--- | ---: | ---: | :--- | ---: |"]
    for m in metrics:
        c = m["conditions"]
        lines.append(
            f"| {c['model']} | {c['phi']} | {c['horizon_actual_laps']} | "
            f"{m['quantity']['metric']} | {m['value']} |"
        )
    lines += [
        "",
        "| Model | Reference | Phi | Horizon | Relative RMSE reduction | 95% CI |",
        "| :--- | :--- | :--- | ---: | ---: | :--- |",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['model']} | {c['reference']} | {c['phi_label']} | {c['horizon']} | "
            f"{c['estimate']:.6g} | [{c['ci_low']:.6g}, {c['ci_high']:.6g}] |"
        )
    lines += [
        "",
        "| Model | Fold | Phi | Variant | Mean thermal RMS | Mean wear RMS | Mean violation share |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: |",
    ]
    for j in jobs:
        values = j["diagnostics"]["representative_stints"]
        means = [
            float(np.mean([v[k] for v in values])) if values else None
            for k in ("thermal_residual_rms", "wear_residual_rms", "wear_violation_share")
        ]
        c = j["conditions"]
        lines.append(
            f"| {c['model']} | {j['population']['fold']} | {c['phi']} | {c['variant']} | "
            + " | ".join("Unknown" if v is None else f"{v:.6g}" for v in means)
            + " |"
        )
    lines += [
        "",
        "Paired CIs and S3/S4/residual diagnostics are retained in review.json and each job.json.",
        "Tuning, validation-scale calibration and final refit costs are recorded separately.",
        "**Physics scales use inner-validation residuals; G1 baseline scales use training residuals.**",
        "Probability comparisons therefore have different calibration conditions, not a physics-only effect.",
        "L2 is refuted. Source-flag slow-lap cases remain in the original population.",
        "A job or a completed notebook is not automatically G2 acceptance. P2 is not chosen by this runner.",
        "H2 ID-versus-OOD comparison is NOT IMPLEMENTED. Inner circuit holdout is not ID data.",
        "",
    ]
    lines += [f"- Missing/incomparable: {g}" for g in gaps]
    (root / "eval_physics.md").write_text("\n".join(lines), encoding="utf-8")
    return int(bool(gaps))
