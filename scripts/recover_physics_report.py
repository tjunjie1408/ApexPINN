"""Build a T13 monitoring report from a verified import and its saved recovery metrics."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import safe_name


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_payloads(root: Path, receipt_path: Path) -> dict:
    manifest = read_json(root / "bundle_manifest.json")
    receipt = read_json(receipt_path)
    if (
        manifest.get("schema") != "apexpinn.cloud_bundle/v1"
        or manifest.get("kind") != "results"
        or manifest != {key: value for key, value in receipt.items() if key != "sha256"}
        or not manifest["files"]
    ):
        raise ValueError("import manifest and result receipt differ")
    for name, expected in manifest["files"].items():
        path = root / safe_name(name)
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"imported payload differs from receipt: {name}")
    return receipt


def forecast_breakdown(forecasts: pd.DataFrame, gold: Path, bands: list[float]) -> list[dict]:
    columns = ["stint_id", "stint_lap", "circuit_id", "compound_c", "obs_track_temp_c"]
    starts = pd.read_parquet(gold, columns=columns).sort_values("stint_lap").drop_duplicates("stint_id")
    rows = forecasts.merge(starts, on="stint_id", how="left", validate="many_to_one")
    if rows[columns[2:]].isna().any().any():
        raise ValueError("missing Gold metadata for a forecast stint")
    rows["temperature_band"] = pd.cut(
        rows.obs_track_temp_c,
        [-float("inf"), *bands, float("inf")],
        labels=["cold", "warm", "hot"],
        right=False,
    )
    rows["squared_error"] = (rows.y_pred - rows.y_true) ** 2
    rows["absolute_error"] = (rows.y_pred - rows.y_true).abs()
    strata = []
    for dimension in ("circuit_id", "event_id", "compound_c", "temperature_band"):
        for (value, horizon), group in rows.groupby([dimension, "horizon"], observed=True):
            strata.append(
                {
                    "dimension": dimension,
                    "stratum": str(value),
                    "horizon": int(horizon),
                    "n": len(group),
                    "rmse_s": float(group.squared_error.mean() ** 0.5),
                    "mae_s": float(group.absolute_error.mean()),
                }
            )
    return strata


def attach_diagnostics(report: dict, path: Path, forecasts: pd.DataFrame, member_count: int) -> None:
    recovered = read_json(path)
    if recovered.get("schema") != "apexpinn.t13_recovered_s4/v1" or any(
        recovered[key] != report["provenance"][key]
        for key in ("forecast_sha256", "gold_sha256", "declaration_sha256")
    ):
        raise ValueError("S4 recovery receipt refers to a different run")
    model_hashes = {row["model_hash"] for row in forecasts.instrument}
    if model_hashes != {recovered["ensemble_sha256"]}:
        raise ValueError("S4 recovery ensemble differs from the saved forecast ensemble")
    expected_origins = forecasts.loc[forecasts.horizon.eq(5)].groupby("stint_id").t0.min().to_dict()
    expected = {
        (stint, origin, member)
        for stint, origin in expected_origins.items()
        for member in range(member_count)
    }
    cases = recovered["diagnostics"]["representative_stints"]
    if (
        not expected
        or len(cases) != len(expected)
        or {(row["stint"], row["origin"], row["member"]) for row in cases} != expected
    ):
        raise ValueError("S4 recovery is missing or repeats representative cases")
    report["status"] = "RECOVERED_EVALUATION_COMPLETE"
    report["s4_status"] = "RECOVERED: physics checks evaluated with the saved ensemble"
    report["s4_recovery"] = recovered
    report["provenance"]["s4_recovery_sha256"] = digest(path)


def read_run(args: argparse.Namespace) -> dict:
    receipt = verified_payloads(args.run, args.receipt)
    model, fold, phi_label, variant = args.job.split("_", 3)
    job_prefix = f"jobs/{safe_name(args.job)}"
    files = receipt["files"]

    def payload(name: str) -> Path:
        if name not in files:
            raise ValueError(f"required payload missing from receipt: {name}")
        return args.run / name

    declaration_path = payload("declaration.json")
    declaration = read_json(declaration_path)
    identity = declaration["identity"]
    gold_sha = digest(args.gold)
    if gold_sha != identity["gold_sha256"][f"{phi_label}/{fold}"]:
        raise ValueError("Gold payload differs from the run declaration")
    metrics = read_json(args.metrics)
    forecast_path = payload(f"{job_prefix}/forecasts.jsonl")
    if metrics["archive_sha256"] != receipt["sha256"] or metrics["forecast_sha256"] != digest(forecast_path):
        raise ValueError("recovered metrics reference different input artifacts")
    forecasts = pd.read_json(forecast_path, lines=True, precise_float=True)
    audit = pd.read_json(payload(f"{job_prefix}/origin_audit.jsonl"), lines=True)
    keys = ["fold", "event_id", "stint_id", "t0", "horizon"]
    if forecasts.duplicated(keys).any() or audit.duplicated(keys).any():
        raise ValueError("duplicate forecast or origin audit keys")
    if set(forecasts[keys].itertuples(index=False, name=None)) != set(
        audit.loc[audit.eligible, keys].itertuples(index=False, name=None)
    ):
        raise ValueError("saved forecasts and eligible origin audit keys differ")
    if set(forecasts.model) != {model} or set(forecasts.fold) != {fold} or forecasts.phi.nunique() != 1:
        raise ValueError("forecast scope differs from the selected job")
    for metric in metrics["metrics"]:
        h = metric["conditions"]["horizon_actual_laps"]
        if metric["population"]["n_forecasts"] != int(forecasts.horizon.eq(h).sum()):
            raise ValueError("recovered metric count differs from saved forecasts")
    timings = {}
    for stage, pattern in (
        ("tuning", "trials/trial_*.json"),
        ("calibration", "calibration/member_*/fit.json"),
        ("final", "final/member_*/fit.json"),
    ):
        paths = sorted((args.run / job_prefix).glob(pattern))
        rows = [read_json(payload(p.relative_to(args.run).as_posix())) for p in paths]
        timings[stage] = {
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "recorded_elapsed_s": sum(row.get("elapsed_s", 0) for row in rows),
        }
    origin_counts = [
        {"horizon": int(h), "reason": str(reason), "count": int(count)}
        for (h, reason), count in audit.groupby(["horizon", "reason"]).size().items()
    ]
    report = {
        "schema": "apexpinn.t13_recovery_report/v1",
        "status": "RECOVERED_PARTIAL",
        "job": args.job,
        "scope": {"model": model, "fold": fold, "phi": float(forecasts.phi.iloc[0]), "variant": variant},
        "execution": {
            "exit_code": args.exit_code,
            "job_record_present": f"{job_prefix}/job.json" in files,
            "underlying_exception": None,
            "recorded_stage_timings": timings,
            "selection": read_json(payload(f"{job_prefix}/selection.json")),
            "training_or_forecast_replay": False,
        },
        "provenance": {
            "verified_payloads": len(receipt["files"]),
            "archive_sha256": receipt["sha256"],
            "declaration_sha256": digest(declaration_path),
            "forecast_sha256": metrics["forecast_sha256"],
            "gold_sha256": gold_sha,
            "baseline_sha256": metrics["baseline_sha256"],
            "recovery_metrics_sha256": digest(args.metrics),
            "method_sha256": identity["method_sha256"],
            "config_sha256": identity["config_sha256"],
            "devices": identity["devices"],
        },
        "metrics": metrics["metrics"],
        "paired_h5": metrics["paired_h5"],
        "pair_errors": metrics["pair_errors"],
        "origin_counts": origin_counts,
        "s3_strata": forecast_breakdown(
            forecasts, args.gold, declaration["method"]["diagnostics"]["temperature_bands_c"]
        ),
        "s4_status": "PENDING: dense fitted-model trajectories and residuals were not saved",
        "G2": "NOT ESTABLISHED",
        "P2": "USER DECISION PENDING",
    }
    if args.diagnostics:
        attach_diagnostics(report, args.diagnostics, forecasts, declaration["method"]["ensemble_members"])
    return report


def render(report: dict) -> str:
    scope = report["scope"]
    execution = report["execution"]
    lines = [
        "# Recovered T13 physics evaluation",
        "",
        f"{scope['model']} / {scope['fold']} / phi {scope['phi']} / {scope['variant']}.",
        f"Status **{report['status']}**. G2 **{report['G2']}**. P2 **{report['P2']}**.",
        "",
        f"Observed subprocess exit code `{execution['exit_code']}`. The underlying exception is unavailable.",
        f"Original job record present `{execution['job_record_present']}`. "
        "The run saved forecasts and origin audit.",
        "Recovery reads saved artifacts without fitting or replaying forecasts.",
        "",
        "## Execution monitoring",
        "",
        "| Stage | Recorded statuses | Recorded hours |",
        "| --- | --- | ---: |",
    ]
    for stage, row in execution["recorded_stage_timings"].items():
        statuses = ", ".join(f"{count} {status}" for status, count in sorted(row["status_counts"].items()))
        lines.append(f"| {stage} | {statuses} | {row['recorded_elapsed_s'] / 3600:.3f} |")
    lines += [
        "",
        "Tuning time includes inner forecast scoring. Unrecorded interruption time is unknown.",
        f"Selected parameters `{json.dumps(execution['selection']['parameters'], sort_keys=True)}`.",
        "",
        "## Held-out forecast scores",
        "",
        "| Horizon | Forecasts | Metric | Value | Unit |",
        "| ---: | ---: | --- | ---: | --- |",
    ]
    for row in report["metrics"]:
        h, n = row["conditions"]["horizon_actual_laps"], row["population"]["n_forecasts"]
        name, unit = row["quantity"]["metric"], row["quantity"]["unit"]
        value = "Unknown" if row["value"] is None else f"{row['value']:.4f}"
        lines.append(f"| {h} | {n} | {name} | {value} | {unit} |")
    lines += [
        "",
        "## Paired five-lap comparison",
        "",
        "Positive relative RMSE reduction favors the physics model.",
        "",
        "| Baseline | RMSE reduction | 95% interval |",
        "| --- | ---: | --- |",
    ]
    for name, row in sorted(report["paired_h5"].items()):
        lines.append(f"| {name} | {row['estimate']:.2%} | [{row['ci_low']:.2%}, {row['ci_high']:.2%}] |")
    lines.extend(
        f"\n- {name} comparison unavailable. {reason}." for name, reason in report["pair_errors"].items()
    )
    lines += ["", "## Origin audit", "", "| Horizon | Disposition | Count |", "| ---: | --- | ---: |"]
    for row in report["origin_counts"]:
        lines.append(f"| {row['horizon']} | {row['reason']} | {row['count']} |")
    lines += [
        "",
        "## S3 error breakdown",
        "",
        "| Dimension | Stratum | Horizon | Forecasts | RMSE s | MAE s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["s3_strata"]:
        lines.append(
            f"| {row['dimension']} | {row['stratum']} | {row['horizon']} | {row['n']} | "
            f"{row['rmse_s']:.4f} | {row['mae_s']:.4f} |"
        )
    if recovery := report.get("s4_recovery"):
        cases = recovery["diagnostics"]["representative_stints"]
        actual = [row for row in cases if row["actual_full_stint_dip"]]
        reproduced = sum(row["predicted_past_scenario_dip"] for row in actual)
        lines += [
            "",
            "## S4 physics checks",
            "",
            f"{len(cases)} member–stint cases across {len({row['stint'] for row in cases})} stints.",
            f"The predicted dip was present in {reproduced}/{len(actual)} cases with an observed dip.",
            f"Diagnostics took {recovery['elapsed_s']:.1f} seconds on {recovery['backend']}.",
            "",
            "| Check | Mean across member–stint cases |",
            "| --- | ---: |",
        ]
        for name in ("thermal_residual_rms", "wear_residual_rms", "wear_violation_share"):
            lines.append(f"| {name} | {sum(row[name] for row in cases) / len(cases):.6g} |")
    lines += [
        "",
        "## Evidence status",
        "",
        report["s4_status"],
        "This report covers one job. The full forecast sweep and G2 review remain open.",
        "",
        "## Provenance",
        "",
    ]
    lines.extend(f"- `{key}` = `{value}`" for key, value in report["provenance"].items())
    return "\n".join([*lines, ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "receipt", "gold", "metrics", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--diagnostics", type=Path, help="Optional verified saved-model S4 recovery receipt")
    parser.add_argument("--exit-code", type=int, required=True, help="Observed training subprocess exit code")
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.run.resolve()):
        parser.error("choose a new output directory outside the imported run")
    report = read_run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        write_json(stage / "review.json", report)
        (stage / "eval_physics.md").write_text(render(report), encoding="utf-8")
        stage.rename(args.out)
    print(args.out / "eval_physics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
