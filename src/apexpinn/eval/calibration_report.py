"""Score the clean pilot on Colab and render saved evaluation artifacts without model imports."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from apexpinn.cloud.artifacts import digest, write_json


def compute_evaluation(raw, calibrated, recipe, plan, trials, fits, run_id) -> dict:
    from apexpinn.eval.protocol import ForecastRecord, score_forecasts

    def key(r):
        return (r["fold"], r["event_id"], r["stint_id"], r["t0"], r["horizon"], r["phi"])

    before, after = {key(r): r for r in raw}, {key(r): r for r in calibrated}
    if not raw or len(before) != len(raw) or len(after) != len(calibrated) or before.keys() != after.keys():
        raise ValueError("raw/calibrated populations differ or duplicate keys")
    if any(before[k][f] != after[k][f] for k in before for f in ("y_true", "y_pred")):
        raise ValueError("calibration must preserve point predictions and targets exactly")
    scores = [
        asdict(m) for rows in (raw, calibrated) for m in score_forecasts([ForecastRecord(**r) for r in rows])
    ]
    circuits = sorted({r["event_id"].split("_", 1)[1] for r in raw})
    strata = []
    for circuit in circuits:
        for rows in (raw, calibrated):
            selected = [r for r in rows if r["event_id"].split("_", 1)[1] == circuit]
            strata.extend(
                {"circuit": circuit, **asdict(m)}
                for m in score_forecasts([ForecastRecord(**r) for r in selected])
            )
    return {
        "schema": "apexpinn.calibration_evaluation/v1",
        "status": "EVALUATION_COMPLETE",
        "evidence": "Observed",
        "method_id": "apexpinn.calibration_clean/v1",
        "run_identity": run_id,
        "plan": plan,
        "recipe": recipe,
        "trials": trials,
        "members": fits,
        "metrics": scores,
        "circuit_metrics": strata,
        "missing_test_circuits": sorted(set(plan["outer_circuits"]) - set(circuits)),
        "population": {"raw_forecasts": len(raw), "calibrated_forecasts": len(calibrated)},
        "conditions": {
            "declared_fit_slots": 25,
            "completed_tuning_fits": sum(t["status"] == "finished" for t in trials),
            "completed_final_fits": len(fits),
            "new_T13_v1_jobs": 0,
            "paired_means_unchanged": True,
            "G2": "NOT ESTABLISHED",
            "P2": "USER DECISION",
            "confidence_intervals": "NOT RUN",
        },
        "limits": "Empirical horizon recalibration; "
        "no coverage guarantee under circuit shift or dependent origins. "
        "Raw/calibrated use identical models; comparisons to v1 change training and noise conditions.",
    }


def _lookup(metrics: list[dict], model: str, horizon: int, metric: str) -> dict:
    return next(
        m
        for m in metrics
        if m["conditions"]["model"] == model
        and m["conditions"]["horizon_actual_laps"] == horizon
        and m["quantity"]["metric"] == metric
    )


def render_report(root: Path) -> dict:
    """Build report and plots from stored scores only. No JAX import, model loading or fitting."""
    started = time.perf_counter()
    evaluation = root / "evaluation.json"
    r = json.loads(evaluation.read_text())
    if r["schema"] != "apexpinn.calibration_evaluation/v1" or r["status"] != "EVALUATION_COMPLETE":
        raise ValueError("completed clean-calibration evaluation required")
    for name, expected in r["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"saved evaluation source changed: {name}")
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted(int(h) for h in r["recipe"]["factors"])
    scores = r["metrics"]
    lines = [
        "# Clean M2 calibration pilot",
        "",
        "Observed results from evaluation.json. "
        "This is a separate post-hoc method, not a replacement T13 v1 result.",
        "",
        f"Outer test: {r['plan']['outer_fold']}; tuning validation: {r['plan']['validation_fold']}; "
        f"calibration: {r['plan']['calibration_fold']}. Models were not refitted after calibration.",
        "",
        "| Horizon | Factor | Calibration origins | Calibration events |",
        "| --- | ---: | ---: | ---: |",
    ]
    for h in horizons:
        p = r["recipe"]["populations"][str(h)]
        lines.append(f"| {h} | {r['recipe']['factors'][str(h)]:.6f} | {p['origins']} | {len(p['events'])} |")
    lines += [
        "",
        "## Independent outer-test results",
        "",
        "| Model | Horizon | Targets | RMSE (s) | CRPS (s) | 50% coverage | 90% coverage | 90% width (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("M2/raw", "M2/calibrated"):
        for h in horizons:

            def fetch(metric, model=model, h=h):
                return _lookup(scores, model, h, metric)

            lines.append(
                f"| {model} | {h} | {fetch('rmse')['population']['n_forecasts']} | "
                f"{fetch('rmse')['value']:.4f} | {fetch('crps')['value']:.4f} | "
                f"{fetch('coverage_50')['value']:.2%} | {fetch('coverage_90')['value']:.2%} | "
                f"{fetch('width_90')['value']:.4f} |"
            )
    lines += [
        "",
        "## Circuit-specific coverage",
        "",
        "| Circuit | Model | Horizon | Targets | 90% coverage | 90% width (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for circuit in sorted({m["circuit"] for m in r["circuit_metrics"]}):
        group = [m for m in r["circuit_metrics"] if m["circuit"] == circuit]
        for model in ("M2/raw", "M2/calibrated"):
            for h in horizons:
                coverage = _lookup(group, model, h, "coverage_90")
                width = _lookup(group, model, h, "width_90")
                lines.append(
                    f"| {circuit} | {model} | {h} | {coverage['population']['n_forecasts']} | "
                    f"{coverage['value']:.2%} | {width['value']:.4f} |"
                )
    lines += [
        "",
        f"Configured circuits with no eligible forecasts: {r['missing_test_circuits']}.",
        "",
        "RMSE/MAE and point predictions are unchanged between raw and calibrated distributions. "
        "Compare coverage with width and CRPS, not coverage alone. No per-circuit coverage guarantee, "
        "statistical-significance claim, new T13 v1 job or G2/P2 decision is made.",
        "",
        r["limits"],
        "",
    ]
    (root / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for model, label in (("M2/raw", "Raw"), ("M2/calibrated", "Calibrated")):
        axes[0].plot(
            horizons,
            [_lookup(scores, model, h, "coverage_90")["value"] for h in horizons],
            marker="o",
            label=label,
        )
        axes[1].plot(
            horizons,
            [_lookup(scores, model, h, "width_90")["value"] for h in horizons],
            marker="o",
            label=label,
        )
    axes[0].axhline(0.9, color="black", linestyle="--", label="Nominal 90%")
    axes[0].set(
        xlabel="Forecast horizon (laps)",
        ylabel="Coverage fraction",
        ylim=(0, 1.02),
        title=f"Outer {r['plan']['outer_fold']}: interval coverage",
    )
    axes[1].set(xlabel="Forecast horizon (laps)", ylabel="Mean interval width (s)", title="Width cost")
    for ax in axes:
        ax.legend()
        ax.set_xticks(horizons)
    fig.savefig(root / "calibration_intervals.png", dpi=150)
    plt.close(fig)
    receipt = {
        "status": "REPORT_COMPLETE",
        "evaluation_sha256": digest(evaluation),
        "render_source_sha256": digest(Path(__file__)),
        "elapsed_s": time.perf_counter() - started,
        "files": {
            name: digest(root / name) for name in ("calibration_report.md", "calibration_intervals.png")
        },
    }
    write_json(root / "report_receipt.json", receipt)
    return receipt
