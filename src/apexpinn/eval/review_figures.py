"""Render an evidence-bound partial-study review without model imports or estimation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from apexpinn.cloud.artifacts import digest, write_json

REFERENCE = "https://github.com/ChenLiu-1996/figures4papers"
STYLE_GUIDES = [
    REFERENCE + "/blob/main/scientific-figure-making/" + name
    for name in (
        "SKILL.md",
        "references/design-theory.md",
        "references/api.md",
        "references/common-patterns.md",
    )
]
HORIZONS = (1, 3, 5, 10)
EXPORT_FORMATS = ("png", "svg")
BLUE, GRAY = "#0F4D92", "#767676"


def budget_plan(balance: float = 100, rate: float = 1.94, reserve: float = 20) -> dict:
    if not 0 <= reserve < balance or rate <= 0 or not np.isfinite([balance, rate, reserve]).all():
        raise ValueError("finite positive balance/rate and reserve below balance required")
    return {
        "evidence": "Assumed user-reported balance/rate; inferred arithmetic; not live billing",
        "balance_units": balance,
        "units_per_hour": rate,
        "reserve_units": reserve,
        "reserve_is": "Editable operating assumption, not a research setting or spending cap",
        "total_hours": balance / rate,
        "hours_above_reserve": (balance - reserve) / rate,
        "scope": "Three remaining M2/mid jobs F4-F6, one at a time; not the full 104-job matrix",
        "scenarios": [
            {
                "hours_per_job": h,
                "jobs": 3,
                "hours": 3 * h,
                "units": 3 * h * rate,
                "remaining_units": balance - 3 * h * rate,
                "preserves_reserve": balance - 3 * h * rate >= reserve,
            }
            for h in (8, 12, 16)
        ],
        "limits": "Assumed scenarios, not bounds or confidence intervals. Compilation, fold size, "
        "interruptions, installation and idle allocated time can add cost. Refresh balance before each job.",
    }


def load_evidence(root: Path) -> dict:
    sources = {}

    def read(relative):
        path = root / relative
        sources[relative] = digest(path)
        return json.loads(path.read_text())

    prior = read("data/colab_exports/t13_mid_three_job_review_v1/review.json")
    for name, expected in prior["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"prior review source changed: {name}")
    f3_name = "data/colab_imports/t13_m2_f3_mid_run01_completion_check.json"
    f3 = read(f3_name)
    f3audit = read("data/colab_imports/t13_m2_f3_mid_run01_verification_audit.json")
    if sources[f3_name] != f3audit["completion_check_sha256"] or not f3["execution_complete"]:
        raise ValueError("F3 completion evidence changed")
    pilot_root = "data/colab_imports/calib_clean_m2_f3_mid_v1_checkpoint_01"
    pilot = read(pilot_root + "/evaluation.json")
    check = read("data/colab_imports/calib_clean_m2_f3_mid_v1_completion_check.json")
    if sources[pilot_root + "/evaluation.json"] != check["evaluation_sha256"]:
        raise ValueError("pilot evaluation changed")
    if pilot["status"] != "EVALUATION_COMPLETE" or pilot["plan"]["outer_fold"] != "F3":
        raise ValueError("completed F3 pilot required")
    for name, expected in pilot["files_sha256"].items():
        if digest(root / pilot_root / name) != expected:
            raise ValueError(f"pilot evaluation source changed: {name}")
    f2 = read("data/colab_imports/t13_m2_f2_mid_run01_completion_check.json")
    jobs = [*prior["jobs"], {"model": "M2", "fold": "F3", **f3["metrics"][2]}]
    baseline = [
        *prior["baselines_h5"],
        *[{"model": r["reference"], "fold": "F3", **r} for r in f3["paired_h5_descriptive"]],
    ]
    if len(jobs) != 4 or len(baseline) != 12 or check["model_execution"]:
        raise ValueError("unexpected verified comparison scope")
    return {
        "pilot": pilot,
        "jobs": jobs,
        "baselines": baseline,
        "sources": sources,
        "runtime_evidence": {
            "F2_v1_complete_process_hours": f2["process_elapsed_s"] / 3600,
            "F3_v1_successful_attempt_hours": f3["process_elapsed_s"] / 3600,
            "F3_v1_timing_limit": "Earlier attempt incomplete; successful attempt is not total study time",
            "clean_pilot_successful_attempt_hours": check["process"]["elapsed_s"] / 3600,
            "clean_pilot_limit": "25 fits and different training population; not a v1 runtime benchmark",
        },
    }


def _score(rows, model, h, metric):
    values = [
        r["value"]
        for r in rows
        if r["conditions"]["model"] == model
        and r["conditions"]["horizon_actual_laps"] == h
        and r["quantity"]["metric"] == metric
    ]
    if len(values) != 1:
        raise ValueError(f"missing/duplicate metric: {model}/{h}/{metric}")
    return values[0]


def _finish(fig, path, title, subtitle, note):
    fig.suptitle(title, x=0.07, y=0.98, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.07, 0.915, subtitle, fontsize=12, color="#4D4D4D")
    fig.text(0.07, 0.025, note, fontsize=10, color="#4D4D4D", va="bottom")
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.20, top=0.80, wspace=0.35)
    for suffix in EXPORT_FORMATS:
        fig.savefig(path.with_suffix("." + suffix), dpi=300, facecolor="white")


def plot_calibration(plt, data, out):
    scores = data["pilot"]["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    specs = [
        ("coverage_90", "A  Coverage", "Observed coverage (%)", 100),
        ("width_90", "B  Interval width", "Mean 90% interval width (s)", 1),
        ("crps", "C  Probabilistic error", "CRPS (s); lower is better", 1),
    ]
    for ax, (metric, title, label, multiplier) in zip(axes, specs, strict=True):
        for model, color, marker, style, name in (
            ("M2/raw", GRAY, "o", "--", "Raw"),
            ("M2/calibrated", BLUE, "D", "-", "Calibrated"),
        ):
            y = [_score(scores, model, h, metric) * multiplier for h in HORIZONS]
            ax.plot(HORIZONS, y, color=color, marker=marker, linestyle=style, linewidth=2, label=name)
            ax.annotate(
                f"{y[2]:.2f}",
                (5, y[2]),
                xytext=(4, 8 if name == "Raw" else -17),
                textcoords="offset points",
                color=color,
                fontsize=9,
            )
        ax.set(title=title, xlabel="Prediction horizon (laps)", ylabel=label, xticks=HORIZONS)
        ax.grid(axis="y", alpha=0.18)
        if metric == "coverage_90":
            ax.axhline(90, color="black", linestyle=":", linewidth=1.5, label="Nominal 90%")
            ax.set_ylim(85, 101)
            ax.legend(loc="lower right", fontsize=11)
        else:
            ax.set_ylim(bottom=0)
    _finish(
        fig,
        out / "01_calibration_overview",
        "Calibration narrows intervals; coverage remains conservative",
        "Clean M2 pilot v1 | outer F3, calibration F5, phi = 0.06 | same five models and point predictions",
        "Observed saved scores; h5 values annotated. Targets at h1/3/5/10: 3055 / 2578 / 2202 / 1417.\n"
        "No confidence intervals or significance claim. Source: pilot evaluation.json; see sources.json.",
    )
    plt.close(fig)


def plot_circuits(plt, data, out):
    from matplotlib.colors import TwoSlopeNorm

    rows = data["pilot"]["circuit_metrics"]
    circuits = ("barcelona", "silverstone", "spielberg", "montreal")
    labels = ("Barcelona", "Silverstone", "Spielberg", "Montreal")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")
    norm = TwoSlopeNorm(vmin=80, vcenter=90, vmax=100)
    for ax, model, title in zip(axes, ("M2/raw", "M2/calibrated"), ("A  Raw", "B  Calibrated"), strict=True):
        matrix = np.full((4, 4), np.nan)
        for i, circuit in enumerate(circuits):
            selected = [r for r in rows if r["circuit"] == circuit]
            for j, h in enumerate(HORIZONS):
                if selected:
                    matrix[i, j] = 100 * _score(selected, model, h, "coverage_90")
                    n = next(
                        r["population"]["n_forecasts"]
                        for r in selected
                        if r["conditions"]["model"] == model and r["conditions"]["horizon_actual_laps"] == h
                    )
                    text = f"{matrix[i, j]:.1f}%\nn={n}"
                else:
                    text = "NA"
                value = matrix[i, j]
                rgb = cmap(norm(value))[:3] if np.isfinite(value) else (0.91, 0.91, 0.91)
                luminance = sum(w * c for w, c in zip((0.2126, 0.7152, 0.0722), rgb, strict=True))
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if luminance < 0.5 else "#202020",
                )
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
        ax.set(
            title=title,
            xticks=range(4),
            xticklabels=HORIZONS,
            yticks=range(4),
            yticklabels=labels,
            xlabel="Prediction horizon (laps)",
        )
        ax.tick_params(length=0)
    _finish(
        fig,
        out / "02_circuit_coverage",
        "Circuit detail exposes the limits of pooled coverage",
        "Clean M2 pilot v1 | F3 test circuits | central 90% prediction intervals | phi = 0.06",
        "White = nominal 90%; red = overcoverage; blue = undercoverage. Counts are forecast endpoints.\n"
        "Montreal has no eligible forecasts. Missing values are not zero. Source: pilot evaluation.json.",
    )
    # Reserve a dedicated strip for the shared scale.
    fig.subplots_adjust(left=0.14, right=0.88, wspace=0.48)
    cax = fig.add_axes((0.91, 0.26, 0.015, 0.48))
    fig.colorbar(image, cax=cax, ticks=[80, 90, 100], label="Observed coverage (%)")
    for suffix in EXPORT_FORMATS:
        fig.savefig((out / "02_circuit_coverage").with_suffix("." + suffix), dpi=300, facecolor="white")
    plt.close(fig)


def plot_accuracy(plt, data, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), gridspec_kw={"width_ratios": [2.4, 1]})
    palette = [
        ("M2", BLUE, "D"),
        ("B0", "#272727", "o"),
        ("B1", "#42949E", "s"),
        ("B2", "#B64342", "^"),
        ("B3", "#9A4D8E", "X"),
    ]
    all_rows = data["jobs"] + data["baselines"]
    for k, (model, color, marker) in enumerate(palette):
        selected = [
            next(r for r in all_rows if r["model"] == model and r["fold"] == f"F{i}") for i in (1, 2, 3)
        ]
        y = np.arange(3) + (k - 2) * 0.12
        axes[0].scatter([r["rmse_s"] for r in selected], y, color=color, marker=marker, s=50, label=model)
        if model == "M2":
            for yy, r in zip(y, selected, strict=True):
                axes[0].annotate(
                    f"{r['rmse_s']:.3f}",
                    (r["rmse_s"], yy),
                    xytext=(6, -1),
                    textcoords="offset points",
                    fontsize=9,
                    color=color,
                )
    axes[0].set(
        title="A  M2 and matched baselines",
        xlabel="h5 RMSE (s); lower is better",
        yticks=range(3),
        yticklabels=["F1  (n=2292)", "F2  (n=1990)", "F3  (n=2202)"],
        xlim=(0, 2.85),
    )
    axes[0].invert_yaxis()
    rows = [
        next(r for r in all_rows if r["model"] == m and r["fold"] == "F1")
        for m in ("M1", "M2", "B0", "B1", "B2", "B3")
    ]
    axes[1].scatter([r["rmse_s"] for r in rows], range(6), color=["#B64342", BLUE, *[GRAY] * 4], s=50)
    axes[1].set(
        title="B  M1/F1 retained",
        xlabel="h5 RMSE (s)",
        yticks=range(6),
        yticklabels=[r["model"] for r in rows],
        xlim=(0, 12.5),
    )
    axes[1].invert_yaxis()
    axes[1].annotate(
        f"{rows[0]['rmse_s']:.3f}",
        (rows[0]["rmse_s"], 0),
        xytext=(-4, -16),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    for ax in axes:
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    _finish(
        fig,
        out / "03_original_accuracy",
        "Physics does not consistently beat the strongest baselines",
        "Original T13 v1 only | mid phi = 0.06, h5 | partial study: four verified jobs; clean pilot excluded",
        "Matched keys/targets within each fold. M1/F2-F3 unavailable; unfavorable M1/F1 retained.\n"
        "No new paired CIs. Different panel x ranges. Sources: verified review/check JSON in sources.json.",
    )
    fig.subplots_adjust(left=0.14, top=0.76, wspace=0.38)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=5, fontsize=11, loc="upper left", bbox_to_anchor=(0.14, 0.89))
    for suffix in EXPORT_FORMATS:
        fig.savefig((out / "03_original_accuracy").with_suffix("." + suffix), dpi=300, facecolor="white")
    plt.close(fig)


def build_review_figures(root: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("use a new output directory; preserve previous figures")
    data = load_evidence(root)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    out.mkdir(parents=True)
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next((f for f in ("DejaVu Sans", "Helvetica", "Arial") if f in installed), "sans-serif")
    with plt.rc_context(
        {
            "font.family": family,
            "font.size": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 2,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    ):
        plot_calibration(plt, data, out)
        plot_circuits(plt, data, out)
        plot_accuracy(plt, data, out)
    budget = {**budget_plan(), "observed_runtime_context": data["runtime_evidence"]}
    write_json(out / "budget.json", budget)
    write_json(
        out / "chart_data.json",
        {
            "pilot_metrics": data["pilot"]["metrics"],
            "pilot_circuit_metrics": data["pilot"]["circuit_metrics"],
            "pilot_population": data["pilot"]["population"],
            "jobs": data["jobs"],
            "baselines": data["baselines"],
            "limits": "Partial evidence; v1 and clean pilot are distinct experiments; CIs not computed",
        },
    )
    notes = [
        "# Partial-study diagnostic figures",
        "",
        "Generated from verified saved records. No training, inference or new interval estimation.",
        "",
        "## Figures",
        "",
        "1. Calibration overview: F3 clean pilot only. Same means/targets before and after scaling.",
        "2. Circuit coverage: each percentage has its endpoint count; Montreal is unavailable.",
        "3. Original v1 accuracy: M2/B0-B3 across F1-F3, with unfavorable M1/F1 retained separately.",
        "",
        "Exports: PNG (300 dpi) and editable-text SVG. Different experiments are not pooled.",
        "No new confidence intervals or significance claims. See chart_data.json and sources.json.",
        "",
        "## Budget, not a runtime guarantee",
        "",
        "User-reported 100 units / 1.94 units per hour = 51.55 L4 hours. No live billing query.",
        "Suggested editable reserve: 20 units, leaving 41.24 hours. One selected fold per launch.",
        "",
        "| Assumed hours per fold | Three-fold units | Units left |",
        "| --- | --- | --- |",
    ]
    for scenario in budget["scenarios"]:
        notes.append(
            f"| {scenario['hours_per_job']} | {scenario['units']:.2f} | {scenario['remaining_units']:.2f} |"
        )
    notes += [
        "",
        "These are scenarios for F4-F6 M2/mid, not the full remaining 104-job matrix.",
        "F2 v1 process: 9.11 h. F3 v1 successful attempt: 5.61 h, excluding earlier attempt.",
        "Clean pilot successful attempt: 10.58 h, with different training data and 25 fits.",
        "Compilation, idle allocated time and interruptions can add cost. Refresh balance before each fold.",
        "Actual total cost and future runtime sessions are Unknown; no billing cap is enforced.",
        "",
        "## Visual reference",
        "",
        REFERENCE,
        "Independent plotting code uses general minimalist publication conventions; no upstream code copied.",
        "",
    ]
    (out / "README.md").write_text("\n".join(notes), encoding="utf-8")
    write_json(
        out / "sources.json",
        {
            "source_sha256": data["sources"],
            "style_reference": REFERENCE,
            "style_guides_read": STYLE_GUIDES,
            "style_adaptations": "Scoped rcParams, 15pt compact-panel text, wider panels, "
            "explicit sample/missing labels and shared 90%-centered diverging coverage scale; "
            "PNG/SVG only per user export preference",
            "export_policy": {"formats": list(EXPORT_FORMATS), "png_dpi": 300, "pdf": "omitted"},
            "style": "Independent Matplotlib implementation; no upstream code copied",
            "builder_sha256": digest(Path(__file__)),
            "model_execution": False,
            "output_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        },
    )
    return {
        "status": "FIGURES_RENDERED",
        "figures": 3,
        "formats": list(EXPORT_FORMATS),
        "out": str(out),
        "model_execution": False,
        "budget": budget,
    }
