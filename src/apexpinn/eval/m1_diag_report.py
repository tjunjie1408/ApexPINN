"""Exploratory M1 diagnostics report (PLAN T5 method declarations, `M1` exploratory diagnostics).

Pure functions over loaded run records: the frozen v1 M1 records of experiments/s1/ give the baseline row,
and each variant of apexpinn.train.m1_diag gives one row per seed. The report applies mechanical checks
only (residual ratio, k_c tolerance, the R2 stall rule of s1_report.stalled). It interprets nothing, it is
not S1b evidence, and the P1 decision stays with the user in experiments/pivots.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apexpinn.eval.s1_report import (
    M1_DIAG_DECLARATION,
    P1_DECISION_SENTENCE,
    RESIDUAL_RATIO_FLAG,
    Records,
    _diag,
    _final_normalized,
    _finite,
    _fmt,
    _not_recovered,
    _pct,
    _table,
    stalled,
    write_report,
)
from apexpinn.train.m1_diag import (
    DEFAULT_SEEDS,
    DIAG_NOISE,
    DIAG_OUT_DIR,
    DIAG_REPORT,
    V1_OUT_DIR,
    VARIANTS,
    base_config,
    variant_config,
)
from apexpinn.train.s1_study import (
    DEFAULT_DECISIONS,
    RunSpec,
    StudyConfig,
    gated_names,
    load_records,
    param_names,
    usable,
)

__all__ = ["DIAG_BANNER", "build_m1_diag_report", "diag_rows", "render_m1_diag_report"]

DIAG_BANNER = (
    "Exploratory M1 diagnostics. Not S1b evidence; no S1b result, label or method changes from these runs."
)
BASELINE = "baseline v1"
MISSING = "missing"


def _run_id(seed: int) -> str:
    return RunSpec("main", "fit", "M1", "well_specified", DIAG_NOISE, seed).run_id


def _group_errors(record: Mapping[str, Any], group: str, cfg: StudyConfig) -> str:
    errs = record.get("rel_error") or {}
    names = param_names(group, cfg.compounds)
    return ", ".join(f"{n.split('[', 1)[1].rstrip(']')} {_pct(errs.get(n))}" for n in names)


def _k_within_tolerance(record: Mapping[str, Any], cfg: StudyConfig) -> bool:
    """Whether any k_c has |relative error| at or below the O6 k tolerance (25% in decisions_v1)."""
    errs = record.get("rel_error") or {}
    tol = cfg.tolerance_map["k"]
    return any(_finite(errs.get(n)) and abs(errs[n]) <= tol for n in param_names("k", cfg.compounds))


def _row(
    label: str, run_cfg: StudyConfig, record: Mapping[str, Any] | None, path: str, cfg: StudyConfig
) -> dict[str, Any]:
    """One table row: the run's settings from run_cfg and its numbers from record (None if absent)."""
    w = run_cfg.m1_weights
    row: dict[str, Any] = {
        "label": label,
        "steps": run_cfg.m1_train.n_steps,
        "weights": f"thermal {w.thermal:g}, wear {w.wear:g}, mono {w.mono:g}",
        "latent": run_cfg.m1_latent,
        "source": path,
        "status": MISSING if record is None else str(record.get("status")),
        "residual_ratio": None,
        "ratio_below_flag": None,
        "k_within_tol": None,
        "stalled": None,
    }
    done = usable(record)
    if done is None:
        return row
    gated = gated_names(cfg)
    diag = _diag(done)
    final = _final_normalized(done)
    ratio = diag.get("residual_ratio")
    row.update(
        residual_ratio=ratio,
        ratio_below_flag=(ratio < RESIDUAL_RATIO_FLAG) if _finite(ratio) else None,
        recovered=f"{len(gated) - len(_not_recovered(done, cfg))} of {len(gated)}",
        k_errors=_group_errors(done, "k", cfg),
        rho_errors=_group_errors(done, "rho", cfg),
        k_within_tol=_k_within_tolerance(done, cfg),
        objective_change=(done.get("convergence") or {}).get("objective_rel_change_last20"),
        stalled=stalled(done),
        final_data=final.get("data"),
        final_thermal=final.get("thermal"),
        thermal_rms=diag.get("thermal_residual_rms"),
        fit_s=(done.get("wall_time_s") or {}).get("fit"),
    )
    return row


def _seeds(diag_records: Mapping[str, Records], seeds: Sequence[int]) -> list[int]:
    """The requested seeds plus every seed that has a diagnostic record."""
    found = set(seeds)
    for records in diag_records.values():
        for record in records.values():
            seed = (record.get("spec") or {}).get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                found.add(seed)
    return sorted(found)


def diag_rows(
    diag_records: Mapping[str, Records],
    v1_records: Records,
    cfg: StudyConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    decisions: str | Path = DEFAULT_DECISIONS,
) -> dict[int, list[dict[str, Any]]]:
    """seed -> rows: the v1 baseline row, then one row per variant in VARIANTS order."""
    out: dict[int, list[dict[str, Any]]] = {}
    for seed in _seeds(diag_records, seeds):
        rid = _run_id(seed)
        rows = [_row(BASELINE, cfg, v1_records.get(rid), f"{V1_OUT_DIR}/{rid}.json", cfg)]
        for name in VARIANTS:
            run_cfg = variant_config(name, (seed,), decisions)
            record = (diag_records.get(name) or {}).get(rid)
            rows.append(_row(name, run_cfg, record, f"{run_cfg.out_dir}/{rid}.json", cfg))
        out[seed] = rows
    return out


def _cells(row: Mapping[str, Any]) -> list[str]:
    head = [row["label"], str(row["steps"]), row["weights"], row["latent"], row["status"]]
    if "recovered" not in row:
        fill = MISSING if row["status"] == MISSING else "n/a"
        return [*head, *([fill] * 11), f"`{row['source']}`"]
    return [
        *head,
        _fmt(row["residual_ratio"]),
        row["recovered"],
        row["k_errors"],
        row["rho_errors"],
        _fmt(row["objective_change"], 4),
        _fmt(row["stalled"]),
        _fmt(row["final_data"]),
        _fmt(row["final_thermal"]),
        _fmt(row["thermal_rms"]),
        _fmt(row["fit_s"], 1),
        _fmt(row["k_within_tol"]),
        f"`{row['source']}`",
    ]


HEADER = [
    "Run",
    "Steps",
    "Weights",
    "Latent",
    "Status",
    "Residual ratio",
    "Gated recovered",
    "`k_c` rel. error",
    "`rho_c` rel. error",
    "Objective change, last 20%",
    "R2 stall flag",
    "Final normalized data",
    "Final normalized thermal",
    "Thermal residual RMS",
    "Fit wall time s",
    "Any `k_c` within tolerance",
    "Source",
]


def _check_text(value: bool | None, unknown: str) -> str:
    return unknown if value is None else ("yes" if value else "no")


def render_m1_diag_report(
    diag_records: Mapping[str, Records],
    v1_records: Records,
    cfg: StudyConfig,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    decisions: str | Path = DEFAULT_DECISIONS,
) -> str:
    """Markdown report of the exploratory M1 diagnostics. cfg is the base config (full_config)."""
    tol_k = cfg.tolerance_map["k"]
    rows_by_seed = diag_rows(diag_records, v1_records, cfg, seeds, decisions)
    lines = [
        "# Exploratory M1 diagnostics (PLAN T5)",
        "",
        f"> {DIAG_BANNER}",
        "",
        "Declared in PLAN T5 (verbatim):",
        "",
        f"- {M1_DIAG_DECLARATION}",
        "",
        f"Built by `apexpinn.eval.m1_diag_report` from the variant records in `{DIAG_OUT_DIR}/<variant>/` "
        f"and the v1 `M1` records in `{V1_OUT_DIR}/` (baseline row). Every number is `Observed`, read "
        "from the record named in the Source column. Relative errors are (estimate - truth) / truth. The "
        "R2 stall flag applies the PLAN T5 R2 rule: objective fell by less than 1% over the last 20% of "
        f"logged steps while the residual ratio stayed at or above {RESIDUAL_RATIO_FLAG:g}. Gated "
        "recovered counts `rho_c` and `k_c` within their `O6` tolerance. `missing` marks a run without a "
        "record.",
        "",
        "Variants:",
        "",
    ]
    lines += [f"- `{name}`: {desc}." for name, (desc, _) in VARIANTS.items()]
    lines.append("")
    for seed, rows in rows_by_seed.items():
        lines += [f"## Seed {seed}, noise {DIAG_NOISE:g} s", ""]
        lines += [*_table(HEADER, [_cells(r) for r in rows]), ""]
    lines += ["## Mechanical reading", ""]
    lines.append(
        "Mechanical checks per run, not an interpretation: whether the residual ratio fell below "
        f"{RESIDUAL_RATIO_FLAG:g}, and whether any `k_c` is within {100 * tol_k:g}% of its truth."
    )
    lines.append("")
    for seed, rows in rows_by_seed.items():
        for r in rows:
            na = MISSING if r["status"] == MISSING else "n/a"
            lines.append(
                f"- Seed {seed}, `{r['label']}`: residual ratio below {RESIDUAL_RATIO_FLAG:g}: "
                f"{_check_text(r['ratio_below_flag'], na)}; any `k_c` within {100 * tol_k:g}%: "
                f"{_check_text(r['k_within_tol'], na)}; R2 stall flag: {_check_text(r['stalled'], na)}."
            )
    lines += ["", P1_DECISION_SENTENCE, ""]
    lines += [
        "## How to run",
        "",
        "```",
        "uv run python scripts/run_m1_diag.py                       # every variant, seed 0",
        "uv run python scripts/run_m1_diag.py --variant bounded_h   # one variant (repeatable)",
        "uv run python scripts/run_m1_diag.py --seed 0 --seed 1     # seed 1 is optional",
        "uv run python scripts/run_m1_diag.py --report-only         # rebuild this report",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_m1_diag_report(
    root: str | Path = ".",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    decisions: str | Path = DEFAULT_DECISIONS,
) -> Path:
    """Load the variant and v1 records under root, render the report and write it to DIAG_REPORT."""
    root = Path(root)
    diag_records = {name: load_records(root / DIAG_OUT_DIR / name) for name in VARIANTS}
    v1_records = load_records(root / V1_OUT_DIR)
    v1_m1 = {rid: rec for rid, rec in v1_records.items() if rid.startswith("main_fit_M1_")}
    text = render_m1_diag_report(diag_records, v1_m1, base_config(decisions), seeds, decisions)
    return write_report(root / DIAG_REPORT, text)
