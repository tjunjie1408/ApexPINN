"""S1 report builder: S1b recovery, S1c labels, S1d misspecification criteria, wall time, P1 evidence.

See PRD.md section 6.4 (S1b-S1d) and section 8 (pivot ladder), GLOSSARY.md section 6 (labels), and
PLAN.md T5 (pass criteria and the 'Method declarations' block). The report reads the run records written
by apexpinn.train.s1_study and applies the declared rules mechanically. It never decides P1: the P1 section
lists trigger evidence only, and the decision is the user's, recorded in experiments/pivots.md.
"""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from apexpinn.physics.identifiability import SHAPE_LABELS, identifiability_label
from apexpinn.physics.synthetic import MISSPEC_VARIANTS
from apexpinn.train.s1_study import (
    RunSpec,
    StudyConfig,
    gated_names,
    group_of,
    load_records,
    m2_misspec_decision,
    param_names,
    plan_runs,
    profiled_names,
    usable,
)

__all__ = [
    "MISSPEC_CRITERIA_V1",
    "MISSPEC_READING_V1",
    "P1_DECISION_SENTENCE",
    "R2_DECLARATION",
    "classify",
    "evaluate_misspec",
    "load_records",
    "p1_evidence",
    "render_report",
    "unconverged_points",
    "write_report",
]

Records = Mapping[str, Mapping[str, Any]]
Labels = Mapping[tuple, Mapping[str, Any]]

# PLAN T5 misspecification pass criteria, verbatim (declared before running).
MISSPEC_CRITERIA_V1 = (
    "Every parameter labeled identifiable in `S1c` stays within 1.5× its `O6` tolerance "
    "(`ρ_c` 22.5%, `k_c` 37.5%) on all 3 seeds, except where criterion 4 predicts a shift.",
    "The 95% profile-likelihood interval of each identifiable parameter covers its variant truth "
    "on at least 2 of 3 seeds.",
    "Diagnostics flag the misfit where it is visible: residual RMS on scored laps rises above "
    "`noise_s` by at least 20% for `piecewise_wear` and `traffic`, and the collapse detector does "
    "not fire on any run. A variant where the fit is wrong and no diagnostic flags it is recorded "
    "as a silent failure.",
    "`wrong_phi`: the recovered `ρ_c` shift relative to the well-specified run with the same seed "
    "is within 30% of `φ_used − φ_true` for each compound.",
)
# PLAN T5 method declarations that clarify the misspecification criteria, verbatim.
MISSPEC_READING_V1 = (
    '**Reading of the misspecification criteria**: in criterion 1, "except where criterion 4 '
    'predicts a shift" means `ρ_c` under `wrong_phi` is judged by criterion 4 only, while `k_c` '
    "under `wrong_phi` is judged by criterion 1. Criterion 4 must hold for every compound on all 3 "
    "seeds. The criterion 3 residual RMS is `sqrt(data MSE)` on scored laps with the training offsets.",
    "**Criteria per model**: criteria 1 to 4 are evaluated per variant and per fitted model (`M1`, "
    "plus `M2` when included).",
    '**"Labeled identifiable in `S1c`"** (criteria 1 and 2) means the main-study label at the '
    "misspecification noise level (0.05 s): the label of that model for `ρ_c` and `k_c`, and the "
    "profile-only label for `γ_c`, `h_opt,c` and `κ`. A parameter with another label gets "
    "`not_applicable` for criteria 1 and 2.",
    "**Criterion 1** covers the parameters with an `O6` tolerance (`ρ_c`, `k_c`), because the other "
    "parameters have no `O6` tolerance to scale: |relative error| against the variant truth must stay "
    "at or below 1.5 × tolerance.",
    "**Criterion 2** uses the profile runs of the variant (exact ODE likelihood). An interval side "
    "that is unbounded inside the grid counts as covering that side, because the 95% region then "
    "extends at least to the grid edge. The cost: a profile that turns flat under misspecification "
    "passes coverage, so the report prints each seed's misspecification profile shape next to its "
    'coverage. "2 of 3 seeds" is applied as at least ceil(2n / 3) of the n configured seeds (2 at '
    "n = 3).",
    "**Criterion 3** counts a residual ratio (residual RMS / `noise_s`) of at least 1.2 as a flag. Any "
    "collapse flag in a variant's fit runs fails criterion 3.",
    "**Silent failures**: a fit is wrong on a seed when it fails criterion 1 or criterion 4 there. "
    "If that seed's run has a residual ratio below 1.2 and no collapse flag, it is a silent failure. "
    "Reason: criterion 4 is the only test of `ρ_c` under `wrong_phi`, so a wrong shift without a flag "
    "is a wrong fit without a flag. A parameter that is not labeled identifiable has no criterion 1 "
    "check and so no silent-failure check, because the main study gives no reliable reference for a "
    "correct fit of it.",
    "**Missing runs**: a planned run without a finished record gives the outcome `missing`, never "
    "`pass`.",
    "**Unconverged profile points**: grid points where L-BFGS-B reports no convergence still enter "
    "the deviance, because scipy also reports failure on harmless line-search stops. The report counts "
    "them per profile and marks a shape that rests on one: an unconverged point inside the interval or "
    "at a grid point bracketing an endpoint. Such a point can only make a profile look sharper than it "
    "is.",
    "**`M2` in the misspecification study**: included if the slowest main-study `M2` fit took under "
    "600 s.",
    "While any main-study `M2` fit record is missing or failed, `M2` is undecided: part misspec runs "
    "the `M1` and profile runs, and adds the `M2` runs once every main-study `M2` fit has finished. A "
    "failing `M2` fit never blocks the mandatory `M1` runs.",
)
# PLAN T5 method declaration of the mechanical R2 indicator, verbatim.
R2_DECLARATION = (
    "**R2 indicator for P1** (mechanical, not the decision): an `M1` fit stalls when its objective "
    "fell by less than 1% over the last 20% of logged steps while its residual ratio stayed at or "
    "above 1.2, so the data term did not reach the noise level. The indicator is present when `M1` "
    "stalls on at least 3 seeds (all seeds if fewer) at one noise level and the `M2` records of those "
    "seeds exist. `M1` and `M2` recovery and the final normalized loss terms are shown as context. "
    "The `R4` remedies and the tuning budget of `PRD §8` are not checked by code."
)
REALISTIC_NOISE_S = 0.15  # PLAN T5 method declarations: realistic noise for the P1 trigger (Assumed)
E5_REL_EIGENVALUE = 1e-6  # PLAN section 2, E5
RESIDUAL_RATIO_FLAG = 1.2  # criterion 3: residual RMS at least 20% above noise_s
STALL_OBJECTIVE_DROP = 0.01  # R2 indicator: objective fell by less than 1% over the last 20% of the log
CRIT1_FACTOR = 1.5
CRIT4_REL_TOL = 0.3
VISIBLE_MISFIT = ("piecewise_wear", "traffic")
OUTCOMES = ("pass", "fail", "not_applicable", "governed_by_criterion_4", "missing")
P1_DECISION_SENTENCE = "The P1 decision is made by the user in `experiments/pivots.md`."
SMOKE_BANNER = (
    "> **Smoke config.** These runs only check that the pipeline works end to end (one seed, one "
    "noise level, short training, small grids). Smoke numbers do not answer S1."
)
MECHANICAL = "mechanical trigger check, not the decision"


def _nkey(noise: float) -> float:
    return round(float(noise), 6)


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _basis(cfg: StudyConfig, name: str, model: str) -> str:
    """Label basis of a parameter: the model for recovery-gated groups, 'profile' otherwise."""
    return model if group_of(name) in cfg.tolerance_map else "profile"


def _label(labels: Labels, name: str, noise: float, basis: str) -> str:
    return labels.get((name, _nkey(noise), basis), {}).get("label", "missing")


def _profile_entry(record: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    return ((record or {}).get("profiles") or {}).get(name) or {}


def _diag(record: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return (record or {}).get("diagnostics") or {}


def _not_recovered(record: Mapping[str, Any], cfg: StudyConfig) -> list[str]:
    """Gated parameters whose |relative error| exceeds the O6 tolerance (or is not finite)."""
    errs = record.get("rel_error") or {}
    tol = cfg.tolerance_map
    return [n for n in gated_names(cfg) if not (_finite(errs.get(n)) and abs(errs[n]) <= tol[group_of(n)])]


def unconverged_points(entry: Mapping[str, Any] | None) -> tuple[int | None, bool]:
    """(number of unconverged profile points, whether the interval rests on one) of one profile entry.

    A point is unconverged when its stored L-BFGS-B success flag is not true. The interval rests on such a
    point if it lies inside the interval or at a grid point bracketing an endpoint (the grid point at or
    below lo, the one at or above hi, or the grid edge), since those points decide where the deviance
    crosses the threshold. The count is None when the entry stores no per-point success flags.
    """
    grid = (entry or {}).get("grid")
    success = (entry or {}).get("success")
    if not isinstance(grid, list | tuple) or not isinstance(success, list | tuple):
        return None, False
    if not grid or len(grid) != len(success):
        return None, False
    bad = [g for g, ok in zip(grid, success, strict=True) if ok is not True]
    interval = (entry or {}).get("interval") or {}
    lo, hi = interval.get("lo"), interval.get("hi")
    points = [g for g in grid if _finite(g)]
    if not bad or not points or not (_finite(lo) and _finite(hi)):
        return len(bad), False
    lo_bracket = max((g for g in points if g <= lo), default=min(points))
    hi_bracket = min((g for g in points if g >= hi), default=max(points))
    return len(bad), any(_finite(g) and lo_bracket <= g <= hi_bracket for g in bad)


# --- S1c labels ---


def classify(records: Records, cfg: StudyConfig) -> dict[tuple, dict[str, Any]]:
    """S1c labels keyed by (parameter, noise_s rounded to 1e-6, basis).

    basis is the model ('M1', 'M2') for parameters with an O6 tolerance, whose label combines recovery of
    that model on every seed with the profile shapes; it is 'profile' for gamma_c, h_opt_c and kappa,
    labeled from the profile shapes only (identifiability.identifiability_label). A label is 'missing'
    while any planned record it needs is missing, failed, or lacks the profile shape.
    """
    main = plan_runs(cfg, "main")
    tol = cfg.tolerance_map
    labels: dict[tuple, dict[str, Any]] = {}
    for noise in cfg.noise_levels:
        profile_specs = [s for s in main if s.kind == "profile" and s.noise_s == noise]
        for name in profiled_names(cfg):
            shapes: list[str] = []
            profile_missing: list[str] = []
            for s in profile_specs:
                shape = _profile_entry(usable(records.get(s.run_id)), name).get("shape")
                if shape in SHAPE_LABELS:
                    shapes.append(shape)
                else:
                    profile_missing.append(s.run_id)
            group = group_of(name)
            for basis in cfg.models_main if group in tol else ("profile",):
                recovered: list[bool] | None = None
                missing = list(profile_missing)
                if basis != "profile":
                    recovered = []
                    fit_specs = [s for s in main if s.kind == "fit" and s.model == basis]
                    for s in [f for f in fit_specs if f.noise_s == noise]:
                        record = usable(records.get(s.run_id))
                        if record is None:
                            missing.append(s.run_id)
                        else:
                            recovered.append(name not in _not_recovered(record, cfg))
                if missing or not shapes:
                    label = "missing"
                else:
                    label = identifiability_label(recovered, shapes)
                labels[(name, _nkey(noise), basis)] = {
                    "param": name,
                    "noise_s": noise,
                    "basis": basis,
                    "label": label,
                    "shapes": shapes,
                    "recovered": recovered,
                    "missing": missing,
                }
    return labels


# --- S1d criteria ---


def _aggregate(outcomes: Sequence[str]) -> str:
    """fail > missing > pass > governed_by_criterion_4 > not_applicable."""
    for outcome in ("fail", "missing", "pass", "governed_by_criterion_4"):
        if outcome in outcomes:
            return outcome
    return "not_applicable"


def _flagged(record: Mapping[str, Any]) -> bool:
    diag = _diag(record)
    ratio = diag.get("residual_ratio")
    return bool(diag.get("collapse_flags")) or (_finite(ratio) and ratio >= RESIDUAL_RATIO_FLAG)


def _criterion_1(cfg, labels, variant, model, fits, fit_ids, silent) -> dict[str, dict[str, Any]]:
    tol = cfg.tolerance_map
    seeds = cfg.seeds_misspec
    missing_fits = [fit_ids[s] for s in seeds if fits[s] is None]
    out: dict[str, dict[str, Any]] = {}
    for name in gated_names(cfg):
        group = group_of(name)
        if variant == "wrong_phi" and group == "rho":
            out[name] = {"outcome": "governed_by_criterion_4"}
            continue
        label = _label(labels, name, cfg.misspec_noise, model)
        if label == "missing":
            out[name] = {"outcome": "missing", "reason": "main-study label is missing"}
            continue
        if label != "identifiable":
            out[name] = {"outcome": "not_applicable", "label": label}
            continue
        if missing_fits:
            out[name] = {"outcome": "missing", "missing": missing_fits}
            continue
        limit = CRIT1_FACTOR * tol[group]
        errors = [(fits[s].get("rel_error") or {}).get(name) for s in seeds]
        ok = [_finite(e) and abs(e) <= limit for e in errors]
        out[name] = {
            "outcome": "pass" if all(ok) else "fail",
            "rel_errors": errors,
            "limit": limit,
            "run_ids": [fit_ids[s] for s in seeds],
        }
        for seed, passed in zip(seeds, ok, strict=True):
            if not passed and not _flagged(fits[seed]):
                silent.append(
                    {
                        "variant": variant,
                        "model": model,
                        "seed": seed,
                        "param": name,
                        "run_id": fit_ids[seed],
                        "criterion": 1,
                    }
                )
    return out


def _criterion_2(cfg, labels, model, profiles, profile_ids) -> dict[str, dict[str, Any]]:
    seeds = cfg.seeds_misspec
    needed = math.ceil(2 * len(seeds) / 3)
    out: dict[str, dict[str, Any]] = {}
    for name in profiled_names(cfg):
        label = _label(labels, name, cfg.misspec_noise, _basis(cfg, name, model))
        if label == "missing":
            out[name] = {"outcome": "missing", "reason": "main-study label is missing"}
            continue
        if label != "identifiable":
            out[name] = {"outcome": "not_applicable", "label": label}
            continue
        covered = [
            None if profiles[s] is None else _profile_entry(profiles[s], name).get("covers_truth")
            for s in seeds
        ]
        if any(c is None for c in covered):
            out[name] = {"outcome": "missing", "missing": [profile_ids[s] for s in seeds]}
            continue
        entries = [_profile_entry(profiles[s], name) for s in seeds]
        n_covered = sum(bool(c) for c in covered)
        out[name] = {
            "outcome": "pass" if n_covered >= needed else "fail",
            "covered": covered,
            "needed": needed,
            "shapes": [e.get("shape") for e in entries],
            "rests_on_unconverged": [unconverged_points(e)[1] for e in entries],
            "run_ids": [profile_ids[s] for s in seeds],
        }
    return out


def _criterion_3(cfg, variant, fits, fit_ids) -> dict[str, Any]:
    seeds = cfg.seeds_misspec
    missing_fits = [fit_ids[s] for s in seeds if fits[s] is None]
    if missing_fits:
        return {"outcome": "missing", "missing": missing_fits}
    ratios = {s: _diag(fits[s]).get("residual_ratio") for s in seeds}
    collapse = {s: list(_diag(fits[s]).get("collapse_flags") or []) for s in seeds}
    low = [s for s in seeds if not (_finite(ratios[s]) and ratios[s] >= RESIDUAL_RATIO_FLAG)]
    fired = [s for s in seeds if collapse[s]]
    fail = bool(fired) or (variant in VISIBLE_MISFIT and bool(low))
    return {
        "outcome": "fail" if fail else "pass",
        "residual_ratio": ratios,
        "collapse_flags": collapse,
        "ratio_required": variant in VISIBLE_MISFIT,
        "run_ids": [fit_ids[s] for s in seeds],
    }


def _criterion_4(cfg, records, model, fits, fit_ids, silent) -> dict[str, Any]:
    seeds = cfg.seeds_misspec
    shift = MISSPEC_VARIANTS["wrong_phi"].rho_shift
    main_ids = {
        s: RunSpec("main", "fit", model, "well_specified", cfg.misspec_noise, s).run_id for s in seeds
    }
    mains = {s: usable(records.get(rid)) for s, rid in main_ids.items()}
    absent = [fit_ids[s] for s in seeds if fits[s] is None]
    absent += [main_ids[s] for s in seeds if mains[s] is None]
    per_compound: dict[str, dict[str, Any]] = {}
    for name in param_names("rho", cfg.compounds):
        if absent:
            per_compound[name] = {"outcome": "missing", "missing": absent}
            continue
        deltas: list[float | None] = []
        for s in seeds:
            hat = (fits[s].get("estimates") or {}).get(name)
            ref = (mains[s].get("estimates") or {}).get(name)
            deltas.append(hat - ref if _finite(hat) and _finite(ref) else None)
        ok = [d is not None and abs(d - shift) <= CRIT4_REL_TOL * abs(shift) for d in deltas]
        per_compound[name] = {"outcome": "pass" if all(ok) else "fail", "deltas": deltas}
        for seed, passed in zip(seeds, ok, strict=True):
            if not passed and not _flagged(fits[seed]):
                silent.append(
                    {
                        "variant": "wrong_phi",
                        "model": model,
                        "seed": seed,
                        "param": name,
                        "run_id": fit_ids[seed],
                        "criterion": 4,
                    }
                )
    return {
        "outcome": _aggregate([v["outcome"] for v in per_compound.values()]),
        "expected_shift": shift,
        "per_compound": per_compound,
        "run_ids": [fit_ids[s] for s in seeds] + [main_ids[s] for s in seeds],
    }


def evaluate_misspec(records: Records, labels: Labels, cfg: StudyConfig) -> dict[str, Any]:
    """Outcomes of the four PLAN T5 misspecification criteria per variant and model, plus silent failures.

    Outcomes are 'pass', 'fail', 'not_applicable', 'governed_by_criterion_4' (rho_c under wrong_phi in
    criterion 1) or 'missing'. labels come from classify on the main study. silent_failures lists every
    seed that fails criterion 1 or criterion 4 while its run has no flag (ratio < 1.2, no collapse flag).
    """
    decision = m2_misspec_decision(cfg, records)
    models = ("M1", "M2") if decision["include"] else ("M1",)
    noise = cfg.misspec_noise
    seeds = cfg.seeds_misspec
    by_variant: dict[str, dict[str, Any]] = {}
    silent: list[dict[str, Any]] = []
    for variant in cfg.misspec_variants:
        by_variant[variant] = {}
        profile_ids = {s: RunSpec("misspec", "profile", None, variant, noise, s).run_id for s in seeds}
        profiles = {s: usable(records.get(rid)) for s, rid in profile_ids.items()}
        for model in models:
            fit_ids = {s: RunSpec("misspec", "fit", model, variant, noise, s).run_id for s in seeds}
            fits = {s: usable(records.get(rid)) for s, rid in fit_ids.items()}
            c1 = _criterion_1(cfg, labels, variant, model, fits, fit_ids, silent)
            c2 = _criterion_2(cfg, labels, model, profiles, profile_ids)
            c3 = _criterion_3(cfg, variant, fits, fit_ids)
            if variant == "wrong_phi":
                c4 = _criterion_4(cfg, records, model, fits, fit_ids, silent)
            else:
                c4 = {"outcome": "not_applicable"}
            by_variant[variant][model] = {
                "criterion_1": c1,
                "criterion_2": c2,
                "criterion_3": c3,
                "criterion_4": c4,
                "summary": {
                    "criterion_1": _aggregate([v["outcome"] for v in c1.values()]),
                    "criterion_2": _aggregate([v["outcome"] for v in c2.values()]),
                    "criterion_3": c3["outcome"],
                    "criterion_4": c4["outcome"],
                },
            }
    return {
        "models": list(models),
        "m2_decision": decision,
        "by_variant": by_variant,
        "silent_failures": silent,
    }


# --- P1 trigger evidence ---


def _r1(labels: Labels, cfg: StudyConfig) -> dict[str, Any]:
    """R1 trigger: k_c or h_opt_c non-identifiable at realistic noise."""
    if _nkey(REALISTIC_NOISE_S) not in {_nkey(n) for n in cfg.noise_levels}:
        reason = f"noise {REALISTIC_NOISE_S:g} s is not in this config"
        return {"status": "not_evaluable", "reason": reason}
    entries: list[dict[str, str]] = []
    not_profiled: list[str] = []
    for group in ("k", "h_opt"):
        if group not in cfg.profile_params:
            not_profiled.append(group)
            continue
        for name in param_names(group, cfg.compounds):
            for basis in cfg.models_main if group in cfg.tolerance_map else ("profile",):
                label = _label(labels, name, REALISTIC_NOISE_S, basis)
                entries.append({"param": name, "basis": basis, "label": label})
    found = [e for e in entries if e["label"] == "non_identifiable"]
    incomplete = bool(not_profiled) or any(e["label"] == "missing" for e in entries)
    if found:
        status = "trigger_condition_met"
    elif incomplete:
        status = "incomplete"
    else:
        status = "trigger_condition_not_met"
    return {
        "status": status,
        "entries": entries,
        "non_identifiable": found,
        "weakly_identifiable": [e for e in entries if e["label"] == "weakly_identifiable"],
        "not_profiled": not_profiled,
    }


def _stalled(record: Mapping[str, Any]) -> bool | None:
    """R2 stall: objective fell by less than 1% over the last 20% of the log and residual ratio >= 1.2.

    None when the objective change or the residual ratio is not available (not evaluable).
    """
    change = (record.get("convergence") or {}).get("objective_rel_change_last20")
    ratio = _diag(record).get("residual_ratio")
    if not (_finite(change) and _finite(ratio)):
        return None
    return change > -STALL_OBJECTIVE_DROP and ratio >= RESIDUAL_RATIO_FLAG


def _final_normalized(record: Mapping[str, Any]) -> dict[str, float]:
    """Last finite logged value of each normalized loss term (term / its initial normalizer)."""
    normalized = (record.get("convergence") or {}).get("normalized") or {}
    out: dict[str, float] = {}
    if not isinstance(normalized, Mapping):
        return out
    for term, values in normalized.items():
        finite = [v for v in (values if isinstance(values, list | tuple) else [values]) if _finite(v)]
        if finite:
            out[str(term)] = float(finite[-1])
    return out


def _r2(records: Records, cfg: StudyConfig) -> dict[str, Any]:
    """R2 indicator: M1 stalls on at least min(3, n_seeds) seeds at one noise level, with M2 records present.

    Recovery of M1 and M2 and the final normalized loss terms are context only (R2_DECLARATION).
    """
    main = plan_runs(cfg, "main")
    need = min(3, len(cfg.seeds_main))
    rows: list[dict[str, Any]] = []
    incomplete = present = False
    for noise in cfg.noise_levels:
        stalled_m1: set[int] = set()
        m2_present: set[int] = set()
        unknown = False
        for s in main:
            if s.kind != "fit" or s.noise_s != noise:
                continue
            row: dict[str, Any] = {"run_id": s.run_id, "model": s.model, "noise_s": noise, "seed": s.seed}
            record = usable(records.get(s.run_id))
            if record is None:
                unknown = unknown or s.model == "M1"
                row.update(status="missing", stalled=None, not_recovered=None, objective_change=None)
                rows.append({**row, "residual_ratio": None, "final_normalized": None})
                continue
            stalled = _stalled(record)
            if s.model == "M1":
                if stalled is None:
                    unknown = True
                elif stalled:
                    stalled_m1.add(s.seed)
            else:
                m2_present.add(s.seed)
            change = (record.get("convergence") or {}).get("objective_rel_change_last20")
            row.update(status=record.get("status"), stalled=stalled, objective_change=change)
            row.update(not_recovered=_not_recovered(record, cfg))
            row.update(residual_ratio=_diag(record).get("residual_ratio"))
            rows.append({**row, "final_normalized": _final_normalized(record)})
        if stalled_m1 and len(stalled_m1) >= need:
            if "M2" in cfg.models_main and not stalled_m1 <= m2_present:
                incomplete = True
            else:
                present = True
        elif unknown:
            incomplete = True
    status = "indicator_present" if present else ("incomplete" if incomplete else "indicator_absent")
    return {"status": status, "need_seeds": need, "rows": rows}


def _e5(records: Records, cfg: StudyConfig) -> dict[str, Any]:
    """E5 is refuted by a relative Gauss-Newton eigenvalue below 1e-6 at the lowest noise level."""
    low = min(cfg.noise_levels)
    rows: list[dict[str, Any]] = []
    for s in plan_runs(cfg, "main"):
        if s.kind == "profile" and s.noise_s == low:
            value = ((usable(records.get(s.run_id)) or {}).get("spectrum") or {}).get("smallest_relative")
            rows.append({"run_id": s.run_id, "seed": s.seed, "smallest_relative": value})
    values = [r["smallest_relative"] for r in rows]
    if any(_finite(v) and v < E5_REL_EIGENVALUE for v in values):
        status = "refuted"
    elif not all(_finite(v) for v in values):
        status = "incomplete"
    else:
        status = "not_refuted"
    return {"status": status, "noise_s": low, "threshold": E5_REL_EIGENVALUE, "rows": rows}


def p1_evidence(records: Records, labels: Labels, cfg: StudyConfig) -> dict[str, Any]:
    """Mechanical checks of the PRD section 8 P1 triggers (R1, R2) and premise E5. Never the decision."""
    return {"r1": _r1(labels, cfg), "r2": _r2(records, cfg), "e5": _e5(records, cfg), "note": MECHANICAL}


# --- rendering ---


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        if value != 0 and (abs(value) < 10 ** (-digits) or abs(value) >= 1e5):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    return f"{100 * value:+.1f}%" if _finite(value) else "n/a"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(":---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _planned(cfg: StudyConfig, decision: Mapping[str, Any]) -> list[RunSpec]:
    specs = plan_runs(cfg, "main")
    if cfg.seeds_misspec and cfg.misspec_variants:
        specs += plan_runs(cfg, "misspec", bool(decision.get("include")))
    return specs


def _recovery_section(records: Records, cfg: StudyConfig) -> list[str]:
    gated = gated_names(cfg)
    tolerances = ", ".join(f"`{g}_c` {100 * t:g}%" for g, t in sorted(cfg.tolerance_map.items()))
    lines = [
        "## S1b recovery",
        "",
        "Relative error (estimate - truth) / truth of the trained `M1` and `M2` estimates, lap time "
        "only, initialized from the seed's prior draw. Recovered means |relative error| within the "
        f"`O6` tolerance ({tolerances}) for every gated parameter. Residual ratio is residual RMS on "
        "scored laps / noise_s. `Observed`.",
        "",
    ]
    header = ["Run id", "Model", "Noise s", "Seed", "Status", *gated, "Recovered"]
    header += ["Max abs err, ungated", "Residual ratio", "Collapse flags"]
    rows = []
    for spec in plan_runs(cfg, "main"):
        if spec.kind != "fit":
            continue
        record = records.get(spec.run_id)
        row: list[Any] = [spec.run_id, spec.model, _fmt(spec.noise_s, 2), spec.seed]
        if usable(record) is None:
            status = "missing" if record is None else record.get("status", "missing")
            rows.append([*row, status, *["n/a"] * (len(gated) + 4)])
            continue
        errs = record.get("rel_error") or {}
        ungated = [abs(v) for n, v in errs.items() if n not in gated and _finite(v)]
        diag = _diag(record)
        row += [record.get("status"), *[_pct(errs.get(n)) for n in gated]]
        row += [_fmt(not _not_recovered(record, cfg)), _pct(max(ungated)) if ungated else "n/a"]
        row += [_fmt(diag.get("residual_ratio")), ", ".join(diag.get("collapse_flags") or []) or "none"]
        rows.append(row)
    return lines + _table(header, rows) + [""]


def _labels_table(labels: Labels) -> list[str]:
    rows = []
    for key in sorted(labels, key=lambda k: (k[1], k[0], k[2])):
        entry = labels[key]
        recovered = entry["recovered"]
        recovered_text = "n/a" if recovered is None else ", ".join(_fmt(r) for r in recovered)
        shapes = ", ".join(entry["shapes"]) or "n/a"
        rows.append(
            [
                entry["param"],
                _fmt(entry["noise_s"], 2),
                entry["basis"],
                f"**{entry['label']}**",
                shapes,
                recovered_text,
            ]
        )
    return _table(["Parameter", "Noise s", "Basis", "Label", "Profile shapes", "Recovered"], rows)


def _spectrum_table(records: Records, cfg: StudyConfig) -> list[str]:
    rows = []
    for spec in plan_runs(cfg, "main"):
        if spec.kind != "profile":
            continue
        record = usable(records.get(spec.run_id))
        row: list[Any] = [spec.run_id, _fmt(spec.noise_s, 2), spec.seed]
        if record is None:
            rows.append([*row, "missing", "n/a", "n/a", "n/a"])
            continue
        spectrum = record.get("spectrum") or {}
        loadings = [(n, v) for n, v in (spectrum.get("weakest_loadings") or {}).items() if _finite(v)]
        top = sorted(loadings, key=lambda kv: -abs(kv[1]))[:3]
        weakest = ", ".join(f"{n} {_fmt(v, 2)}" for n, v in top) or "n/a"
        mle_sse = (record.get("mle") or {}).get("sse")
        row += [_fmt(mle_sse, 4), _fmt(record.get("sse_truth"), 4)]
        row += [_fmt(spectrum.get("smallest_relative")), weakest]
        rows.append(row)
    header = ["Run id", "Noise s", "Seed", "MLE SSE", "SSE at truth", "Smallest relative eigenvalue"]
    return _table([*header, "Weakest direction"], rows)


def _interval_table(records: Records, cfg: StudyConfig) -> list[str]:
    rows = []
    for spec in plan_runs(cfg, "main"):
        if spec.kind != "profile":
            continue
        record = usable(records.get(spec.run_id))
        for name in profiled_names(cfg):
            entry = _profile_entry(record, name)
            iv = entry.get("interval")
            if not iv:
                rows.append([spec.run_id, name, "n/a", "n/a", "n/a", "missing", "n/a", "n/a"])
                continue
            lo = ("open " if not iv.get("lo_bounded") else "") + _fmt(iv.get("lo"), 4)
            hi = ("open " if not iv.get("hi_bounded") else "") + _fmt(iv.get("hi"), 4)
            truth, mle = _fmt(entry.get("truth"), 4), _fmt(entry.get("mle"), 4)
            covers = _fmt(entry.get("covers_truth"))
            n_bad, rests = unconverged_points(entry)
            shape = f"{entry.get('shape')}{'*' if rests else ''}"
            unconverged = "n/a" if n_bad is None else str(n_bad)
            if rests:
                unconverged += ", interval rests on one"
            rows.append([spec.run_id, name, truth, mle, f"[{lo}, {hi}]", shape, covers, unconverged])
    header = ["Run id", "Parameter", "Truth", "MLE", "95% interval", "Shape", "Covers truth"]
    return _table([*header, "Unconverged points"], rows)


def _identifiability_section(records: Records, labels: Labels, cfg: StudyConfig) -> list[str]:
    lines = [
        "## S1c identifiability",
        "",
        "Labels follow the PLAN T5 method declarations: identifiable if every seed recovers within "
        "the `O6` tolerance and every profile is sharp; non-identifiable if at least half of the "
        "profiles are flat; weakly identifiable otherwise. Parameters without an `O6` tolerance are "
        "labeled from the profiles only (basis `profile`). Profiles use the exact ODE likelihood with "
        "`b_s` concentrated out, so the shapes are shared by `M1` and `M2`; the recovery part of the "
        "label is per model. Shapes and recovery flags are listed per seed, in seed order.",
        "",
        *_labels_table(labels),
        "",
        "### Gauss-Newton spectrum and MLE",
        "",
        "Smallest relative eigenvalue (eigenvalue / largest) of J^T J at the ODE-likelihood MLE, and "
        "the three largest loadings of its eigenvector (the least resolved direction). `Observed`.",
        "",
        *_spectrum_table(records, cfg),
        "",
        "### Profile intervals",
        "",
        "95% profile-likelihood intervals (deviance n log(SSE / SSE_min) <= 3.841) in natural units. "
        "A side marked open is unbounded inside the grid and reports the grid edge. Unconverged points "
        "are grid points where L-BFGS-B reported no convergence; they still enter the deviance, and a "
        "shape marked `*` rests on one (inside the interval or at a bracketing grid point), so it may be "
        "sharper than the true profile. `Observed`.",
        "",
        *_interval_table(records, cfg),
        "",
    ]
    return lines


def _misspec_details(misspec: Mapping[str, Any]) -> list[str]:
    lines = []
    for variant, per_model in misspec["by_variant"].items():
        for model, result in per_model.items():
            parts = []
            for name, entry in result["criterion_1"].items():
                errs = entry.get("rel_errors")
                extra = f" (errors {', '.join(_pct(e) for e in errs)})" if errs else ""
                parts.append(f"C1 {name}: {entry['outcome']}{extra}")
            for name, entry in result["criterion_2"].items():
                cov = entry.get("covered")
                extra = ""
                if cov:
                    marks = entry.get("rests_on_unconverged") or [False] * len(cov)
                    shape_list = entry.get("shapes") or ["n/a"] * len(cov)
                    pairs = zip(shape_list, marks, strict=False)
                    shapes = ", ".join(f"{shape}{'*' if mark else ''}" for shape, mark in pairs)
                    extra = f" (covered on {sum(bool(c) for c in cov)} of {len(cov)} seeds; shapes {shapes})"
                parts.append(f"C2 {name}: {entry['outcome']}{extra}")
            c3 = result["criterion_3"]
            ratio_items = (c3.get("residual_ratio") or {}).items()
            ratios = ", ".join(f"seed {s} {_fmt(r)}" for s, r in ratio_items)
            parts.append(f"C3: {c3['outcome']}" + (f" (residual ratio {ratios})" if ratios else ""))
            for name, entry in (result["criterion_4"].get("per_compound") or {}).items():
                deltas = entry.get("deltas")
                extra = f" (shifts {', '.join(_fmt(d, 4) for d in deltas)})" if deltas else ""
                parts.append(f"C4 {name}: {entry['outcome']}{extra}")
            lines.append(f"- **{variant} / {model}**: " + "; ".join(parts) + ".")
    return lines


def _misspec_section(misspec: Mapping[str, Any], cfg: StudyConfig) -> list[str]:
    decision = misspec["m2_decision"]
    if decision["include"] is None:
        m2_line = (
            f"undecided ({decision['reason']}); the M1 runs proceed, and the M2 misspecification runs "
            "are planned once every main-study M2 fit has finished"
        )
    else:
        verdict = "included" if decision["include"] else "excluded"
        where = f", run {decision['max_run_id']}" if decision.get("max_run_id") else ""
        m2_line = f"{verdict} ({decision['reason']}{where}, measured)"
    rows = []
    for variant, per_model in misspec["by_variant"].items():
        for model, result in per_model.items():
            s = result["summary"]
            rows.append([variant, model, *(s[f"criterion_{i}"] for i in range(1, 5))])
    silent = [
        f"- {f['variant']} / {f['model']} seed {f['seed']}: {f['param']} fails criterion {f['criterion']} "
        f"with no diagnostic flag (run {f['run_id']})."
        for f in misspec["silent_failures"]
    ]
    return [
        "## S1d misspecification",
        "",
        f"Noise {cfg.misspec_noise:g} s, lap time only, seeds {list(cfg.seeds_misspec)}, variants "
        f"{list(cfg.misspec_variants)}. Each variant is compared against the truth it records "
        "(`rho_target`, else `Θ_true`).",
        "",
        "Pass criteria, declared before running (PLAN T5, verbatim):",
        "",
        *[f"{i}. {text}" for i, text in enumerate(MISSPEC_CRITERIA_V1, start=1)],
        "",
        "Declared clarifications (PLAN T5 method declarations, verbatim):",
        "",
        *[f"- {text}" for text in MISSPEC_READING_V1],
        "",
        f"**M2 in the misspecification study**: {m2_line}.",
        "",
        "### Outcomes",
        "",
        *_table(["Variant", "Model", "Criterion 1", "Criterion 2", "Criterion 3", "Criterion 4"], rows),
        "",
        "### Details",
        "",
        "Criterion 2 lists each seed's misspecification profile shape next to coverage, because an open "
        "interval side counts as covering; `*` marks a shape that rests on an unconverged profile point.",
        "",
        *_misspec_details(misspec),
        "",
        "### Silent failures",
        "",
        *(silent or ["- None found among the finished runs."]),
        "",
    ]


def _wall_time_section(records: Records, decision: Mapping[str, Any]) -> list[str]:
    totals: dict[tuple[str, str, str], list[float]] = {}
    fits: dict[tuple[str, str, str], list[float]] = {}
    for record in records.values():
        spec = record.get("spec") or {}
        key = (str(spec.get("part")), str(spec.get("kind")), str(spec.get("model") or "ode"))
        wall = record.get("wall_time_s") or {}
        if _finite(wall.get("total")):
            totals.setdefault(key, []).append(float(wall["total"]))
        if _finite(wall.get("fit")):
            fits.setdefault(key, []).append(float(wall["fit"]))
    rows = []
    for key in sorted(totals):
        values = totals[key]
        max_fit = _fmt(max(fits[key]), 1) if fits.get(key) else "n/a"
        stats = [_fmt(statistics.median(values), 1), _fmt(max(values), 1), _fmt(sum(values), 1)]
        rows.append([*key, len(values), *stats, max_fit])
    total = sum(sum(v) for v in totals.values())
    if decision.get("max_fit_s") is not None:
        slowest = f"{_fmt(decision['max_fit_s'], 1)} s (run {decision['max_run_id']})"
    else:
        slowest = "n/a (no measured main-study M2 fit)"
    header = ["Part", "Kind", "Model", "Runs", "Median total s", "Max total s", "Sum total s", "Max fit s"]
    return [
        "## Wall time",
        "",
        "Measured wall time in seconds per run, read from the JSON records (`Observed`; machine and "
        "device in each record's `meta`). Runs without a record have no measured time, and no "
        "estimate is given for them.",
        "",
        *_table(header, rows),
        "",
        f"Sum over all records: {_fmt(total, 1)} s. Slowest main-study M2 fit: {slowest}; the "
        f"threshold for M2 in the misspecification study is {decision['threshold_s']:g} s.",
        "",
    ]


def _p1_section(evidence: Mapping[str, Any]) -> list[str]:
    r1, r2, e5 = evidence["r1"], evidence["r2"], evidence["e5"]
    lines = [
        "## P1 trigger evidence",
        "",
        "These are mechanical trigger checks against PRD section 8, not the decision. "
        f"{P1_DECISION_SENTENCE} Realistic noise for R1 is {REALISTIC_NOISE_S:g} s (`Assumed`, PLAN "
        "T5 method declarations).",
        "",
        "- **R0 (plan)**: the default rung. It stays unless the user moves to R1 or R2.",
        f"- **R1 trigger** (`k_c` or `h_opt,c` non-identifiable at realistic noise), {MECHANICAL}: "
        f"`{r1['status']}`.",
    ]
    if r1.get("entries"):
        found = ", ".join(f"{e['param']} ({e['basis']})" for e in r1["non_identifiable"]) or "none"
        weak = ", ".join(f"{e['param']} ({e['basis']})" for e in r1["weakly_identifiable"]) or "none"
        lines.append(f"  Non-identifiable: {found}. Weakly identifiable: {weak}.")
    if r1.get("not_profiled"):
        lines.append(f"  Not profiled in this config: {', '.join(r1['not_profiled'])}.")
    if r1.get("reason"):
        lines.append(f"  {r1['reason']}.")
    lines.append(
        f"- **R2 indicator** (`M1` stalls on at least {r2['need_seeds']} seeds at one noise level, "
        f"with the `M2` records of those seeds present), {MECHANICAL}: `{r2['status']}`. Declared rule "
        f"(PLAN T5, verbatim): {R2_DECLARATION}"
    )
    if r2["rows"]:
        rows = []
        for r in r2["rows"]:
            failed = r["not_recovered"]
            failed_text = "n/a" if failed is None else (", ".join(failed) or "none")
            final = sorted((r["final_normalized"] or {}).items())
            final_text = ", ".join(f"{t} {_fmt(v)}" for t, v in final) or "n/a"
            row = [r["run_id"], r["model"], _fmt(r["noise_s"], 2), r["seed"], r["status"]]
            row += [_fmt(r["stalled"]), _fmt(r["objective_change"], 4), _fmt(r["residual_ratio"])]
            rows.append([*row, final_text, failed_text])
        header = ["Run id", "Model", "Noise s", "Seed", "Status", "Stalled", "Objective change, last 20%"]
        header += ["Residual ratio", "Final normalized terms", "Not recovered"]
        lines += ["", *_table(header, rows), ""]
    values = ", ".join(f"{r['run_id']} {_fmt(r['smallest_relative'])}" for r in e5["rows"]) or "none"
    lines.append(
        f"- **E5** (refuted if a relative Gauss-Newton eigenvalue is below {E5_REL_EIGENVALUE:g} at "
        f"noise {e5['noise_s']:g} s), {MECHANICAL}: `{e5['status']}`. Values: {values}."
    )
    return lines + [""]


def _missing_section(records: Records, cfg: StudyConfig, decision: Mapping[str, Any]) -> list[str]:
    missing = []
    for spec in _planned(cfg, decision):
        record = records.get(spec.run_id)
        if usable(record) is not None:
            continue
        if record is None:
            missing.append(f"- {spec.run_id}: no record")
        else:
            error = record.get("error", "no error text")
            missing.append(f"- {spec.run_id}: status {record.get('status')} ({error})")
    lines = ["## Missing runs", ""]
    if decision.get("include") is None and cfg.seeds_misspec and cfg.misspec_variants:
        lines += ["M2 misspecification runs are listed once every main-study M2 fit has a record.", ""]
    return lines + (missing or ["- None: every planned run has a finished record."]) + [""]


def _how_to_run(cfg: StudyConfig) -> list[str]:
    flag = f"--config {cfg.name}" if cfg.name in ("smoke", "full") else "--config <smoke|full>"
    cmd = f"uv run python scripts/run_s1.py {flag}"
    first_profile = next(s.run_id for s in plan_runs(cfg, "main") if s.kind == "profile")
    return [
        "## How to run",
        "",
        "From the repository root, after `uv sync` (uv installs Python 3.12 itself):",
        "",
        "```",
        f"{cmd}                  # part main, then part misspec; finished runs are skipped",
        f"{cmd} --part main",
        f"{cmd} --part misspec   # M2 runs are added once every main-study M2 fit has finished",
        f"{cmd} --report-only    # rebuild this report from the JSON records",
        f"{cmd} --part main --shard 0/2   # machine A",
        f"{cmd} --part main --shard 1/2   # machine B",
        f"{cmd} --part main --only {first_profile}   # one run",
        "```",
        "",
        f"With shards, copy every machine's `{cfg.out_dir}/` into one folder before `--part misspec`. "
        "No full-study wall time has been measured yet; the wall-time table above is the only source. "
        "Before an overnight run, time one profile run with `--only` as above: its record is kept and "
        "the full run skips it.",
        "",
    ]


def render_report(records: Records, cfg: StudyConfig) -> str:
    """Markdown report of the S1 study from the run records (load_records)."""
    labels = classify(records, cfg)
    decision = m2_misspec_decision(cfg, records)
    misspec = evaluate_misspec(records, labels, cfg)
    evidence = p1_evidence(records, labels, cfg)
    lines = [f"# S1 identifiability study (PLAN T5), {cfg.name} config", ""]
    if cfg.name == "smoke":
        lines += [SMOKE_BANNER, ""]
    lines += [
        f"Built from {len(records)} run records in `{cfg.out_dir}/` by `apexpinn.eval.s1_report`. "
        "Every number is `Observed`, read from the record named by its run id. Labels and criterion "
        "outcomes apply the rules declared in PLAN T5 to those numbers. Config: "
        f"{cfg.n_stints} stints, compounds {list(cfg.compounds)}, seeds {list(cfg.seeds_main)}, "
        f"noise {list(cfg.noise_levels)} s, models {list(cfg.models_main)}.",
        "",
    ]
    lines += _recovery_section(records, cfg)
    lines += _identifiability_section(records, labels, cfg)
    lines += _misspec_section(misspec, cfg)
    lines += _wall_time_section(records, decision)
    lines += _p1_section(evidence)
    lines += _missing_section(records, cfg, decision)
    lines += _how_to_run(cfg)
    return "\n".join(lines)


def write_report(path: str | Path, text: str) -> Path:
    """Write the report atomically (tmp file plus os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
