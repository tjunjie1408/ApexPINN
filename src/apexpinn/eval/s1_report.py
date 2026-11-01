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
    V1_MODELS,
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
    "INTERVAL_V2_DECLARATION",
    "M1_DIAG_DECLARATION",
    "M2_V2_DECLARATION",
    "MISSPEC_CRITERIA_V1",
    "MISSPEC_READING_V1",
    "P1_DECISION_SENTENCE",
    "R2_DECLARATION",
    "S1B_V1_FREEZE",
    "S1C_SOLVER_DECLARATIONS",
    "at_bound_points",
    "classify",
    "evaluate_misspec",
    "failed_points",
    "load_records",
    "p1_evidence",
    "render_report",
    "stalled",
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
# PLAN T5 method declarations of the S1c optimizer box and failed ODE solves, verbatim.
S1C_SOLVER_DECLARATIONS = (
    "**Optimizer box for `S1c`** (added 2026-11-01 after the first smoke run stopped on `max_steps` in "
    "two profile runs, before any full-config result was read): the MLE and every profile solve keep each "
    "free coordinate inside a box: `k_c` [0.01, 10], `ρ_c` [0.002, 2], `γ_c` [0.1, 60], `h_opt,c` "
    "[0.02, 0.98], `κ` [0.03, 30] (`physics.identifiability.S1C_BOUNDS`: the prior ranges widened 10× on "
    "each side, and the `h_opt` grid clip). The profiled coordinate is held at its grid value even "
    "outside the box. A solution with a free coordinate on the box edge is recorded and counted in the "
    "report. Reason: the likely cause of the stops (`Inferred`, not confirmed) is the unbounded L-BFGS-B "
    "line search in log coordinates reaching very large `k_c`, where the explicit Tsit5 solve needs more "
    "than 4096 steps.",
    "**Failed ODE solves in `S1c`** (added with the optimizer box; the fallback to the lowest finite "
    "evaluation added after review v1 of this change, before any result was read): an ODE solve that "
    "stops without success gives a non-finite SSE instead of stopping the run. Inside an L-BFGS-B solve, "
    "such an evaluation returns a large finite penalty with a zero gradient, so the line search "
    "backtracks. If the final iterate of a solve is still non-finite, the solve falls back to its lowest "
    "finite evaluation and counts as unconverged. A profile grid point whose solve found no finite "
    "evaluation after its start is a failed point: it has no deviance, it is not counted as unconverged, "
    "and the next grid point starts from the last finite solution. The 95% interval is read outward from "
    "the minimum; if that walk reaches a failed point before the deviance crosses 3.841, that side counts "
    "as unbounded, so the profile is `flat`. The report counts failed points per profile next to the "
    "unconverged points. Reason: a failed point gives no evidence, and reading it as unbounded can only "
    "make a profile look flatter than it is, never sharper.",
)
# PLAN T5 method declaration of interval method v2, verbatim.
INTERVAL_V2_DECLARATION = (
    "**Interval method v2** (declared 2026-11-01, after the seed 0, noise 0.05 s profile was read and "
    "before any other profile run was read; the re-solved bracket fallback added after review v1 of this "
    "change, before any v2 profile was run): each bounded endpoint of a 95% profile interval is found by "
    "Brent's method (`scipy.optimize.brentq`) on the coarse-grid bracket (the last grid point inside the "
    "3.841 threshold and the first outside), where every evaluation is a real profile solve: L-BFGS-B "
    "with the profiled coordinate pinned, warm-started from the coarse solution at the inside bracket "
    "point, with the same box and `maxiter`, and a deviance against the same `SSE_min` as the coarse "
    "profile. The tolerance is `xtol` = 1e-4 in the grid coordinate (log, or logit for `h_opt_c`), with "
    "at most 30 iterations per side. If a refinement solve finds no finite SSE, the re-solved bracket "
    "ends no longer straddle the 3.841 threshold, or Brent's method does not converge, that side keeps "
    "the v1 linear endpoint and the report marks it. Refinement "
    "evaluations where L-BFGS-B reports no convergence are counted per side. Unbounded sides (grid "
    "edge, or a failed coarse point before the crossing) are unchanged. Shapes, coverage and labels use "
    "the v2 interval. Every profile record also stores the v1 linear interval, and the report prints it "
    "as `Superseded`. The seed 0, noise 0.05 s profile is rerun under v2, so every reported interval "
    "uses one method. The quadratic fit in `scripts/make_figures.py` stays a diagnostic only. Reason: "
    "on the coarse grid (factors 0.85, 1.0, 1.15 near the MLE, neighbour deviance 26 to 30) linear "
    "interpolation puts the crossing at about 3.841 / D of the grid step instead of about "
    "sqrt(3.841 / D), so the v1 intervals came out about 2.6 to 2.8 times too narrow (`Inferred`), and "
    "6 of 13 seed 0 intervals missed the truth."
)
INTERVAL_METHODS = {"v2_refined": "v2 refined", "v1_linear": "v1 linear"}
# PLAN T5 method declaration freezing the S1b v1 outcome, verbatim.
S1B_V1_FREEZE = (
    "**S1b v1 outcome** (declared 2026-11-01, after all 20 main fits under the v1 declarations were "
    "read): the S1b recovery outcome under the v1 method declarations is final as read: `M1` recovered "
    "on 0 of 10 main fits and `M2` on 5 of 10. The v1 run records in `experiments/s1/` are kept, never "
    "overwritten or relabeled, and the report keeps showing them. Later method versions are reported "
    "alongside v1 and never replace it."
)
# PLAN T5 method declaration of M2 v2 (multi-start), verbatim.
M2_V2_DECLARATION = (
    "**`M2` v2, multi-start** (declared 2026-11-01, after the v1 `M2` results were read; a post-hoc "
    "method version reported alongside v1, never in place of it; the R1 trigger basis added after "
    "review v1 of this change, before any `M2` v2 fit was run): an `M2` v2 fit trains the v1 `M2` "
    "setup (Adam, learning rate 1e-2, 1000 steps, lap time only) from 3 independent prior draws per "
    "seed (2 in the smoke config): start 0 is the v1 prior draw `fold_in(key, 101)`, start i >= 1 is "
    "the profile MLE start `fold_in(key, 200 + i)`. The fit keeps the start with the lowest final "
    "scored residual RMS (the square root of its final data term with its training offsets); the truth "
    "is never used to select, and normalized objectives are not compared because each start "
    "normalizes by its own initial value. The step budget stays at 1000, so v2 changes only the "
    "starts: on seed 0 the v1 objective fell only 10.1% from step 500 to step 999. Reason: on seed 0 "
    "the v1 `M2` fit stopped at an SSE about 3.1 times the ODE-likelihood MLE SSE on the same data, "
    "while the MLE recovers C2, which points to an optimizer failure from that prior draw "
    "(`Inferred`). Scope: v2 targets optimizer failures only. An `M2` v2 miss while the fit sits at "
    "the noise floor (residual ratio below 1.2) remains evidence of weak identifiability, and no "
    "further optimizer change may be used to remove such misses. Run ids use the model tag `M2v2`. "
    "S1c labels are computed for `M2` and `M2v2` separately; the R1 trigger and the R2 indicator read "
    "only the v1 models (`M1` and `M2` v1), and the `M2` misspecification decision keeps using `M2` v1."
)
MODEL_METHOD = {"M1": "v1", "M2": "v1", "M2v2": "v2"}  # S1b method version per fit model tag
# M2v2 start 0 repeats the v1 M2 fit; a relative residual-ratio gap above this flags drift (check only).
START0_RTOL = 1e-6
# PLAN T5 method declaration of the exploratory M1 diagnostics, verbatim.
M1_DIAG_DECLARATION = (
    "**`M1` exploratory diagnostics** (declared 2026-11-01, after the v1 `M1` results were read): runs "
    "in `experiments/s1_m1_diag/` separate three candidate causes of the v1 `M1` failure (`k_c` "
    "collapsed on 10 of 10 main fits): (a) a longer budget (20000 steps), (b) the physics weight "
    "(thermal, wear and mono weights 0; or thermal and wear weights 10), (c) a bounded latent "
    "`h = tanh(tau * n_h)`. They use seed 0 (seed 1 optional), noise 0.05 s, and otherwise the v1 `M1` "
    "setup. They are exploratory: they are not S1b evidence, they change no v1 or v2 result or label, "
    "and no `M1` method change is declared from them. Any later `M1` method version must be declared "
    "here before its runs are read. The P1 decision stays with the user in `experiments/pivots.md`."
)
M1_DIAG_REPORT = "reports/s1_m1_diag.md"  # generated by scripts/run_m1_diag.py
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

    A point is unconverged when its stored L-BFGS-B success flag is not true and it is not a failed point
    (entry['failed'], when stored per grid point): failed points have no deviance, so the "Unconverged
    profile points" declaration does not cover them (failed_points counts them). The interval rests on
    an unconverged point if it lies inside the interval or at a grid point bracketing an endpoint (the
    grid point at or below lo, the one at or above hi, or the grid edge), since those points decide where
    the deviance crosses the threshold. The count is None when the entry stores no per-point success
    flags.
    """
    grid = (entry or {}).get("grid")
    success = (entry or {}).get("success")
    if not isinstance(grid, list | tuple) or not isinstance(success, list | tuple):
        return None, False
    if not grid or len(grid) != len(success):
        return None, False
    failed = _per_point(entry, "failed") or [False] * len(grid)
    bad = [g for g, ok, f in zip(grid, success, failed, strict=True) if ok is not True and f is not True]
    interval = (entry or {}).get("interval") or {}
    lo, hi = interval.get("lo"), interval.get("hi")
    points = [g for g in grid if _finite(g)]
    if not bad or not points or not (_finite(lo) and _finite(hi)):
        return len(bad), False
    lo_bracket = max((g for g in points if g <= lo), default=min(points))
    hi_bracket = min((g for g in points if g >= hi), default=max(points))
    return len(bad), any(_finite(g) and lo_bracket <= g <= hi_bracket for g in bad)


def _per_point(entry: Mapping[str, Any] | None, key: str) -> list | None:
    """entry[key] if it is a list with one item per grid point, else None."""
    grid = (entry or {}).get("grid")
    values = (entry or {}).get(key)
    if not isinstance(grid, list | tuple) or not isinstance(values, list | tuple):
        return None
    return list(values) if grid and len(grid) == len(values) else None


def failed_points(entry: Mapping[str, Any] | None) -> int | None:
    """Number of failed profile points (no finite ODE-likelihood value after the start) of one profile entry.

    Failed points have no deviance (PLAN T5 method declarations). None when the entry stores no per-point
    failed flags (records from before profile method version 2) or their count differs from the grid.
    """
    failed = _per_point(entry, "failed")
    return None if failed is None else sum(f is True for f in failed)


def at_bound_points(entry: Mapping[str, Any] | None) -> int | None:
    """Number of profile points with a free coordinate on the S1C_BOUNDS box edge, or None if not stored."""
    at_bound = _per_point(entry, "at_bound")
    return None if at_bound is None else sum(bool(names) for names in at_bound)


# --- S1c labels ---


def classify(records: Records, cfg: StudyConfig) -> dict[tuple, dict[str, Any]]:
    """S1c labels keyed by (parameter, noise_s rounded to 1e-6, basis).

    basis is the model ('M1', 'M2', 'M2v2') for parameters with an O6 tolerance, whose label combines
    recovery of that model on every seed with the profile shapes; it is 'profile' for gamma_c, h_opt_c and
    kappa, labeled from the profile shapes only (identifiability.identifiability_label). A label is
    'missing' while any planned record it needs is missing, failed, or lacks the profile shape.
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
    """R1 trigger: k_c or h_opt_c non-identifiable at realistic noise.

    Only the v1 models are a basis: M2v2 labels neither enter the entries nor keep R1 incomplete
    (M2_V2_DECLARATION).
    """
    if _nkey(REALISTIC_NOISE_S) not in {_nkey(n) for n in cfg.noise_levels}:
        reason = f"noise {REALISTIC_NOISE_S:g} s is not in this config"
        return {"status": "not_evaluable", "reason": reason}
    entries: list[dict[str, str]] = []
    not_profiled: list[str] = []
    v1_models = tuple(m for m in cfg.models_main if m in V1_MODELS)
    for group in ("k", "h_opt"):
        if group not in cfg.profile_params:
            not_profiled.append(group)
            continue
        for name in param_names(group, cfg.compounds):
            for basis in v1_models if group in cfg.tolerance_map else ("profile",):
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


def stalled(record: Mapping[str, Any]) -> bool | None:
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

    Recovery of M1 and M2 and the final normalized loss terms are context only (R2_DECLARATION). Only the
    v1 models count: M2v2 records neither fill in a missing M2 record nor appear in the rows
    (M2_V2_DECLARATION).
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
            if s.kind != "fit" or s.noise_s != noise or s.model not in V1_MODELS:
                continue
            row: dict[str, Any] = {"run_id": s.run_id, "model": s.model, "noise_s": noise, "seed": s.seed}
            record = usable(records.get(s.run_id))
            if record is None:
                unknown = unknown or s.model == "M1"
                row.update(status="missing", stalled=None, not_recovered=None, objective_change=None)
                rows.append({**row, "residual_ratio": None, "final_normalized": None})
                continue
            is_stalled = stalled(record)
            if s.model == "M1":
                if is_stalled is None:
                    unknown = True
                elif is_stalled:
                    stalled_m1.add(s.seed)
            elif s.model == "M2":
                m2_present.add(s.seed)
            change = (record.get("convergence") or {}).get("objective_rel_change_last20")
            row.update(status=record.get("status"), stalled=is_stalled, objective_change=change)
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
        "Relative error (estimate - truth) / truth of the trained `M1`, `M2` and `M2v2` estimates, lap "
        "time only. `M1` and `M2` (S1b v1) start from the seed's prior draw; `M2v2` (`M2` v2) keeps the "
        "best of several prior draws by final scored residual RMS. Recovered means |relative error| "
        f"within the `O6` tolerance ({tolerances}) for every gated parameter. Residual ratio is residual "
        "RMS on scored laps / noise_s. `Observed`.",
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
    lines += _table(header, rows) + [""]
    lines += _recovery_tally(records, cfg)
    lines += _m2_versions_table(records, cfg)
    lines += [
        "Declared S1b method versions (PLAN T5 method declarations, verbatim):",
        "",
        f"- {S1B_V1_FREEZE}",
        f"- {M2_V2_DECLARATION}",
        "",
    ]
    return lines


def _recovered(record: Mapping[str, Any] | None, cfg: StudyConfig) -> bool | None:
    """Whether a finished fit record recovers every gated parameter; None for a missing or failed one."""
    return None if usable(record) is None else not _not_recovered(record, cfg)


def _recovery_tally(records: Records, cfg: StudyConfig) -> list[str]:
    """Recovered fits per model and noise level, plus a row over all noise levels per model."""
    fits = [s for s in plan_runs(cfg, "main") if s.kind == "fit"]
    rows = []
    for model in cfg.models_main:
        for noise in (*cfg.noise_levels, None):
            specs = [s for s in fits if s.model == model and (noise is None or s.noise_s == noise)]
            flags = [_recovered(records.get(s.run_id), cfg) for s in specs]
            noise_text = "all" if noise is None else _fmt(noise, 2)
            recovered = f"{sum(f is True for f in flags)} of {len(specs)}"
            rows.append([model, MODEL_METHOD.get(model, "n/a"), noise_text, recovered, flags.count(None)])
    return [
        "### Recovered fits per model and noise level",
        "",
        "Count of planned main fits that recover every gated parameter, per model and noise level, and "
        "over all noise levels. Missing counts planned fits without a finished record; they are never "
        "counted as recovered. The v1 rows (`M1`, `M2`) are the frozen S1b v1 outcome; `M2v2` is `M2` "
        "v2, reported alongside v1. `Observed`.",
        "",
        *_table(["Model", "Method version", "Noise s", "Recovered", "Missing"], rows),
        "",
    ]


def _start_ratios(diag: Mapping[str, Any]) -> str:
    """'index: residual ratio' per stored M2v2 start, or 'n/a'."""
    starts = diag.get("starts")
    if not isinstance(starts, list) or not starts:
        return "n/a"
    parts = []
    for i, item in enumerate(starts):
        entry = item if isinstance(item, Mapping) else {}
        parts.append(f"{entry.get('index', i)}: {_fmt(entry.get('residual_ratio'))}")
    return ", ".join(parts)


def _version_cells(record: Mapping[str, Any] | None, cfg: StudyConfig) -> list[str]:
    """Recovered, residual ratio and not-recovered parameters of one fit, or 'missing' three times."""
    if usable(record) is None:
        return ["missing"] * 3
    failed = ", ".join(_not_recovered(record, cfg)) or "none"
    return [_fmt(_recovered(record, cfg)), _fmt(_diag(record).get("residual_ratio")), failed]


def _start0_check(v1: Mapping[str, Any] | None, v2: Mapping[str, Any] | None) -> str:
    """Whether M2v2 start 0 reproduces the M2 v1 residual ratio of the same seed and noise level.

    Start 0 repeats the v1 fit (same prior draw, train key and setup, M2_V2_DECLARATION), so a relative
    gap above START0_RTOL flags nondeterminism or setup drift. 'yes' or 'no'; 'n/a' when either ratio is
    not available. A consistency check only: it changes no selection, recovery or label.
    """
    if usable(v1) is None or usable(v2) is None:
        return "n/a"
    starts = _diag(v2).get("starts")
    entries = [s for s in starts if isinstance(s, Mapping)] if isinstance(starts, list) else []
    first = next((s for s in entries if s.get("index") == 0), {})
    a, b = _diag(v1).get("residual_ratio"), first.get("residual_ratio")
    if not (_finite(a) and _finite(b)):
        return "n/a"
    return "yes" if abs(b - a) <= START0_RTOL * abs(a) else "no"


def _m2_versions_table(records: Records, cfg: StudyConfig) -> list[str]:
    """M2 v1 and M2 v2 side by side per noise level and seed, with the start 0 check."""
    lines = ["### `M2` v1 vs `M2` v2", ""]
    if not {"M2", "M2v2"} <= set(cfg.models_main):
        return lines + ["`M2` and `M2v2` are not both in models_main of this config.", ""]
    rows = []
    drift = []
    for noise in cfg.noise_levels:
        for seed in cfg.seeds_main:
            v1 = records.get(RunSpec("main", "fit", "M2", "well_specified", noise, seed).run_id)
            v2_id = RunSpec("main", "fit", "M2v2", "well_specified", noise, seed).run_id
            v2 = records.get(v2_id)
            row = [_fmt(noise, 2), seed, *_version_cells(v1, cfg), *_version_cells(v2, cfg)]
            if usable(v2) is None:
                row += ["missing", "missing", "missing"]
            else:
                diag = _diag(v2)
                check = _start0_check(v1, v2)
                row += [_fmt(diag.get("selected_start")), _start_ratios(diag), check]
                if check == "no":
                    drift.append(v2_id)
            rows.append(row)
    header = ["Noise s", "Seed", "v1 recovered", "v1 residual ratio", "v1 not recovered"]
    header += ["v2 recovered", "v2 residual ratio", "v2 not recovered", "v2 selected start"]
    header += ["v2 residual ratio per start", "Start 0 matches v1"]
    out = lines + [
        "Per seed: `M2` v1 (one prior draw) and `M2` v2 (best start by final scored residual RMS; the "
        "truth is not used to select). Residual ratio per start lists start index: ratio. `missing`: no "
        "finished record. `Observed`.",
        "",
        "Start 0 matches v1: start 0 repeats the v1 fit (same prior draw, train key and setup), so its "
        f"residual ratio should equal the v1 ratio within a relative {START0_RTOL:g}; `no` points to "
        "nondeterminism or setup drift. A consistency check only; it changes no selection, recovery or "
        "label.",
        "",
        *_table(header, rows),
        "",
    ]
    if drift:
        out += [
            f"**Warning**: `M2v2` start 0 does not reproduce the `M2` v1 residual ratio in {len(drift)} "
            f"record(s): {', '.join(drift)}.",
            "",
        ]
    return out


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
            rows.append([*row, "missing", "n/a", "n/a", "n/a", "n/a"])
            continue
        spectrum = record.get("spectrum") or {}
        loadings = [(n, v) for n, v in (spectrum.get("weakest_loadings") or {}).items() if _finite(v)]
        top = sorted(loadings, key=lambda kv: -abs(kv[1]))[:3]
        weakest = ", ".join(f"{n} {_fmt(v, 2)}" for n, v in top) or "n/a"
        mle = record.get("mle") or {}
        at_bound = mle.get("at_bound")
        edge = "n/a" if not isinstance(at_bound, list) else (", ".join(map(str, at_bound)) or "none")
        row += [_fmt(mle.get("sse"), 4), _fmt(record.get("sse_truth"), 4)]
        row += [_fmt(spectrum.get("smallest_relative")), weakest, edge]
        rows.append(row)
    header = ["Run id", "Noise s", "Seed", "MLE SSE", "SSE at truth", "Smallest relative eigenvalue"]
    return _table([*header, "Weakest direction", "MLE at box edge"], rows)


def _side_text(interval: Mapping[str, Any], side: str) -> str:
    """Prefix of one interval endpoint: 'open at failed point ', 'open ' or ''."""
    if interval.get(f"{side}_bounded"):
        return ""
    return "open at failed point " if interval.get(f"{side}_failed") is True else "open "


def _interval_text(interval: Mapping[str, Any]) -> str:
    lo = _side_text(interval, "lo") + _fmt(interval.get("lo"), 4)
    hi = _side_text(interval, "hi") + _fmt(interval.get("hi"), 4)
    return f"[{lo}, {hi}]"


def _interval_method(entry: Mapping[str, Any]) -> str:
    """'v2 refined' or 'v1 linear'; an entry without interval_method predates profile method version 3."""
    return INTERVAL_METHODS.get(str(entry.get("interval_method")), "v1 linear")


def _refinement_text(entry: Mapping[str, Any]) -> str:
    """Per side: Brent evaluations, or why the side kept its v1 linear endpoint; then unconverged count."""
    refinement = entry.get("refinement")
    if not isinstance(refinement, Mapping):
        return "n/a"
    parts, n_unconverged = [], 0
    for side in ("lo", "hi"):
        info = refinement.get(side) or {}
        reason = info.get("reason")
        if info.get("refined") is True:
            parts.append(f"{side} {_fmt(info.get('n_evals'))} evals")
        elif reason == "unbounded":
            parts.append(f"{side} unbounded")
        else:
            parts.append(f"{side} kept v1 linear ({reason or 'no reason stored'})")
        count = info.get("n_unconverged")
        n_unconverged += count if isinstance(count, int) and not isinstance(count, bool) else 0
    return ", ".join(parts) + f", {n_unconverged} unconverged"


def _interval_methods_line(records: Records, cfg: StudyConfig) -> list[str]:
    """Count of main profile records per interval method, with a warning when the methods mix."""
    counts = dict.fromkeys(INTERVAL_METHODS.values(), 0)
    for spec in plan_runs(cfg, "main"):
        record = usable(records.get(spec.run_id))
        if spec.kind != "profile" or record is None:
            continue
        entries = [e for e in (record.get("profiles") or {}).values() if isinstance(e, Mapping)]
        entries = [e for e in entries if e.get("interval")]
        if entries:
            v2 = any(_interval_method(e) == "v2 refined" for e in entries)
            counts["v2 refined" if v2 else "v1 linear"] += 1
    text = ", ".join(f"{method} {n}" for method, n in counts.items())
    lines = [f"Interval method per main profile record: {text}."]
    if all(counts.values()):
        lines += [
            "",
            "**Warning**: the profile records mix interval methods, so the shapes, coverage and labels "
            "below mix v1 linear and v2 refined intervals until every profile run is rerun under v2.",
        ]
    return lines


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
                rows.append([spec.run_id, name, "n/a", "n/a", "n/a", "missing", *["n/a"] * 7])
                continue
            truth, mle = _fmt(entry.get("truth"), 4), _fmt(entry.get("mle"), 4)
            covers = _fmt(entry.get("covers_truth"))
            linear = entry.get("interval_linear")
            method = [_interval_method(entry), _interval_text(linear) if linear else "n/a"]
            method.append(_refinement_text(entry))
            n_bad, rests = unconverged_points(entry)
            shape = f"{entry.get('shape')}{'*' if rests else ''}"
            unconverged = "n/a" if n_bad is None else str(n_bad)
            if rests:
                unconverged += ", interval rests on one"
            n_failed, n_edge = failed_points(entry), at_bound_points(entry)
            counts = [unconverged, *("n/a" if n is None else str(n) for n in (n_failed, n_edge))]
            rows.append([spec.run_id, name, truth, mle, _interval_text(iv), shape, covers, *method, *counts])
    header = ["Run id", "Parameter", "Truth", "MLE", "95% interval", "Shape", "Covers truth"]
    header += ["Interval method", "v1 linear interval (Superseded)", "Refinement"]
    return _table([*header, "Unconverged points", "Failed points", "At box edge"], rows)


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
        "the three largest loadings of its eigenvector (the least resolved direction). MLE at box edge "
        "lists the MLE coordinates on an edge of the optimizer box (`S1C_BOUNDS`). `Observed`.",
        "",
        *_spectrum_table(records, cfg),
        "",
        "### Profile intervals",
        "",
        "95% profile-likelihood intervals (deviance n log(SSE / SSE_min) <= 3.841) in natural units. "
        "A side marked open is unbounded inside the grid and reports the grid edge; a side marked open at "
        "failed point met a failed point before the threshold and reports the last finite grid point "
        "inside. Unconverged points "
        "are grid points where L-BFGS-B reported no convergence; they still enter the deviance, and a "
        "shape marked `*` rests on one (inside the interval or at a bracketing grid point), so it may be "
        "sharper than the true profile. Failed points are grid points whose solve found no finite "
        "ODE-likelihood value after its start (not counted among the unconverged points); they have no "
        "deviance, and a side that reaches one before the threshold is open. At box edge counts grid "
        "points with a free coordinate on the box edge. `n/a`: the record does not store the count. "
        "`Observed`.",
        "",
        "Endpoints follow interval method v2 (PLAN T5 method declarations, verbatim):",
        "",
        f"- {INTERVAL_V2_DECLARATION}",
        "",
        "Interval method is `v2 refined` for records of profile method version 3 and `v1 linear` for "
        "older records. The v1 linear interval column is `Superseded` and shown for comparison only. "
        "Refinement lists the Brent evaluations per side (or why a side kept its v1 linear endpoint) and "
        "the refinement evaluations where L-BFGS-B reported no convergence.",
        "",
        *_interval_methods_line(records, cfg),
        "",
        *_interval_table(records, cfg),
        "",
        "Declared treatment of the optimizer box and failed ODE solves (PLAN T5 method declarations, "
        "verbatim):",
        "",
        *[f"- {text}" for text in S1C_SOLVER_DECLARATIONS],
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
    lines.append(
        f"- **`M1` exploratory diagnostics** (PLAN T5, verbatim): {M1_DIAG_DECLARATION} Their runs and "
        f"mechanical checks are in `{M1_DIAG_REPORT}`, built by `scripts/run_m1_diag.py`."
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
