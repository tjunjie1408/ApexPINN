"""Figures and derived numbers for the research record (RECORD.md), built from the existing JSON records.

Inputs: experiments/data_probe/probe_results.json (FastF1 probe), the S1 run records in experiments/s1/
(full config) and experiments/s1_smoke/ (smoke config), configs/decisions_v1.yaml (O6 tolerance), and the
full-config synthetic dataset make_synthetic_dataset(jax.random.key(0)). Outputs: one PNG per figure and
derived_numbers.json in the output folder (reports/figures/ by default, see scripts/make_figures.py).

The module is read-only over the records: it never writes into experiments/ and never changes a study
result, label or report. Smoke records do not answer S1, so every figure title names its source record and
config, and figures built from smoke records say 'smoke'.

The quadratic interval (quadratic_interval) is a post-hoc diagnostic added after the full-config profile
was read. The reported S1c interval is the linear interpolation of likelihood_interval; the quadratic
diagnostic only shows how much a coarse grid can narrow that interval, and it changes no report or label.

Pure computation functions use numpy only. matplotlib is imported inside _pyplot() with the Agg backend, so
importing this module needs neither matplotlib nor a display.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats

from apexpinn.contracts import load_decisions
from apexpinn.physics.identifiability import Interval, likelihood_interval
from apexpinn.physics.residuals import COMPOUNDS
from apexpinn.train.s1_study import GROUPS, MISSPEC_STUDY_VARIANTS, O6_NAMES, group_of, load_records, usable

__all__ = [
    "ARTIFACT_NAMES",
    "CHI2_1_95",
    "FIGURE_NAMES",
    "FigureInputMissing",
    "build_figures",
    "config_label",
    "coord_to_natural",
    "derived_numbers",
    "deviance_at_truth",
    "linear_interval",
    "o6_tolerance",
    "pick_fit_pair",
    "pick_record",
    "plot_fit_convergence",
    "plot_gn_spectrum",
    "plot_intervals",
    "plot_misspec_smoke",
    "plot_probe_pace_pit",
    "plot_probe_warmup",
    "plot_profiles",
    "plot_recovery",
    "plot_synthetic_stints",
    "quadratic_interval",
]

CHI2_1_95 = float(scipy.stats.chi2.ppf(0.95, 1))  # 3.841, the S1c profile threshold
DPI = 150
FIGURE_NAMES = (
    "probe_warmup",
    "probe_pace_pit",
    "synthetic_stints",
    "s1_fit_convergence",
    "s1_recovery",
    "s1_profiles",
    "s1_intervals",
    "s1_gn_spectrum",
    "s1_misspec_smoke",
)
ARTIFACT_NAMES = (*FIGURE_NAMES, "derived_numbers")
PROBE_PATH = "experiments/data_probe/probe_results.json"
FULL_DIR = "experiments/s1"
SMOKE_DIR = "experiments/s1_smoke"
DECISIONS_PATH = "configs/decisions_v1.yaml"
UNITS = {"k": "1/lap", "rho": "s/lap", "gamma": "s", "h_opt": "dimensionless", "kappa": "dimensionless"}
QUADRATIC_LABEL = "quadratic fit, post-hoc diagnostic"
LINEAR_LABEL = "reported 95% interval (linear interpolation)"

Record = Mapping[str, Any]
Records = Mapping[str, Record]
Sources = Sequence[tuple[str, Records]]
Log = Callable[[str], Any]


class FigureInputMissing(Exception):
    """The input a figure needs is missing or empty. build_figures reports the figure as skipped.

    Any other exception from a figure build is reported as failed, and scripts/make_figures.py exits 1.
    """


# --- pure helpers ---


def _floats(values: Sequence[Any] | None) -> np.ndarray:
    """float64 array with None (JSON form of a non-finite value) as NaN; empty for None."""
    if values is None:
        return np.zeros(0, dtype=np.float64)
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _num(value: Any) -> float:
    return float("nan") if value is None else float(value)


def quadratic_interval(
    grid_coord: Sequence[Any], deviance: Sequence[Any], threshold: float = CHI2_1_95
) -> tuple[float, float] | None:
    """Post-hoc diagnostic: threshold crossings of a parabola through the deviance minimum.

    The parabola passes exactly through the minimum finite grid point and its two neighbours, in the
    coordinate the record interpolates in (grid_coord: log for positive parameters, logit for h_opt). It
    returns the crossings (lo, hi) in that coordinate, or None when the minimum is at the grid edge, a
    neighbour is non-finite, or the parabola does not open upward or stays above the threshold.

    Reason: the reported interval (likelihood_interval) interpolates linearly between grid points. On a
    coarse grid around a roughly quadratic profile the chord lies above the curve, so the linear crossing
    sits closer to the minimum and the interval comes out too narrow. This diagnostic shows how large that
    effect is. It is not a reported result and changes no S1 label.
    """
    x, d = _floats(grid_coord), _floats(deviance)
    if x.shape != d.shape:
        raise ValueError(f"grid_coord and deviance must have one shape, got {x.shape} and {d.shape}")
    if x.size < 3:
        return None
    finite = np.isfinite(x) & np.isfinite(d)
    if not np.any(finite):
        return None
    i = int(np.argmin(np.where(finite, d, np.inf)))
    if i == 0 or i == x.size - 1 or not (finite[i - 1] and finite[i + 1]):
        return None
    a, b, c = np.linalg.solve(np.vander(x[i - 1 : i + 2], 3), d[i - 1 : i + 2])
    if not a > 0:
        return None
    disc = b * b - 4.0 * a * (c - threshold)
    if disc < 0:
        return None
    root = math.sqrt(disc)
    return float((-b - root) / (2.0 * a)), float((-b + root) / (2.0 * a))


def linear_interval(grid_coord: Sequence[Any], deviance: Sequence[Any]) -> Interval:
    """The reported S1c interval method: likelihood_interval on the grid coordinate (None as NaN)."""
    return likelihood_interval(_floats(grid_coord), _floats(deviance))


def deviance_at_truth(sse_truth: float, sse_min: float, n_obs: int) -> float:
    """Joint deviance of the true Theta: n_obs * log(sse_truth / sse_min)."""
    if not (math.isfinite(sse_truth) and sse_truth > 0 and math.isfinite(sse_min) and sse_min > 0):
        raise ValueError(f"sse_truth and sse_min must be finite and > 0, got {sse_truth!r}, {sse_min!r}")
    if isinstance(n_obs, bool) or int(n_obs) != n_obs or n_obs < 1:
        raise ValueError(f"n_obs must be a positive integer, got {n_obs!r}")
    return float(int(n_obs) * math.log(sse_truth / sse_min))


def coord_to_natural(group: str, coord: float) -> float:
    """Profile coordinate to natural units, as s1_study._to_natural: logistic for h_opt, exp otherwise."""
    if group == "h_opt":
        return float(1.0 / (1.0 + np.exp(-coord)))
    return float(np.exp(coord))


def _neighbour_steps(deviance: np.ndarray) -> tuple[int | None, float | None, float | None]:
    """(index of the finite minimum, deviance step to the lower neighbour, step to the upper neighbour)."""
    finite = np.isfinite(deviance)
    if not np.any(finite):
        return None, None, None
    i = int(np.argmin(np.where(finite, deviance, np.inf)))
    lower = float(deviance[i - 1] - deviance[i]) if i > 0 and finite[i - 1] else None
    upper = float(deviance[i + 1] - deviance[i]) if i + 1 < deviance.size and finite[i + 1] else None
    return i, lower, upper


def _param_entry(name: str, entry: Record) -> dict[str, Any]:
    group = str(entry.get("group") or group_of(name))
    truth = float(entry["truth"])
    coord = _floats(entry.get("grid_coord"))
    dev = _floats(entry.get("deviance"))
    iv = entry.get("interval")
    reported = None
    if iv is not None:
        reported = {k: iv[k] for k in ("lo", "hi", "lo_bounded", "hi_bounded")}
    quad = quadratic_interval(coord, dev) if dev.size == coord.size else None
    quad_natural = None
    quad_covers = None
    if quad is not None:
        lo, hi = coord_to_natural(group, quad[0]), coord_to_natural(group, quad[1])
        quad_natural = {"lo": lo, "hi": hi}
        quad_covers = bool(lo <= truth <= hi)
    i_min, step_lower, step_upper = _neighbour_steps(dev) if dev.size else (None, None, None)
    return {
        "group": group,
        "truth": truth,
        "mle": _num(entry.get("mle")),
        "shape": entry.get("shape"),
        "reported_interval": reported,
        "reported_covers_truth": entry.get("covers_truth"),
        "quadratic_interval": quad_natural,
        "quadratic_covers_truth": quad_covers,
        "min_index": i_min,
        "deviance_step_lower": step_lower,
        "deviance_step_upper": step_upper,
    }


def derived_numbers(profile_record: Record) -> dict[str, Any]:
    """Per-parameter interval numbers and the record-level deviance at truth of one profile record.

    Per parameter: truth, MLE, the reported interval and its covers_truth flag (copied from the record),
    the quadratic-diagnostic interval in natural units and whether it covers the truth (post-hoc
    diagnostic, see quadratic_interval), and the deviance step from the minimum to each neighbouring grid
    point. Record level: deviance_at_truth with sse_min the lowest of the MLE SSE and every profile's
    sse_min, compared with the chi2(n_params) 95% quantile, where n_params counts the MLE values.
    """
    mle = profile_record.get("mle") or {}
    profiles = profile_record.get("profiles") or {}
    params = {name: _param_entry(name, entry) for name, entry in sorted(profiles.items())}
    candidates = [_num(mle.get("sse"))] + [_num(e.get("sse_min")) for e in profiles.values()]
    finite = [s for s in candidates if math.isfinite(s) and s > 0]
    sse_min = min(finite) if finite else None
    sse_truth = profile_record.get("sse_truth")
    n_obs = profile_record.get("n_obs")
    n_params = len(mle.get("values") or {})
    dev_truth = None
    if sse_min is not None and sse_truth is not None and n_obs:
        dev_truth = deviance_at_truth(float(sse_truth), sse_min, int(n_obs))
    chi2_joint = float(scipy.stats.chi2.ppf(0.95, n_params)) if n_params else None
    inside = None if dev_truth is None or chi2_joint is None else bool(dev_truth <= chi2_joint)
    return {
        "run_id": profile_record.get("run_id"),
        "config_name": profile_record.get("config_name"),
        "config_hash": profile_record.get("config_hash"),
        "n_obs": n_obs,
        "sse_truth": sse_truth,
        "sse_min": sse_min,
        "deviance_at_truth": dev_truth,
        "n_params": n_params,
        "chi2_95_n_params": chi2_joint,
        "truth_inside_joint_95_region": inside,
        "threshold_1d": CHI2_1_95,
        "quadratic_note": "quadratic_interval is a post-hoc diagnostic; the reported interval is unchanged",
        "params": params,
    }


def o6_tolerance(decisions_path: str | Path) -> dict[str, float]:
    """O6 recovery tolerance per group ('rho', 'k') from the decisions file, as fractions.

    The parameter-to-group mapping is s1_study.O6_NAMES, the one the study itself uses.
    """
    raw = dict(load_decisions(Path(decisions_path)).synthetic.get("tolerance") or {})
    return {O6_NAMES[k]: float(v) for k, v in raw.items() if k in O6_NAMES}


def config_label(record: Record) -> str:
    """'smoke config, does not answer S1' for smoke records, else '<name> config'."""
    name = record.get("config_name")
    return "smoke config, does not answer S1" if name == "smoke" else f"{name} config"


def _source(record: Record) -> str:
    return f"{record.get('run_id')} ({config_label(record)})"


def pick_record(
    sources: Sources,
    *,
    kind: str,
    model: str | None = None,
    part: str = "main",
    variant: str = "well_specified",
) -> tuple[str, Record] | None:
    """First finished record matching the spec, searching sources in order (full before smoke).

    Within one source the lowest noise level, then the lowest seed, wins. model is ignored for profiles.
    Returns (source label, record) or None.
    """
    for label, records in sources:
        found = []
        for record in records.values():
            spec = record.get("spec") or {}
            if usable(record) is None or spec.get("kind") != kind or spec.get("part") != part:
                continue
            if spec.get("variant") != variant or (kind == "fit" and spec.get("model") != model):
                continue
            found.append(record)
        if found:
            return label, min(found, key=_spec_rank)
    return None


def pick_fit_pair(
    sources: Sources, *, part: str = "main", variant: str = "well_specified"
) -> tuple[str, Record | None, Record | None] | None:
    """(source label, M1 record, M2 record) taken from one source only.

    The source is the first one (full before smoke) holding any finished fit for the spec. A model with no
    finished record in that source is None; it is never filled in from a later source, so the fit figures
    never pair a full-config fit with a smoke fit. Returns None when no source has a finished fit.
    """
    for label, records in sources:
        one = [(label, records)]
        m1 = pick_record(one, kind="fit", model="M1", part=part, variant=variant)
        m2 = pick_record(one, kind="fit", model="M2", part=part, variant=variant)
        if m1 is not None or m2 is not None:
            return label, None if m1 is None else m1[1], None if m2 is None else m2[1]
    return None


def _spec_rank(record: Record) -> tuple[float, int, str]:
    spec = record["spec"]
    return _num(spec.get("noise_s")), int(spec.get("seed") or 0), str(record.get("run_id"))


def _param_order(names: Sequence[str]) -> list[str]:
    rank = {g: i for i, g in enumerate(GROUPS)}
    return sorted(set(names), key=lambda n: (rank.get(group_of(n), len(rank)), n))


def _label(name: str) -> str:
    return f"{name} [{UNITS.get(group_of(name), '?')}]"


# --- plotting ---


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, path: str | Path) -> Path:
    plt = _pyplot()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _dev_races(probe: Record | None) -> list[Record]:
    races = [r for r in (probe or {}).get("races") or [] if r.get("role") == "dev"]
    if not races:
        raise FigureInputMissing("probe results have no development races")
    return races


def _iqr(stats: Mapping[str, Any] | None) -> tuple[float, float, float]:
    """(median, lower whisker length, upper whisker length); NaN median and zero whiskers when missing."""
    stats = stats or {}
    med, q25, q75 = _num(stats.get("median")), _num(stats.get("q25")), _num(stats.get("q75"))
    low = med - q25 if math.isfinite(med) and math.isfinite(q25) else 0.0
    high = q75 - med if math.isfinite(med) and math.isfinite(q75) else 0.0
    return med, max(low, 0.0), max(high, 0.0)


def _probe_title(probe: Record, what: str) -> str:
    return f"{what}, development races\nsource: {PROBE_PATH} (preset {probe.get('preset')})"


def plot_probe_warmup(probe: Record | None, path: str | Path) -> Path:
    """Per development race: share of stints with a warm-up dip, and the dip gain median with IQR."""
    races = _dev_races(probe)
    plt = _pyplot()
    names = [str(r.get("event_id")) for r in races]
    x = np.arange(len(races))
    share = [_num((r.get("warmup_visibility") or {}).get("dip_share")) for r in races]
    gains = [_iqr((r.get("warmup_visibility") or {}).get("dip_gain_s")) for r in races]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(x, share, color="C0")
    ax1.set_ylabel("share of stints with a warm-up dip [-]")
    ax1.set_ylim(0, 1)
    ax2.errorbar(
        x,
        [g[0] for g in gains],
        yerr=[[g[1] for g in gains], [g[2] for g in gains]],
        fmt="o",
        capsize=4,
        color="C1",
        label="median, whiskers q25 to q75",
    )
    ax2.set_ylabel("dip gain: first scored lap minus fastest lap, dip stints [s]")
    ax2.legend(loc="upper right")
    for ax in (ax1, ax2):
        ax.set_xticks(x, names, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(_probe_title(probe, "FastF1 probe: warm-up visibility"))
    return _save(fig, path)


def plot_probe_pace_pit(probe: Record | None, path: str | Path) -> Path:
    """Per development race: raw within-stint slope median, and pit-lane time median with IQR."""
    races = _dev_races(probe)
    plt = _pyplot()
    names = [str(r.get("event_id")) for r in races]
    x = np.arange(len(races))
    slope = [_num((r.get("raw_stint_slope_s_per_lap") or {}).get("median")) for r in races]
    pits = [_iqr(r.get("pit_lane_time_s")) for r in races]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(x, slope, color="C0")
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_ylabel("raw within-stint lap-time slope, median [s/lap]")
    ax2.errorbar(
        x,
        [p[0] for p in pits],
        yerr=[[p[1] for p in pits], [p[2] for p in pits]],
        fmt="o",
        capsize=4,
        color="C2",
        label="median, whiskers q25 to q75",
    )
    ax2.set_ylabel("pit-lane time [s]")
    ax2.legend(loc="upper right")
    for ax in (ax1, ax2):
        ax.set_xticks(x, names, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(_probe_title(probe, "FastF1 probe: raw pace slope and pit-lane time"))
    return _save(fig, path)


def plot_synthetic_stints(
    dataset: Any,
    path: str | Path,
    *,
    per_compound: int = 3,
    source: str = "make_synthetic_dataset(jax.random.key(0)), full-config dataset",
) -> Path:
    """A few stints per compound: noisy y - b_s (points) and y_clean - b_s (line) on scored laps only."""
    if dataset is None:
        raise FigureInputMissing("no synthetic dataset")
    plt = _pyplot()
    compound_idx = np.asarray(dataset.compound_idx)
    mask = np.asarray(dataset.mask, dtype=bool)
    tau = np.asarray(dataset.tau, dtype=np.float64)
    y, y_clean = np.asarray(dataset.y, dtype=np.float64), np.asarray(dataset.y_clean, dtype=np.float64)
    b = np.asarray(dataset.b, dtype=np.float64)
    present = sorted({int(c) for c in compound_idx})
    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], present, strict=True):
        stints = [int(s) for s in np.flatnonzero(compound_idx == c)[:per_compound]]
        for j, s in enumerate(stints):
            m = mask[s]
            color = f"C{j}"
            noisy = y[s][m] - b[s]
            ax.plot(tau[m], noisy, "o", ms=3, color=color, alpha=0.6, label=f"stint {s}: y - b_s")
            ax.plot(tau[m], y_clean[s][m] - b[s], "-", color=color, label=f"stint {s}: y_clean - b_s")
        ax.set_title(f"compound {COMPOUNDS[c]}")
        ax.set_xlabel("stint lap tau [lap]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("lap time minus stint offset b_s [s]")
    n_stints = int(compound_idx.shape[0])
    noise = getattr(dataset, "noise_s", None)
    fig.suptitle(
        f"Synthetic stints (scored laps only): warm-up dip and wear the models must recover\n"
        f"source: {source}, {n_stints} stints, noise_s {noise} s"
    )
    return _save(fig, path)


def _positive(values: Sequence[Any] | None) -> np.ndarray:
    arr = _floats(values)
    return np.where(arr > 0, arr, np.nan)


def plot_fit_convergence(m1: Record | None, m2: Record | None, path: str | Path) -> Path:
    """M1 normalized loss terms vs step, and M1 vs M2 objective vs step with residual ratios."""
    if m1 is None:
        raise FigureInputMissing("no finished main M1 fit record")
    plt = _pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    conv = m1.get("convergence") or {}
    steps = _floats(conv.get("steps"))
    for term in ("data", "thermal", "wear", "mono", "reg"):
        values = _positive((conv.get("normalized") or {}).get(term))
        if values.size != steps.size:
            continue
        if not np.any(np.isfinite(values)):
            ax1.plot([], [], " ", label=f"{term}: all zero, not drawn on log axis")
            continue
        ax1.plot(steps, values, "o-", ms=3, label=term)
    ax1.set_yscale("log")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("M1 normalized loss term [-]")
    ax1.set_title("M1 loss terms (each divided by its start value)")
    ax1.legend(fontsize=8)
    for name, record in (("M1", m1), ("M2", m2)):
        if record is None:
            continue
        c = record.get("convergence") or {}
        ratio = _num((record.get("diagnostics") or {}).get("residual_ratio"))
        ax2.plot(
            _floats(c.get("steps")),
            _positive(c.get("objective")),
            "o-",
            ms=3,
            label=f"{name}: residual ratio {ratio:.2f} ({record.get('config_name')})",
        )
    ax2.set_yscale("log")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("normalized objective [-]")
    ax2.set_title("objective; residual ratio = scored residual RMS / noise_s")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    lines = [f"M1: {_source(m1)}"] + ([f"M2: {_source(m2)}"] if m2 is not None else [])
    fig.suptitle("S1 main fits, well specified\n" + "\n".join(lines), fontsize=10)
    return _save(fig, path)


def _profile_rel_error(profile: Record) -> dict[str, float]:
    values = (profile.get("mle") or {}).get("values") or {}
    truth = profile.get("truth") or {}
    return {n: (_num(values[n]) - float(truth[n])) / float(truth[n]) for n in values if n in truth}


def plot_recovery(
    m1: Record | None,
    m2: Record | None,
    profile: Record | None,
    tolerance: Mapping[str, float],
    path: str | Path,
) -> Path:
    """Relative error per parameter for M1, M2 and the ODE-likelihood MLE, with O6 tolerance bands."""
    series = []
    if m1 is not None:
        series.append(("M1", {k: _num(v) for k, v in (m1.get("rel_error") or {}).items()}, m1))
    if m2 is not None:
        series.append(("M2", {k: _num(v) for k, v in (m2.get("rel_error") or {}).items()}, m2))
    if profile is not None:
        series.append(("ODE-likelihood MLE", _profile_rel_error(profile), profile))
    series = [s for s in series if s[1]]
    if not series:
        raise FigureInputMissing("no main fit or profile record with relative errors")
    plt = _pyplot()
    names = _param_order([n for _, errs, _ in series for n in errs])
    x = np.arange(len(names))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(13, 5))
    band_label = "O6 tolerance band (+-tolerance)"
    for i, name in enumerate(names):
        tol = tolerance.get(group_of(name))
        if tol is not None:
            band = 100.0 * tol
            ax.fill_between([i - 0.45, i + 0.45], -band, band, color="C2", alpha=0.2, label=band_label)
            band_label = None
    for j, (label, errs, rec) in enumerate(series):
        values = [100.0 * errs.get(n, float("nan")) for n in names]
        legend = f"{label} ({rec.get('config_name')} config)"
        ax.bar(x - 0.4 + width * (j + 0.5), values, width, label=legend, color=f"C{j}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylabel("relative error (estimate - truth) / truth [%], symlog")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    sources = "\n".join(f"{label}: {_source(rec)}" for label, _, rec in series)
    fig.suptitle(f"S1 parameter recovery, well specified\n{sources}", fontsize=10)
    return _save(fig, path)


def _require_profile(profile: Record | None) -> Record:
    if profile is None or not profile.get("profiles"):
        raise FigureInputMissing("no finished main profile record with profiles")
    return profile


def _bools(values: Sequence[Any] | None, n: int, default: bool) -> np.ndarray:
    if values is None or len(values) != n:
        return np.full(n, default)
    return np.array([bool(v) for v in values])


def _zoom(x: np.ndarray, d: np.ndarray, truth: float) -> tuple[float, float]:
    finite = np.isfinite(d)
    if not np.any(finite):
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    else:
        i = int(np.argmin(np.where(finite, d, np.inf)))
        lo, hi = float(x[max(0, i - 2)]), float(x[min(x.size - 1, i + 2)])
    lo, hi = min(lo, truth), max(hi, truth)
    pad = 0.05 * (hi - lo) if hi > lo else 0.05 * abs(lo) + 1e-6
    return lo - pad, hi + pad


def plot_profiles(profile: Record | None, path: str | Path) -> Path:
    """One panel per profiled parameter: deviance vs value, threshold, truth, MLE and both intervals.

    Converged points are filled, unconverged points (success false) open, failed points (no deviance) are
    crosses at zero. x is zoomed to the two grid points on each side of the minimum (and the truth).
    """
    profile = _require_profile(profile)
    plt = _pyplot()
    from matplotlib.lines import Line2D

    derived = derived_numbers(profile)["params"]
    names = _param_order(list(profile["profiles"]))
    ncols = min(4, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat[len(names) :]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names, strict=False):
        entry = profile["profiles"][name]
        x = _floats(entry.get("grid"))
        d = _floats(entry.get("deviance"))
        if d.size != x.size:
            d = np.full(x.size, np.nan)
        success = _bools(entry.get("success"), x.size, True)
        failed = _bools(entry.get("failed"), x.size, False) | ~np.isfinite(d)
        ok = ~failed & success
        unconverged = ~failed & ~success
        truth = float(entry["truth"])
        ax.plot(x[~failed], d[~failed], "-", color="C0", lw=0.8, alpha=0.5)
        ax.plot(x[ok], d[ok], "o", color="C0", ms=5)
        ax.plot(x[unconverged], d[unconverged], "o", mfc="none", mec="C1", ms=7, mew=1.5)
        ax.plot(x[failed], np.zeros(int(failed.sum())), "x", color="C3", ms=7)
        ax.axhline(CHI2_1_95, ls="--", color="k", lw=0.8)
        ax.axvline(truth, color="C2", lw=1.2)
        ax.plot([_num(entry.get("mle"))], [0.0], "*", color="k", ms=10)
        iv = entry.get("interval")
        if iv is not None:
            ax.hlines(2.5, _num(iv["lo"]), _num(iv["hi"]), color="C4", lw=4)
        quad = derived[name]["quadratic_interval"]
        if quad is not None:
            ax.hlines(1.2, quad["lo"], quad["hi"], color="C5", lw=4)
        ax.set_yscale("symlog", linthresh=10.0)
        ax.set_xlim(*_zoom(x, d, truth))
        ax.set_ylim(-0.5, None)
        ax.set_title(f"{name} ({entry.get('shape')})", fontsize=9)
        ax.set_xlabel(_label(name), fontsize=8)
        ax.set_ylabel("profile deviance", fontsize=8)
        ax.grid(alpha=0.3)
    handles = [
        Line2D([], [], marker="o", ls="", color="C0", label="converged grid point"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="C1", label="unconverged (success false)"),
        Line2D([], [], marker="x", ls="", color="C3", label="failed (no deviance), drawn at 0"),
        Line2D([], [], ls="--", color="k", label=f"threshold chi2(1) 95% = {CHI2_1_95:.3f}"),
        Line2D([], [], color="C2", label="truth"),
        Line2D([], [], marker="*", ls="", color="k", label="MLE"),
        Line2D([], [], color="C4", lw=4, label=LINEAR_LABEL),
        Line2D([], [], color="C5", lw=4, label=QUADRATIC_LABEL),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"S1c profile likelihood (exact ODE likelihood), symlog y, x zoomed around the minimum\n"
        f"source: {_source(profile)}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _save(fig, path)


def plot_intervals(derived: Mapping[str, Any] | None, path: str | Path) -> Path:
    """Forest plot of interval / truth: reported and quadratic diagnostic, coloured by coverage."""
    params = (derived or {}).get("params") or {}
    if not params:
        raise FigureInputMissing("no derived profile numbers")
    plt = _pyplot()
    from matplotlib.lines import Line2D

    names = _param_order(list(params))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 2))
    for i, name in enumerate(names):
        p = params[name]
        truth = p["truth"]
        y = len(names) - 1 - i
        rows = (
            (p["reported_interval"], p["reported_covers_truth"], 0.15, "-", 4),
            (p["quadratic_interval"], p["quadratic_covers_truth"], -0.15, ":", 3),
        )
        for iv, covers, dy, ls, lw in rows:
            if iv is None or covers is None:
                continue
            color = "C2" if covers else "C3"
            ax.hlines(y + dy, _num(iv["lo"]) / truth, _num(iv["hi"]) / truth, color=color, ls=ls, lw=lw)
        ax.plot([p["mle"] / truth], [y], "k|", ms=10)
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_yticks(np.arange(len(names))[::-1], names)
    ax.set_xlabel("interval endpoint / truth [-] (1 = truth)")
    ax.grid(axis="x", alpha=0.3)
    handles = [
        Line2D([], [], color="C2", lw=4, label=f"{LINEAR_LABEL}, covers truth"),
        Line2D([], [], color="C3", lw=4, label=f"{LINEAR_LABEL}, misses truth"),
        Line2D([], [], color="C2", lw=3, ls=":", label=f"{QUADRATIC_LABEL}, covers truth"),
        Line2D([], [], color="C3", lw=3, ls=":", label=f"{QUADRATIC_LABEL}, misses truth"),
        Line2D([], [], marker="|", ls="", color="k", ms=10, label="MLE / truth"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")
    label = config_label(derived)
    fig.suptitle(f"S1c 95% profile intervals relative to truth\nsource: {derived.get('run_id')} ({label})")
    return _save(fig, path)


def plot_gn_spectrum(profile: Record | None, path: str | Path) -> Path:
    """Relative Gauss-Newton eigenvalues (log scale) and the loadings of the weakest direction."""
    spectrum = (profile or {}).get("spectrum") or {}
    relative = _positive(spectrum.get("relative"))
    if relative.size == 0 or profile is None:
        raise FigureInputMissing("no profile record with a Gauss-Newton spectrum")
    plt = _pyplot()
    loadings = spectrum.get("weakest_loadings") or {}
    names = _param_order(list(loadings))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.semilogy(np.arange(1, relative.size + 1), relative, "o-")
    ax1.set_xlabel("eigenvalue rank (1 = largest)")
    ax1.set_ylabel("eigenvalue / largest eigenvalue [-]")
    smallest = _num(spectrum.get("smallest_relative"))
    ax1.set_title(f"J^T J spectrum at the MLE, smallest relative {smallest:.2e}")
    ax1.grid(alpha=0.3, which="both")
    values = [_num(loadings[n]) for n in names]
    ax2.barh(np.arange(len(names))[::-1], values, color=["C3" if v < 0 else "C0" for v in values])
    ax2.set_yticks(np.arange(len(names))[::-1], names)
    ax2.axvline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("loading in the weakest eigenvector [-]")
    ax2.set_title("weakest direction (smallest eigenvalue)")
    ax2.grid(axis="x", alpha=0.3)
    fig.suptitle(f"S1c Gauss-Newton spectrum\nsource: {_source(profile)}", fontsize=10)
    return _save(fig, path)


def plot_misspec_smoke(records: Records | None, path: str | Path) -> Path:
    """Per misspecification variant: M2 relative errors of rho_c and k_c, and M1 / M2 residual ratios."""
    by_key: dict[tuple[str, str], Record] = {}
    for record in sorted((records or {}).values(), key=lambda r: str(r.get("run_id"))):
        spec = record.get("spec") or {}
        if usable(record) is None or spec.get("part") != "misspec" or spec.get("kind") != "fit":
            continue
        by_key.setdefault((str(spec.get("variant")), str(spec.get("model"))), record)
    if not by_key:
        raise FigureInputMissing("no finished misspecification fit records")
    plt = _pyplot()
    order = [v for v in MISSPEC_STUDY_VARIANTS if any(k[0] == v for k in by_key)]
    order += sorted({k[0] for k in by_key} - set(order))
    x = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    m2_errors = {v: (by_key.get((v, "M2")) or {}).get("rel_error") or {} for v in order}
    gated = _param_order([n for errs in m2_errors.values() for n in errs if group_of(n) in ("rho", "k")])
    if gated:
        width = 0.8 / len(gated)
        for j, name in enumerate(gated):
            vals = [100.0 * _num(m2_errors[v].get(name)) for v in order]
            ax1.bar(x - 0.4 + width * (j + 0.5), vals, width, label=name, color=f"C{j}")
        ax1.set_yscale("symlog", linthresh=10.0)
        ax1.legend(fontsize=7, ncol=2)
    else:
        ax1.text(0.5, 0.5, "no M2 misspecification records", ha="center", transform=ax1.transAxes)
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_ylabel("M2 relative error vs variant truth [%], symlog")
    ax1.set_title("M2 recovery of rho_c and k_c")
    for j, model in enumerate(("M1", "M2")):
        diags = [(by_key.get((v, model)) or {}).get("diagnostics") or {} for v in order]
        vals = [_num(d.get("residual_ratio")) for d in diags]
        ax2.bar(x - 0.2 + 0.4 * j, vals, 0.4, label=model, color=f"C{j}")
    ax2.axhline(1.2, color="k", ls="--", lw=0.8, label="criterion 3 flag at 1.2")
    ax2.set_yscale("log")
    ax2.set_ylabel("residual ratio = scored residual RMS / noise_s [-]")
    ax2.set_title("residual ratio per model")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.set_xticks(x, order, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
    first = next(iter(by_key.values()))
    configs = sorted({str(r.get("config_name")) for r in by_key.values()})
    run_ids = sorted(str(r.get("run_id")) for r in by_key.values())
    suffix = "smoke" if configs == ["smoke"] else "/".join(configs)
    fig.suptitle(
        f"S1d misspecification fits ({suffix} config; smoke numbers do not answer S1)\n"
        f"source: {len(run_ids)} records like {first.get('run_id')} ({config_label(first)})",
        fontsize=10,
    )
    return _save(fig, path)


# --- orchestration ---


def _read_probe(root: Path) -> Record:
    path = root / PROBE_PATH
    if not path.is_file():
        raise FigureInputMissing(f"{PROBE_PATH} not found")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _synthetic_dataset() -> Any:
    import jax

    from apexpinn.physics.synthetic import make_synthetic_dataset

    return make_synthetic_dataset(jax.random.key(0))


def _fit_summary(record: Record | None) -> dict[str, Any] | None:
    if record is None:
        return None
    convergence = record.get("convergence") or {}
    return {
        "run_id": record.get("run_id"),
        "config_name": record.get("config_name"),
        "config_hash": record.get("config_hash"),
        "residual_ratio": (record.get("diagnostics") or {}).get("residual_ratio"),
        "rel_error": record.get("rel_error"),
        "recovered": record.get("recovered"),
        "objective_rel_change_last20": convergence.get("objective_rel_change_last20"),
    }


def _write_derived(profile: Record | None, m1: Record | None, m2: Record | None, path: Path) -> Path:
    profile = _require_profile(profile)
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "profile": derived_numbers(profile),
        "main_fits": {"M1": _fit_summary(m1), "M2": _fit_summary(m2)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def build_figures(
    root: str | Path,
    out_dir: str | Path = "reports/figures",
    *,
    only: Sequence[str] | None = None,
    log: Log = print,
    dataset_fn: Callable[[], Any] = _synthetic_dataset,
) -> dict[str, str]:
    """Write the requested artifacts (default all of ARTIFACT_NAMES) and return name -> outcome.

    The outcome is 'written: <path>', 'skipped: <reason>' when the figure's input is missing
    (FigureInputMissing), or 'failed: <type>: <message>' when the build raised anything else. Every
    outcome is logged and the run never stops on one figure. M1 and M2 come from one source
    (pick_fit_pair); the profile record is picked on its own, and every figure names the config of each
    record it shows. Relative out_dir is resolved against root. dataset_fn builds the synthetic dataset
    (only called for synthetic_stints).
    """
    root = Path(root).resolve()
    out = Path(out_dir)
    out = out if out.is_absolute() else root / out
    wanted = list(ARTIFACT_NAMES if not only else only)
    unknown = [n for n in wanted if n not in ARTIFACT_NAMES]
    if unknown:
        raise ValueError(f"unknown figure names {unknown}, expected names from {ARTIFACT_NAMES}")
    sources: list[tuple[str, Records]] = [
        (FULL_DIR, load_records(root / FULL_DIR)),
        (SMOKE_DIR, load_records(root / SMOKE_DIR)),
    ]

    def record(**spec: Any) -> Record | None:
        found = pick_record(sources, **spec)
        return None if found is None else found[1]

    pair = pick_fit_pair(sources)
    m1, m2 = (None, None) if pair is None else pair[1:]
    profile = record(kind="profile")

    def tolerance() -> dict[str, float]:
        path = root / DECISIONS_PATH
        if not path.is_file():
            raise FigureInputMissing(f"{DECISIONS_PATH} not found")
        return o6_tolerance(path)

    builders: dict[str, Callable[[Path], Path]] = {
        "probe_warmup": lambda p: plot_probe_warmup(_read_probe(root), p),
        "probe_pace_pit": lambda p: plot_probe_pace_pit(_read_probe(root), p),
        "synthetic_stints": lambda p: plot_synthetic_stints(dataset_fn(), p),
        "s1_fit_convergence": lambda p: plot_fit_convergence(m1, m2, p),
        "s1_recovery": lambda p: plot_recovery(m1, m2, profile, tolerance(), p),
        "s1_profiles": lambda p: plot_profiles(profile, p),
        "s1_intervals": lambda p: plot_intervals(derived_numbers(_require_profile(profile)), p),
        "s1_gn_spectrum": lambda p: plot_gn_spectrum(profile, p),
        "s1_misspec_smoke": lambda p: plot_misspec_smoke(dict(sources)[SMOKE_DIR], p),
        "derived_numbers": lambda p: _write_derived(profile, m1, m2, p),
    }
    results: dict[str, str] = {}
    for name in wanted:
        suffix = ".json" if name == "derived_numbers" else ".png"
        try:
            written = builders[name](out / f"{name}{suffix}")
        except FigureInputMissing as exc:
            results[name] = f"skipped: {exc}"
        except Exception as exc:  # one broken figure must not stop the others, but it fails the run
            results[name] = f"failed: {type(exc).__name__}: {exc}"
        else:
            results[name] = f"written: {written}"
        log(f"{name}: {results[name]}")
    return results
