"""Tests for the research-record figures (apexpinn.eval.figures). Fake records only: the tests never read
experiments/. Plot tests need matplotlib and are skipped without it."""

import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import scipy.stats

from apexpinn.eval.figures import (
    ARTIFACT_NAMES,
    CHI2_1_95,
    FIGURE_NAMES,
    FigureInputMissing,
    build_figures,
    config_label,
    coord_to_natural,
    derived_numbers,
    deviance_at_truth,
    linear_interval,
    o6_tolerance,
    pick_fit_pair,
    pick_record,
    plot_fit_convergence,
    plot_gn_spectrum,
    plot_intervals,
    plot_misspec_smoke,
    plot_probe_pace_pit,
    plot_probe_warmup,
    plot_profiles,
    plot_recovery,
    plot_synthetic_stints,
    quadratic_interval,
)
from apexpinn.physics.identifiability import likelihood_interval

THRESHOLD = 3.841458820694124
GRID = np.linspace(-1.0, 1.0, 9)  # spacing 0.25, index 4 is 0.0
REPO = Path(__file__).resolve().parents[1]
MAKE_FIGURES = REPO / "scripts" / "make_figures.py"


@pytest.fixture
def mpl():
    return pytest.importorskip("matplotlib")


def _parabola(a: float, x0: float) -> np.ndarray:
    return a * (GRID - x0) ** 2


def _is_png(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- pure helpers ---


def test_threshold_is_chi2_1_95():
    assert CHI2_1_95 == pytest.approx(THRESHOLD, abs=1e-12)


def test_quadratic_interval_recovers_exact_parabola():
    a, x0 = 200.0, 0.02
    lo, hi = quadratic_interval(GRID, _parabola(a, x0))
    half = math.sqrt(THRESHOLD / a)
    assert lo == pytest.approx(x0 - half, abs=1e-9)
    assert hi == pytest.approx(x0 + half, abs=1e-9)


def test_linear_interval_is_narrower_when_neighbours_exceed_threshold():
    # Both neighbours of the minimum are above 3.841, as in the full-config profile: the linear
    # interpolation of a convex profile crosses the threshold closer to the minimum.
    dev = _parabola(200.0, 0.02)
    assert dev[3] > THRESHOLD and dev[5] > THRESHOLD
    iv = linear_interval(GRID, dev)
    assert iv == likelihood_interval(GRID, dev)  # the reported method, unchanged
    assert iv.lo_bounded and iv.hi_bounded
    lo, hi = quadratic_interval(GRID, dev)
    assert lo < iv.lo and iv.hi < hi


def test_quadratic_interval_none_at_grid_edge_or_nonfinite_neighbour():
    assert quadratic_interval(GRID, _parabola(200.0, -1.0)) is None
    assert quadratic_interval(GRID, _parabola(200.0, 1.0)) is None
    dev = list(_parabola(200.0, 0.02))
    dev[5] = None  # JSON form of a failed point
    assert quadratic_interval(GRID, dev) is None
    dev = _parabola(200.0, 0.02)
    dev[3] = np.nan
    assert quadratic_interval(GRID, dev) is None


def test_deviance_at_truth_hand_computed():
    assert deviance_at_truth(2.0 * math.e, 2.0, 10) == pytest.approx(10.0)
    assert deviance_at_truth(1.5, 1.5, 7) == 0.0
    with pytest.raises(ValueError):
        deviance_at_truth(0.0, 1.0, 10)


def test_coord_to_natural_matches_record_transform():
    assert coord_to_natural("h_opt", math.log(0.6 / 0.4)) == pytest.approx(0.6)
    assert coord_to_natural("k", math.log(0.3)) == pytest.approx(0.3)
    assert coord_to_natural("kappa", 0.0) == pytest.approx(1.0)


# --- fake records ---


def _entry(group, grid, coord, deviance, truth, mle, success=None, failed=None):
    """Profile entry as s1_study._profile_entry writes it (interval in natural units)."""
    iv = likelihood_interval(coord, [np.nan if d is None else d for d in deviance])
    lo, hi = coord_to_natural(group, iv.lo), coord_to_natural(group, iv.hi)
    n = len(grid)
    return {
        "group": group,
        "grid": [float(g) for g in grid],
        "grid_coord": [float(c) for c in coord],
        "deviance": [None if d is None else float(d) for d in deviance],
        "success": success or [True] * n,
        "failed": failed or [False] * n,
        "interval": {
            "lo": lo,
            "hi": hi,
            "lo_bounded": iv.lo_bounded,
            "hi_bounded": iv.hi_bounded,
            "lo_failed": iv.lo_failed,
            "hi_failed": iv.hi_failed,
            "threshold": iv.threshold,
        },
        "mle": mle,
        "truth": truth,
        "shape": "sharp",
        "covers_truth": (not iv.lo_bounded or lo <= truth) and (not iv.hi_bounded or truth <= hi),
        "sse_min": 1.0,
    }


def _profile_record(config="full"):
    k_grid = 0.29 * np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    k_coord = np.log(k_grid)
    k_dev = 1000.0 * (k_coord - math.log(0.29)) ** 2
    h_grid = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
    h_coord = np.log(h_grid) - np.log1p(-h_grid)
    return {
        "run_id": "main_profile_ode_well_specified_noise0.05_seed0",
        "config_name": config,
        "config_hash": "hash-profile",
        "status": "ok",
        "spec": {
            "kind": "profile",
            "model": None,
            "part": "main",
            "variant": "well_specified",
            "noise_s": 0.05,
            "seed": 0,
        },
        "n_obs": 1000,
        "sse_truth": math.exp(0.003),
        "mle": {"sse": 1.0, "success": True, "values": {"k_c[C2]": 0.29, "h_opt_c[C2]": 0.5}},
        "truth": {"k_c[C2]": 0.276, "h_opt_c[C2]": 0.52},
        "profiles": {
            "k_c[C2]": _entry("k", k_grid, k_coord, list(k_dev), truth=0.276, mle=0.29),
            "h_opt_c[C2]": _entry(
                "h_opt",
                h_grid,
                h_coord,
                [0.0, 2.0, 5.0, 9.0, None],
                truth=0.52,
                mle=0.5,
                success=[True, False, True, True, False],
                failed=[False, False, False, False, True],
            ),
        },
        "spectrum": {
            "eigenvalues": [10.0, 0.1],
            "relative": [1.0, 0.01],
            "smallest_relative": 0.01,
            "weakest_loadings": {"k_c[C2]": 0.8, "h_opt_c[C2]": -0.6},
        },
    }


def _fit_record(model, config="smoke", part="main", variant="well_specified", ratio=210.0):
    normalized = {"data": [1.0, 0.4, 0.36]}
    if model == "M1":
        normalized |= {"thermal": [1.0, 0.1, 0.06], "wear": [0.8, 0.1, 0.03], "mono": [0.0, 0.0, 0.0]}
        normalized["reg"] = [1.0, 0.9, 1.1]
    spec = {"kind": "fit", "model": model, "part": part, "variant": variant, "noise_s": 0.05, "seed": 0}
    return {
        "run_id": f"{part}_fit_{model}_{variant}_noise0.05_seed0",
        "config_name": config,
        "config_hash": f"hash-{model}-{config}",
        "status": "ok",
        "spec": spec,
        "convergence": {
            "steps": [0, 50, 100],
            "objective": [1.0, 0.5, 0.45],
            "normalized": normalized,
            "objective_rel_change_last20": -0.1,
        },
        "diagnostics": {"residual_ratio": ratio, "collapse_flags": []},
        "rel_error": {"k_c[C2]": -0.8, "rho_c[C2]": 3.09, "kappa": None},
        "recovered": {"k_c[C2]": False, "rho_c[C2]": False},
        "truth": {"k_c[C2]": 0.28, "rho_c[C2]": 0.045, "kappa": 1.0},
    }


def _probe():
    def race(event_id, role, gain):
        return {
            "event_id": event_id,
            "role": role,
            "warmup_visibility": {"dip_share": 0.3, "dip_gain_s": gain},
            "raw_stint_slope_s_per_lap": {"median": 0.013},
            "pit_lane_time_s": {"n": 40, "median": 22.6, "q25": 22.2, "q75": 23.4},
        }

    gain = {"n": 18, "median": 0.31, "q25": 0.18, "q75": 0.52}
    races = [race("2023_barcelona", "dev", gain), race("2024_bahrain", "dev", None)]
    return {"preset": "extended", "races": [*races, race("2025_suzuka", "holdout", gain)]}


def _dataset():
    n_laps = np.array([5, 6, 4])
    tau = np.arange(1, 8, dtype=float)
    mask = tau[None, :] <= n_laps[:, None]
    y_clean = np.where(mask, 90.0 + 0.05 * tau[None, :], 0.0)
    return SimpleNamespace(
        compound_idx=np.array([1, 2, 1]),
        n_laps=n_laps,
        mask=mask,
        tau=tau,
        y=y_clean + np.where(mask, 0.05, 0.0),
        y_clean=y_clean,
        b=np.array([88.0, 89.0, 90.0]),
        noise_s=0.05,
    )


# --- derived numbers and record selection ---


def test_derived_numbers_keys_and_coverage():
    out = derived_numbers(_profile_record())
    assert out["run_id"] == "main_profile_ode_well_specified_noise0.05_seed0"
    assert out["config_hash"] == "hash-profile"
    assert out["deviance_at_truth"] == pytest.approx(3.0)
    assert out["n_params"] == 2
    assert out["chi2_95_n_params"] == pytest.approx(float(scipy.stats.chi2.ppf(0.95, 2)))
    assert out["truth_inside_joint_95_region"] is True
    k = out["params"]["k_c[C2]"]
    expected_keys = {
        "group",
        "truth",
        "mle",
        "shape",
        "reported_interval",
        "reported_covers_truth",
        "quadratic_interval",
        "quadratic_covers_truth",
        "min_index",
        "deviance_step_lower",
        "deviance_step_upper",
    }
    assert set(k) == expected_keys
    # The linear interval misses the truth 0.276; the quadratic diagnostic of the exact parabola covers it.
    assert k["reported_covers_truth"] is False
    assert k["quadratic_covers_truth"] is True
    half = math.sqrt(THRESHOLD / 1000.0)
    assert k["quadratic_interval"]["lo"] == pytest.approx(0.29 * math.exp(-half), rel=1e-9)
    assert k["quadratic_interval"]["hi"] == pytest.approx(0.29 * math.exp(half), rel=1e-9)
    assert k["min_index"] == 2
    assert k["deviance_step_lower"] == pytest.approx(1000.0 * math.log(0.9) ** 2)
    assert k["deviance_step_upper"] == pytest.approx(1000.0 * math.log(1.1) ** 2)
    h = out["params"]["h_opt_c[C2]"]
    assert h["reported_covers_truth"] is True
    assert h["quadratic_interval"] is None and h["quadratic_covers_truth"] is None
    assert h["min_index"] == 0 and h["deviance_step_lower"] is None
    assert h["deviance_step_upper"] == pytest.approx(2.0)


def test_pick_record_prefers_full_config_and_falls_back_to_smoke():
    full = _fit_record("M1", config="full")
    smoke = _fit_record("M1", config="smoke")
    m2 = _fit_record("M2", config="smoke")
    sources = [("experiments/s1", {full["run_id"]: full}), ("experiments/s1_smoke", {"a": smoke, "b": m2})]
    assert pick_record(sources, kind="fit", model="M1") == ("experiments/s1", full)
    assert pick_record(sources, kind="fit", model="M2") == ("experiments/s1_smoke", m2)
    sources[0] = ("experiments/s1", {})
    assert pick_record(sources, kind="fit", model="M1") == ("experiments/s1_smoke", smoke)
    failed = dict(smoke, status="error")
    assert pick_record([("x", {"a": failed})], kind="fit", model="M1") is None
    assert "smoke" in config_label(smoke)
    assert "smoke" not in config_label(full)


def test_pick_record_ranks_lowest_noise_then_seed():
    def fit(noise, seed):
        r = _fit_record("M1", config="full")
        return dict(r, spec=dict(r["spec"], noise_s=noise, seed=seed), run_id=f"fit_{noise}_{seed}")

    records = {r["run_id"]: r for r in (fit(0.15, 0), fit(0.05, 2), fit(0.15, 1), fit(0.05, 1))}
    label, picked = pick_record([("experiments/s1", records)], kind="fit", model="M1")
    assert label == "experiments/s1"
    assert picked["run_id"] == "fit_0.05_1"


def test_pick_fit_pair_takes_both_models_from_one_source():
    full_m1 = _fit_record("M1", config="full")
    smoke_m1, smoke_m2 = _fit_record("M1"), _fit_record("M2")
    smoke = {r["run_id"]: r for r in (smoke_m1, smoke_m2)}
    # Only the full M1 fit exists in experiments/s1: M2 is left out, not filled in from smoke.
    sources = [("experiments/s1", {full_m1["run_id"]: full_m1}), ("experiments/s1_smoke", smoke)]
    assert pick_fit_pair(sources) == ("experiments/s1", full_m1, None)
    sources[0] = ("experiments/s1", {})
    assert pick_fit_pair(sources) == ("experiments/s1_smoke", smoke_m1, smoke_m2)
    assert pick_fit_pair([("experiments/s1", {})]) is None


def _decisions_fixture(path: Path) -> Path:
    """Copy of the decisions file with O6 tolerances rho_c 0.11 and k_c 0.22, independent of the draft."""
    text = (REPO / "configs" / "decisions_v1.yaml").read_text(encoding="utf-8")
    text, n_rho = re.subn(r"(\n\s+rho_c:) [0-9.]+", r"\1 0.11", text)
    text, n_k = re.subn(r"(\n\s+k_c:) [0-9.]+", r"\1 0.22", text)
    assert (n_rho, n_k) == (1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_o6_tolerance_from_decisions(tmp_path):
    assert o6_tolerance(_decisions_fixture(tmp_path / "decisions.yaml")) == {"rho": 0.11, "k": 0.22}


# --- plots ---


def test_probe_plots_write_png(mpl, tmp_path):
    assert _is_png(plot_probe_warmup(_probe(), tmp_path / "probe_warmup.png"))
    assert _is_png(plot_probe_pace_pit(_probe(), tmp_path / "probe_pace_pit.png"))
    with pytest.raises(FigureInputMissing):
        plot_probe_warmup({"races": [{"event_id": "x", "role": "holdout"}]}, tmp_path / "none.png")


def test_synthetic_stints_writes_png(mpl, tmp_path):
    assert _is_png(plot_synthetic_stints(_dataset(), tmp_path / "synthetic_stints.png", per_compound=2))


def test_fit_and_recovery_plots_write_png(mpl, tmp_path):
    m1, m2 = _fit_record("M1"), _fit_record("M2", ratio=2.6)
    assert _is_png(plot_fit_convergence(m1, m2, tmp_path / "s1_fit_convergence.png"))
    tolerance = {"rho": 0.15, "k": 0.25}
    path = plot_recovery(m1, m2, _profile_record(), tolerance, tmp_path / "s1_recovery.png")
    assert _is_png(path)
    with pytest.raises(FigureInputMissing):
        plot_fit_convergence(None, m2, tmp_path / "none.png")


def test_profile_plots_write_png(mpl, tmp_path):
    profile = _profile_record()
    assert _is_png(plot_profiles(profile, tmp_path / "s1_profiles.png"))
    assert _is_png(plot_intervals(derived_numbers(profile), tmp_path / "s1_intervals.png"))
    assert _is_png(plot_gn_spectrum(profile, tmp_path / "s1_gn_spectrum.png"))
    with pytest.raises(FigureInputMissing):
        plot_profiles(None, tmp_path / "none.png")


def test_misspec_plot_writes_png(mpl, tmp_path):
    records = {}
    for variant in ("wrong_phi", "traffic"):
        for model in ("M1", "M2"):
            r = _fit_record(model, part="misspec", variant=variant, ratio=200.0 if model == "M1" else 1.5)
            records[r["run_id"]] = r
    assert _is_png(plot_misspec_smoke(records, tmp_path / "s1_misspec_smoke.png"))
    with pytest.raises(FigureInputMissing):
        plot_misspec_smoke({}, tmp_path / "none.png")


def test_build_figures_skips_missing_inputs(tmp_path):
    lines = []
    names = ["probe_warmup", "s1_profiles", "derived_numbers"]
    results = build_figures(tmp_path, only=names, log=lines.append)
    assert set(results) == set(names)
    assert all(r.startswith("skipped") for r in results.values())
    assert len(lines) == 3
    assert not (tmp_path / "reports" / "figures" / "probe_warmup.png").exists()


def _write_records(root: Path, folder: str, records) -> None:
    out = root / "experiments" / folder
    out.mkdir(parents=True, exist_ok=True)
    for r in records:
        (out / f"{r['run_id']}.json").write_text(json.dumps(r), encoding="utf-8")


def test_build_figures_writes_every_artifact_from_fake_records(mpl, tmp_path):
    misspec = [
        _fit_record(model, part="misspec", variant=variant, ratio=200.0 if model == "M1" else 1.5)
        for variant in ("wrong_phi", "traffic")
        for model in ("M1", "M2")
    ]
    _write_records(tmp_path, "s1", [_profile_record()])
    _write_records(tmp_path, "s1_smoke", [_fit_record("M1"), _fit_record("M2", ratio=2.6), *misspec])
    probe = tmp_path / "experiments" / "data_probe" / "probe_results.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(json.dumps(_probe()), encoding="utf-8")
    _decisions_fixture(tmp_path / "configs" / "decisions_v1.yaml")
    results = build_figures(tmp_path, log=lambda _: None, dataset_fn=_dataset)
    assert set(results) == set(ARTIFACT_NAMES)
    assert all(r.startswith("written") for r in results.values()), results
    figures = tmp_path / "reports" / "figures"
    for name in FIGURE_NAMES:
        assert _is_png(figures / f"{name}.png"), name
    derived = json.loads((figures / "derived_numbers.json").read_text(encoding="utf-8"))
    assert {"generated_utc", "profile", "main_fits"} <= set(derived)
    assert derived["profile"]["config_hash"] == "hash-profile"
    assert derived["main_fits"]["M1"]["run_id"] == "main_fit_M1_well_specified_noise0.05_seed0"
    assert derived["main_fits"]["M1"]["config_name"] == "smoke"
    assert derived["main_fits"]["M2"]["config_name"] == "smoke"


def test_build_figures_never_pairs_full_and_smoke_fits(tmp_path):
    # State right after the first full-config M1 fit: experiments/s1 has the profile and M1 only.
    _write_records(tmp_path, "s1", [_profile_record(), _fit_record("M1", config="full")])
    _write_records(tmp_path, "s1_smoke", [_fit_record("M1"), _fit_record("M2", ratio=2.6)])
    results = build_figures(tmp_path, only=["derived_numbers"], log=lambda _: None)
    assert results["derived_numbers"].startswith("written"), results
    path = tmp_path / "reports" / "figures" / "derived_numbers.json"
    fits = json.loads(path.read_text(encoding="utf-8"))["main_fits"]
    assert fits["M1"]["config_name"] == "full"
    assert fits["M2"] is None


def test_build_figures_reports_failed_builds(tmp_path):
    def broken():
        raise RuntimeError("boom")

    results = build_figures(tmp_path, only=["synthetic_stints"], log=lambda _: None, dataset_fn=broken)
    assert results == {"synthetic_stints": "failed: RuntimeError: boom"}


def _load_make_figures():
    """Load the script, then restore x64: its module-level update must not leak into later tests."""
    previous = jax.config.x64_enabled
    try:
        spec = importlib.util.spec_from_file_location("make_figures", MAKE_FIGURES)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        jax.config.update("jax_enable_x64", previous)
    return module


@pytest.mark.parametrize(
    ("outcomes", "code"),
    [
        ({"a": "written: a.png", "b": "skipped: no input"}, 0),
        ({"a": "written: a.png", "b": "failed: RuntimeError: boom"}, 1),
        ({"a": "skipped: no input"}, 1),
    ],
)
def test_make_figures_exit_code(monkeypatch, outcomes, code):
    script = _load_make_figures()
    monkeypatch.setattr(script, "build_figures", lambda *args, **kwargs: outcomes)
    assert script.main([]) == code


def test_loading_make_figures_restores_x64():
    previous = jax.config.x64_enabled
    _load_make_figures()
    assert jax.config.x64_enabled == previous


def test_figure_names_cover_the_nine_figures():
    assert len(FIGURE_NAMES) == 9
    assert {
        "probe_warmup",
        "probe_pace_pit",
        "synthetic_stints",
        "s1_fit_convergence",
        "s1_recovery",
        "s1_profiles",
        "s1_intervals",
        "s1_gn_spectrum",
        "s1_misspec_smoke",
    } == set(FIGURE_NAMES)
