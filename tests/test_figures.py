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
    interval_legend_labels,
    interval_methods,
    linear_interval,
    linear_interval_v1,
    o6_tolerance,
    pick_fit_pair,
    pick_fit_source,
    pick_record,
    plot_fit_convergence,
    plot_gn_spectrum,
    plot_intervals,
    plot_misspec_smoke,
    plot_probe_pace_pit,
    plot_probe_warmup,
    plot_profiles,
    plot_recovery,
    plot_recovery_seeds,
    plot_synthetic_stints,
    quadratic_interval,
    reported_interval,
    seed_recovery_table,
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
    assert iv == likelihood_interval(GRID, dev)  # interval method v1, unchanged
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
        "interval_method",
        "reported_interval",
        "reported_covers_truth",
        "linear_interval_v1",
        "linear_v1_covers_truth",
        "linear_v1_status",
        "quadratic_interval",
        "quadratic_covers_truth",
        "min_index",
        "deviance_step_lower",
        "deviance_step_upper",
    }
    assert set(k) == expected_keys
    # A record from before profile method version 3: the reported interval is the v1 linear one.
    assert out["interval_methods"] == ["v1_linear"]
    assert k["interval_method"] == "v1_linear" and k["linear_v1_status"] == "reported"
    assert k["linear_interval_v1"] == k["reported_interval"]
    assert k["linear_v1_covers_truth"] is False
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


def test_figure_names_cover_the_ten_figures():
    assert len(FIGURE_NAMES) == 10
    assert {
        "probe_warmup",
        "probe_pace_pit",
        "synthetic_stints",
        "s1_fit_convergence",
        "s1_recovery",
        "s1_recovery_seeds",
        "s1_profiles",
        "s1_intervals",
        "s1_gn_spectrum",
        "s1_misspec_smoke",
    } == set(FIGURE_NAMES)
    assert FIGURE_NAMES.index("s1_recovery_seeds") == FIGURE_NAMES.index("s1_recovery") + 1


# --- interval method (PLAN T5 'Interval method v2') ---

K_HALF = math.sqrt(THRESHOLD / 1000.0)  # exact half-width of the fake k profile in log coordinates


def _v2_profile_record(config="full", store_linear=True):
    """_profile_record with the k_c[C2] entry as profile method version 3 writes it: the v2 interval is
    the exact crossing (covers the truth 0.276), the v1 linear interval misses it."""
    record = _profile_record(config)
    entry = dict(record["profiles"]["k_c[C2]"])
    linear = dict(entry["interval"])
    entry["interval"] = dict(linear, lo=0.29 * math.exp(-K_HALF), hi=0.29 * math.exp(K_HALF))
    entry["interval_method"] = "v2_refined"
    entry["covers_truth"] = True
    if store_linear:
        entry["interval_linear"] = linear
    record["profiles"] = {**record["profiles"], "k_c[C2]": entry}
    return record, linear


def test_reported_interval_follows_interval_method():
    v1 = _profile_record()["profiles"]["k_c[C2]"]
    iv, method = reported_interval(v1)
    assert method == "v1_linear" and iv is v1["interval"]
    record, linear = _v2_profile_record()
    v2 = record["profiles"]["k_c[C2]"]
    iv, method = reported_interval(v2)
    assert method == "v2_refined"
    assert iv["lo"] == pytest.approx(0.29 * math.exp(-K_HALF)) and iv["lo"] != pytest.approx(linear["lo"])
    assert linear_interval_v1(v2) == linear
    assert linear_interval_v1(v1) == v1["interval"]
    # A v2 entry without interval_linear gets the v1 interval recomputed from its deviance.
    record, linear = _v2_profile_record(store_linear=False)
    recomputed = linear_interval_v1(record["profiles"]["k_c[C2]"])
    assert recomputed["lo"] == pytest.approx(linear["lo"], rel=1e-12)
    assert recomputed["hi"] == pytest.approx(linear["hi"], rel=1e-12)
    assert interval_methods(record["profiles"].values()) == ["v2_refined", "v1_linear"]


def test_derived_numbers_reports_v2_and_keeps_v1_as_superseded():
    record, linear = _v2_profile_record()
    out = derived_numbers(record)
    assert out["interval_methods"] == ["v1_linear", "v2_refined"]
    k = out["params"]["k_c[C2]"]
    assert k["interval_method"] == "v2_refined"
    assert k["reported_interval"]["lo"] == pytest.approx(0.29 * math.exp(-K_HALF))
    assert k["reported_interval"]["hi"] == pytest.approx(0.29 * math.exp(K_HALF))
    assert k["reported_covers_truth"] is True
    assert k["linear_interval_v1"] == {key: linear[key] for key in ("lo", "hi", "lo_bounded", "hi_bounded")}
    assert k["linear_v1_covers_truth"] is False
    assert k["linear_v1_status"] == "Superseded"
    # The quadratic diagnostic is still there and unchanged.
    assert k["quadratic_interval"]["lo"] == pytest.approx(0.29 * math.exp(-K_HALF), rel=1e-9)
    assert out["params"]["h_opt_c[C2]"]["interval_method"] == "v1_linear"


def test_interval_legend_labels_name_the_method():
    assert interval_legend_labels(["v1_linear"]) == ["reported 95% interval (linear interpolation)"]
    assert interval_legend_labels(["v2_refined"]) == [
        "reported 95% interval (v2 refined)",
        "v1 linear (Superseded)",
    ]


def test_profile_plots_write_png_for_v2_records(mpl, tmp_path):
    record, _ = _v2_profile_record()
    assert _is_png(plot_profiles(record, tmp_path / "s1_profiles.png"))
    assert _is_png(plot_intervals(derived_numbers(record), tmp_path / "s1_intervals.png"))


# --- all-seeds fit figure ---

TOL = {"rho": 0.15, "k": 0.25}
SEED_NOISES = (0.05, 0.15)
SEED_SEEDS = (0, 1)
SEED_MODEL_TAGS = ("M1", "M2", "M2v2")


def _k_err(noise, seed):
    return 0.1 * (1 + 3 * seed) + noise / 10  # seed 0 inside the 25% k tolerance, seed 1 outside


def _rho_err(noise, seed):
    return -0.05 * (1 + 4 * seed) - noise / 10  # seed 0 inside the 15% rho tolerance, seed 1 outside


def _ratio(model, noise, seed):
    return 1.0 + seed + noise + SEED_MODEL_TAGS.index(model)


def _seed_fit(model, noise, seed, config="full", status="ok"):
    r = _fit_record(model, config=config)
    return dict(
        r,
        run_id=f"main_fit_{model}_well_specified_noise{noise:g}_seed{seed}",
        status=status,
        spec=dict(r["spec"], noise_s=noise, seed=seed),
        diagnostics={"residual_ratio": _ratio(model, noise, seed), "collapse_flags": []},
        rel_error={
            "k_c[C2]": _k_err(noise, seed),
            "rho_c[C2]": _rho_err(noise, seed),
            "gamma_c[C2]": 0.5,
            "kappa": None,
        },
    )


def _seed_records(models=SEED_MODEL_TAGS, config="full"):
    fits = [_seed_fit(m, n, s, config) for m in models for n in SEED_NOISES for s in SEED_SEEDS]
    return {r["run_id"]: r for r in fits}


def test_seed_recovery_table_has_every_seed_and_noise_level():
    rows = seed_recovery_table(_seed_records(), TOL)
    assert len(rows) == 3 * 2 * 2 * 2  # models x noise levels x seeds x gated parameters
    assert {(r["noise_s"], r["seed"]) for r in rows} == {(n, s) for n in SEED_NOISES for s in SEED_SEEDS}
    assert {r["param"] for r in rows} == {"k_c[C2]", "rho_c[C2]"}  # no tolerance for gamma, kappa
    assert [r["model"] for r in rows[:8]] == ["M1"] * 8
    assert (rows[0]["noise_s"], rows[0]["seed"], rows[0]["param"]) == (0.05, 0, "rho_c[C2]")
    for r in rows:
        noise, seed = r["noise_s"], r["seed"]
        err = _k_err(noise, seed) if r["param"] == "k_c[C2]" else _rho_err(noise, seed)
        assert r["rel_error"] == err
        assert r["residual_ratio"] == _ratio(r["model"], noise, seed)
        # Computed with the O6 tolerance, not copied: the fake records store recovered False everywhere.
        assert r["recovered"] is (seed == 0)
        assert r["config_name"] == "full"


def test_seed_recovery_table_omits_missing_models_and_unfinished_records():
    records = _seed_records(models=("M1", "M2"))
    errored = _seed_fit("M2v2", 0.05, 0, status="error")
    misspec = _fit_record("M2", config="full", part="misspec", variant="traffic")
    profile = _profile_record()
    records |= {r["run_id"]: r for r in (errored, misspec, profile)}
    rows = seed_recovery_table(records, TOL)
    assert {r["model"] for r in rows} == {"M1", "M2"}
    assert len(rows) == 2 * 2 * 2 * 2
    nan_fit = _seed_fit("M1", 0.05, 2)
    nan_fit["rel_error"] = {"k_c[C2]": None, "rho_c[C2]": float("nan")}
    rows = seed_recovery_table({nan_fit["run_id"]: nan_fit}, TOL)
    assert [(r["rel_error"], r["recovered"]) for r in rows] == [(None, False), (None, False)]


def test_pick_fit_source_never_mixes_full_and_smoke():
    full = _seed_records(models=("M1",), config="full")
    smoke = _seed_records(config="smoke")
    sources = [("experiments/s1", full), ("experiments/s1_smoke", smoke)]
    label, records = pick_fit_source(sources)
    assert label == "experiments/s1" and records is full
    rows = seed_recovery_table(records, TOL)
    assert {r["model"] for r in rows} == {"M1"} and {r["config_name"] for r in rows} == {"full"}
    profile = _profile_record()
    sources[0] = ("experiments/s1", {profile["run_id"]: profile})  # a profile is not a fit
    assert pick_fit_source(sources) == ("experiments/s1_smoke", smoke)
    assert pick_fit_source([("experiments/s1", {})]) is None


def test_plot_recovery_seeds_writes_png(mpl, tmp_path):
    path = plot_recovery_seeds(_seed_records(), TOL, tmp_path / "s1_recovery_seeds.png", source="x")
    assert _is_png(path)
    smoke = _seed_records(models=("M2",), config="smoke")
    assert _is_png(plot_recovery_seeds(smoke, TOL, tmp_path / "smoke.png", source="smoke"))
    with pytest.raises(FigureInputMissing):
        plot_recovery_seeds({}, TOL, tmp_path / "none.png")


def test_build_figures_recovery_seeds_from_full_folder_only(mpl, tmp_path):
    _write_records(tmp_path, "s1", list(_seed_records(models=("M1", "M2v2")).values()))
    _write_records(tmp_path, "s1_smoke", list(_seed_records(config="smoke").values()))
    _decisions_fixture(tmp_path / "configs" / "decisions_v1.yaml")
    results = build_figures(tmp_path, only=["s1_recovery_seeds"], log=lambda _: None)
    assert results["s1_recovery_seeds"].startswith("written"), results
    assert _is_png(tmp_path / "reports" / "figures" / "s1_recovery_seeds.png")


def test_build_figures_skips_recovery_seeds_without_fits(tmp_path):
    results = build_figures(tmp_path, only=["s1_recovery_seeds"], log=lambda _: None)
    assert results["s1_recovery_seeds"].startswith("skipped"), results
