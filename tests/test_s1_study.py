"""Tests for the S1 study runner and report (PLAN T5). Fast tests use stub runners and fabricated
records only: they never call train() or compile Diffrax. The end-to-end test is marked slow."""

import json
from dataclasses import asdict, fields, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexpinn.contracts import FOUR_FIELDS, STATUSES, MetricRecord
from apexpinn.eval.s1_report import (
    INTERVAL_V2_DECLARATION,
    M1_DIAG_DECLARATION,
    M2_V2_DECLARATION,
    MISSPEC_CRITERIA_V1,
    MISSPEC_READING_V1,
    P1_DECISION_SENTENCE,
    R2_DECLARATION,
    S1B_V1_FREEZE,
    S1C_SOLVER_DECLARATIONS,
    at_bound_points,
    classify,
    evaluate_misspec,
    failed_points,
    load_records,
    p1_evidence,
    render_report,
    unconverged_points,
    write_report,
)
from apexpinn.physics.identifiability import (
    S1C_BOUNDS,
    ProfilePointError,
    ProfileResult,
    RefinedSide,
    ThetaLayout,
    likelihood_interval,
    sample_prior_physics,
)
from apexpinn.physics.synthetic import MISSPEC_VARIANTS, default_theta_true
from apexpinn.train import s1_study
from apexpinn.train.s1_study import (
    RUN_STATUSES,
    RunSpec,
    build_fit_record,
    build_profile_record,
    config_hash,
    data_key,
    dataset_kwargs,
    full_config,
    init_physics,
    objective_rel_change,
    parse_shard,
    plan_runs,
    prior_key,
    prior_physics,
    profile_grid,
    run_study,
    select_start,
    shard_runs,
    smoke_config,
    start_key,
    theta_values,
)

VARIANTS = ("wrong_phi", "student_t", "piecewise_wear", "drop_early", "traffic")


@pytest.fixture
def x64():
    """Enable float64 for one test only, then restore the previous setting."""
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _quiet(_message):
    pass


def _rid(part, kind, model, variant, noise, seed):
    return RunSpec(part, kind, model, variant, noise, seed).run_id


def _read(out, run_id):
    return json.loads((out / f"{run_id}.json").read_text(encoding="utf-8"))


# --- (1) configs ---


def test_full_config_reads_o6_from_the_decisions_file():
    cfg = full_config()
    assert cfg.name == "full"
    assert cfg.seeds_main == (0, 1, 2, 3, 4)
    assert cfg.noise_levels == (0.05, 0.15)
    assert dict(cfg.tolerance) == {"rho": 0.15, "k": 0.25}
    assert cfg.seeds_misspec == (0, 1, 2)
    assert cfg.misspec_variants == VARIANTS
    assert cfg.misspec_noise == 0.05
    assert cfg.m2_misspec_max_fit_s == 600.0
    assert cfg.n_stints == 48 and cfg.compounds == ("C2", "C3", "C4")
    assert cfg.models_main == ("M1", "M2", "M2v2") and cfg.m2v2_n_starts == 3
    assert cfg.profile_params == ("rho", "k", "gamma", "h_opt", "kappa")
    assert len(cfg.grid_factors) == 13 and 1.0 in cfg.grid_factors
    assert len(cfg.h_opt_offsets) == 13
    assert cfg.h_opt_offsets[0] == -0.35 and cfg.h_opt_offsets[-1] == 0.35 and cfg.h_opt_offsets[6] == 0.0
    assert cfg.m1_width == 32 and cfg.m1_depth == 2
    assert (cfg.m1_train.n_steps, cfg.m1_train.n_warm, cfg.m1_train.n_ramp) == (5000, 200, 800)
    assert cfg.m1_train.float64 and cfg.m2_train.float64
    assert cfg.m1_weights.reg == 1e-6 and cfg.m2_weights.data == 1.0
    assert cfg.m2_train.n_steps == 1000 and cfg.m2_train.learning_rate == 1e-2
    assert (cfg.n_starts, cfg.minimize_maxiter) == (3, 500)
    assert cfg.out_dir == "experiments/s1" and cfg.report_path == "reports/s1_identifiability.md"
    hash(cfg)  # frozen and hashable


def test_full_config_follows_the_decisions_file(tmp_path):
    text = (s1_study.REPO_ROOT / "configs" / "decisions_v1.yaml").read_text(encoding="utf-8")
    assert "n_seeds: 5" in text and "rho_c: 0.15" in text
    path = tmp_path / "decisions.yaml"
    path.write_text(text.replace("n_seeds: 5", "n_seeds: 3").replace("rho_c: 0.15", "rho_c: 0.2"), "utf-8")
    cfg = full_config(path)
    assert cfg.seeds_main == (0, 1, 2)
    assert dict(cfg.tolerance) == {"rho": 0.2, "k": 0.25}


def test_smoke_config_and_validation():
    cfg = smoke_config()
    assert cfg.name == "smoke" and cfg.n_stints == 12
    assert cfg.seeds_main == (0,) and cfg.noise_levels == (0.05,) and cfg.seeds_misspec == (0,)
    assert cfg.misspec_variants == VARIANTS
    assert cfg.m1_train.n_steps == 300 and cfg.m1_train.n_colloc == 16 and cfg.m2_train.n_steps == 100
    assert cfg.profile_params == ("rho", "k") and cfg.n_starts == 1 and cfg.minimize_maxiter == 100
    assert cfg.grid_factors == (0.5, 0.75, 1.0, 1.33, 2.0)
    assert cfg.out_dir == "experiments/s1_smoke"
    assert cfg.report_path == "reports/s1_identifiability_smoke.md"
    assert cfg.models_main == ("M1", "M2", "M2v2") and cfg.m2v2_n_starts == 2
    with pytest.raises(ValueError, match="m2v2_n_starts"):
        replace(cfg, m2v2_n_starts=0)
    with pytest.raises(ValueError, match="models_main"):
        replace(cfg, models_main=("M1", "M3"))
    with pytest.raises(ValueError, match="profile_params"):
        replace(cfg, profile_params=("rho", "theta"))
    with pytest.raises(ValueError, match="grid_factors"):
        replace(cfg, grid_factors=(1.0, 0.5))
    with pytest.raises(ValueError, match="compounds"):
        replace(cfg, compounds=("C9",))
    with pytest.raises(ValueError, match="misspec_variants"):
        replace(cfg, misspec_variants=("well_specified",))
    with pytest.raises(ValueError, match="tolerance"):
        replace(cfg, tolerance=(("gamma", 0.1),))
    with pytest.raises(ValueError, match="n_stints"):
        replace(cfg, n_stints=0)
    assert replace(cfg, tolerance={"rho": 0.1, "k": 0.2}).tolerance == (("k", 0.2), ("rho", 0.1))


# --- (2) run plan ---


def test_plan_counts_and_stable_unique_run_ids():
    cfg = full_config()
    main = plan_runs(cfg, "main")
    assert len(main) == 40
    assert sum(s.kind == "profile" for s in main) == 10
    fits = [s for s in main if s.kind == "fit"]
    assert len(fits) == 30 and {s.model for s in fits} == {"M1", "M2", "M2v2"}
    assert all(sum(s.model == m for s in fits) == 10 for m in ("M1", "M2", "M2v2"))
    misspec = plan_runs(cfg, "misspec")
    assert len(misspec) == 30
    assert sum(s.kind == "profile" for s in misspec) == 15
    assert {s.model for s in misspec if s.kind == "fit"} == {"M1"}
    with_m2 = plan_runs(cfg, "misspec", include_m2_misspec=True)
    assert len(with_m2) == 45
    ids = [s.run_id for s in main + with_m2]
    assert len(set(ids)) == len(ids)
    assert [s.run_id for s in plan_runs(cfg, "main")] == [s.run_id for s in main]
    assert [s.run_id for s in main] == sorted(s.run_id for s in main)
    assert "main_fit_M1_well_specified_noise0.05_seed0" in ids
    assert "main_fit_M2v2_well_specified_noise0.15_seed4" in ids
    assert not any("M2v2" in s.run_id for s in with_m2)  # M2 v2 is a main-study fit only
    assert "main_profile_ode_well_specified_noise0.15_seed4" in ids
    assert "misspec_profile_ode_traffic_noise0.05_seed2" in ids
    assert "misspec_fit_M2_wrong_phi_noise0.05_seed1" in ids

    smoke = smoke_config()
    assert len(plan_runs(smoke, "main")) == 4
    assert len(plan_runs(smoke, "misspec")) == 10
    assert len(plan_runs(smoke, "misspec", include_m2_misspec=True)) == 15
    with pytest.raises(ValueError, match="part"):
        plan_runs(cfg, "extra")
    with pytest.raises(ValueError, match="model"):
        RunSpec("main", "fit", None, "well_specified", 0.05, 0)
    with pytest.raises(ValueError, match="model"):
        RunSpec("main", "profile", "M1", "well_specified", 0.05, 0)
    with pytest.raises(ValueError, match="variant"):
        RunSpec("main", "profile", None, "traffic", 0.05, 0)


# --- (3) keys ---


def _key_bits(key):
    return np.asarray(jax.random.key_data(key))


def test_data_key_is_paired_and_the_prior_draw_is_shared_by_m1_m2_and_profiles():
    cfg = replace(smoke_config(), noise_levels=(0.05, 0.15))
    specs = [s for s in plan_runs(cfg, "main") + plan_runs(cfg, "misspec", True) if s.seed == 0]
    kwargs = [dataset_kwargs(cfg, s) for s in specs]
    for kw in kwargs:
        np.testing.assert_array_equal(_key_bits(kw["key"]), _key_bits(data_key(0)))
    assert {kw["noise_s"] for kw in kwargs} == {0.05, 0.15}
    assert {kw["misspec"] for kw in kwargs} == {"well_specified", *VARIANTS}
    assert all(kw["n_stints"] == cfg.n_stints and kw["compounds"] == cfg.compounds for kw in kwargs)
    assert not np.array_equal(_key_bits(prior_key(0)), _key_bits(data_key(0)))
    assert not np.array_equal(_key_bits(data_key(0)), _key_bits(data_key(1)))

    m1 = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    m2 = RunSpec("main", "fit", "M2", "well_specified", 0.15, 0)
    profile = RunSpec("misspec", "profile", None, "traffic", 0.05, 0)
    reference = jax.tree.leaves(prior_physics(0))
    for spec in (m1, m2, profile):
        for a, b in zip(jax.tree.leaves(init_physics(spec)), reference, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.allclose(np.asarray(prior_physics(1).rho), np.asarray(prior_physics(0).rho))


# --- (4) config hash ---


def test_config_hash_tracks_only_the_settings_a_run_uses():
    cfg = smoke_config()
    m1 = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    m2 = RunSpec("main", "fit", "M2", "well_specified", 0.05, 0)
    prof = RunSpec("main", "profile", None, "well_specified", 0.05, 0)
    base = {s: config_hash(cfg, s) for s in (m1, m2, prof)}
    assert {s: config_hash(cfg, s) for s in (m1, m2, prof)} == base
    assert len(set(base.values())) == 3 and all(len(h) == 64 for h in base.values())

    def changed(new_cfg):
        return {s for s in (m1, m2, prof) if config_hash(new_cfg, s) != base[s]}

    assert changed(replace(cfg, m1_train=replace(cfg.m1_train, n_steps=301))) == {m1}
    assert changed(replace(cfg, m1_width=16)) == {m1}
    assert changed(replace(cfg, m2_train=replace(cfg.m2_train, learning_rate=5e-3))) == {m2}
    assert changed(replace(cfg, grid_factors=(0.5, 1.0, 2.0))) == {prof}
    assert changed(replace(cfg, n_starts=2)) == {prof}
    assert changed(replace(cfg, h_opt_offsets=(-0.1, 0.0, 0.1))) == set()  # h_opt is not profiled here
    assert changed(replace(cfg, n_stints=10)) == {m1, m2, prof}
    assert changed(replace(cfg, name="other", out_dir="elsewhere", m2_misspec_max_fit_s=10.0)) == set()
    assert changed(replace(cfg, tolerance={"rho": 0.1, "k": 0.2})) == set()  # recovery is recomputed
    other_seed = RunSpec("main", "fit", "M1", "well_specified", 0.05, 1)
    assert config_hash(cfg, other_seed) != base[m1]


def test_config_hash_tracks_the_generator_priors_and_key_folds(monkeypatch):
    cfg = smoke_config()
    specs = (
        RunSpec("main", "fit", "M1", "well_specified", 0.05, 0),
        RunSpec("misspec", "profile", None, "traffic", 0.05, 0),
    )
    base = {s: config_hash(cfg, s) for s in specs}
    digest = s1_study.generator_digest()
    assert len(digest) == 64 and digest == s1_study.generator_digest()
    with monkeypatch.context() as m:
        m.setattr(s1_study, "generator_digest", lambda: "0" * 64)
        assert all(config_hash(cfg, s) != base[s] for s in specs)
    with monkeypatch.context() as m:
        m.setitem(s1_study.PRIOR_RANGES, "k", (0.1, 2.0, "log"))
        assert all(config_hash(cfg, s) != base[s] for s in specs)
    with monkeypatch.context() as m:
        m.setattr(s1_study, "PRIOR_FOLD", 999)
        assert all(config_hash(cfg, s) != base[s] for s in specs)
    with monkeypatch.context() as m:
        m.setitem(s1_study.MISSPEC_VARIANTS, "traffic", replace(MISSPEC_VARIANTS["traffic"], traffic_laps=2))
        assert config_hash(cfg, specs[0]) == base[specs[0]] and config_hash(cfg, specs[1]) != base[specs[1]]
    assert {s: config_hash(cfg, s) for s in specs} == base


def test_profile_hash_tracks_the_optimizer_box_max_steps_and_method_version(monkeypatch):
    cfg = smoke_config()
    fit = RunSpec("main", "fit", "M2", "well_specified", 0.05, 0)
    prof = RunSpec("main", "profile", None, "well_specified", 0.05, 0)
    settings = s1_study._settings(cfg, prof)["profile"]
    assert settings["method_version"] == s1_study.PROFILE_METHOD_VERSION == 3
    assert settings["refine_xtol"] == s1_study.REFINE_XTOL == 1e-4  # interval method v2 (PLAN T5)
    assert settings["refine_max_iter"] == s1_study.REFINE_MAX_ITER == 30
    assert settings["bounds"] == {g: list(v) for g, v in sorted(S1C_BOUNDS.items())}
    assert settings["max_steps"] == 4096  # ODEModel.max_steps class default
    assert "profile" not in s1_study._settings(cfg, fit)
    fit_keys = {"schema", "spec", "n_stints", "compounds", "generator", "prior_ranges", "key_folds"}
    assert set(s1_study._settings(cfg, fit)) == fit_keys | {"m2"}  # v1 fit hashes must not change
    assert set(s1_study._settings(cfg, fit)["m2"]) == {"train", "weights", "rtol", "atol"}
    m1_fit = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    assert set(s1_study._settings(cfg, m1_fit)) == fit_keys | {"m1"}
    assert set(s1_study._settings(cfg, m1_fit)["m1"]) == {"width", "depth", "train", "weights"}
    base = {s: config_hash(cfg, s) for s in (fit, prof)}
    with monkeypatch.context() as m:
        m.setitem(s1_study.S1C_BOUNDS, "k", (0.01, 20.0))
        assert config_hash(cfg, prof) != base[prof] and config_hash(cfg, fit) == base[fit]
    with monkeypatch.context() as m:
        m.setattr(s1_study, "PROFILE_METHOD_VERSION", 2)
        assert config_hash(cfg, prof) != base[prof] and config_hash(cfg, fit) == base[fit]
    with monkeypatch.context() as m:
        m.setattr(s1_study, "REFINE_XTOL", 1e-6)
        assert config_hash(cfg, prof) != base[prof] and config_hash(cfg, fit) == base[fit]
    with monkeypatch.context() as m:
        m.setattr(s1_study, "REFINE_MAX_ITER", 50)
        assert config_hash(cfg, prof) != base[prof] and config_hash(cfg, fit) == base[fit]
    with monkeypatch.context() as m:
        m.setattr(s1_study, "_ode_max_steps", lambda: 8192)
        assert config_hash(cfg, prof) != base[prof] and config_hash(cfg, fit) == base[fit]
    assert {s: config_hash(cfg, s) for s in (fit, prof)} == base


def test_generator_digest_covers_the_residual_equations(tmp_path, monkeypatch):
    sources = s1_study.generator_sources()
    assert [p.name for p in sources] == ["synthetic.py", "residuals.py"]
    copies = []
    for path in sources:
        copy = tmp_path / path.name
        text = path.read_text(encoding="utf-8")
        copy.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))  # CRLF copy
        copies.append(copy)
    assert s1_study.source_digest(copies) == s1_study.generator_digest()  # line endings do not matter
    copies[1].write_text(copies[1].read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    edited = s1_study.source_digest(copies)
    assert edited != s1_study.generator_digest()

    cfg = smoke_config()
    spec = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    base = config_hash(cfg, spec)
    try:
        with monkeypatch.context() as m:
            m.setattr(s1_study, "generator_sources", lambda: tuple(copies))
            s1_study.generator_digest.cache_clear()
            assert s1_study.generator_digest() == edited
            assert config_hash(cfg, spec) != base
    finally:
        s1_study.generator_digest.cache_clear()
    assert config_hash(cfg, spec) == base


# S1b v1 hashes read from the frozen records (FEAT-001 of task-s1-v2-methods, Observed).
V1_SEED0_HASHES = {
    "main_fit_M1_well_specified_noise0.05_seed0": (
        "5865e8579449d96979fe7b2b4280101eed794297b2bb391243a7d125ebd6a0b0"
    ),
    "main_fit_M2_well_specified_noise0.05_seed0": (
        "dd615fc5fbbdf4cac0383834e245c2090db0e7cd0abfcbabc0391623c5a31483"
    ),
}


def test_v1_fit_config_hashes_match_the_frozen_s1_records():
    """S1b v1 is frozen (PLAN T5): no later change may alter the config_hash of a v1 M1 or M2 main fit."""
    cfg = full_config()
    for run_id, stored in V1_SEED0_HASHES.items():
        model = "M1" if "_M1_" in run_id else "M2"
        assert config_hash(cfg, RunSpec("main", "fit", model, "well_specified", 0.05, 0)) == stored
    folder = s1_study.REPO_ROOT / "experiments" / "s1"
    paths = sorted(folder.glob("main_fit_M1_*.json")) + sorted(folder.glob("main_fit_M2_*.json"))
    if not paths:
        pytest.skip("no v1 main fit records in experiments/s1")
    expected = {s.run_id for s in plan_runs(cfg, "main") if s.kind == "fit" and s.model in ("M1", "M2")}
    assert len(expected) == 20 and {p.stem for p in paths} == expected  # all 20 frozen v1 fits, no more
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        spec = RunSpec(**record["spec"])
        assert spec.model in ("M1", "M2") and record["config_name"] == "full", path.name
        assert config_hash(cfg, spec) == record["config_hash"], path.name
        if spec.run_id in V1_SEED0_HASHES:
            assert record["config_hash"] == V1_SEED0_HASHES[spec.run_id]


def test_m2v2_settings_hash_and_run_ids_are_separate_from_m2_v1():
    cfg = full_config()
    m2 = RunSpec("main", "fit", "M2", "well_specified", 0.05, 0)
    v2 = RunSpec("main", "fit", "M2v2", "well_specified", 0.05, 0)
    assert v2.run_id == "main_fit_M2v2_well_specified_noise0.05_seed0" and v2.run_id != m2.run_id
    settings = s1_study._settings(cfg, v2)
    fit_keys = {"schema", "spec", "n_stints", "compounds", "generator", "prior_ranges", "key_folds"}
    assert set(settings) == fit_keys | {"m2v2"}
    block = settings["m2v2"]
    assert block["n_starts"] == 3 and block["selection"] == "min_final_scored_rms"
    assert block["method_version"] == s1_study.M2_V2_METHOD_VERSION == 2
    assert {k: block[k] for k in ("train", "weights", "rtol", "atol")} == s1_study._settings(cfg, m2)["m2"]
    assert config_hash(cfg, v2) != config_hash(cfg, m2)
    more = replace(cfg, m2v2_n_starts=4)
    assert config_hash(more, v2) != config_hash(cfg, v2) and config_hash(more, m2) == config_hash(cfg, m2)
    m1 = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    assert config_hash(more, m1) == config_hash(cfg, m1)
    defaults = {f.name: f.default for f in fields(s1_study.StudyConfig)}
    assert defaults["models_main"] == s1_study.V1_MODELS == ("M1", "M2")  # code defaulting to v1 stays v1
    assert defaults["m2v2_n_starts"] == 3


def test_m1_latent_is_hashed_only_when_not_the_default():
    cfg = full_config()
    m1 = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    m2 = RunSpec("main", "fit", "M2", "well_specified", 0.05, 0)
    assert cfg.m1_latent == "linear"
    assert "latent" not in s1_study._settings(cfg, m1)["m1"]  # v1 M1 settings, so v1 hashes are unchanged
    assert config_hash(cfg, m1) == V1_SEED0_HASHES[m1.run_id]
    bounded = replace(cfg, m1_latent="tanh")
    assert s1_study._settings(bounded, m1)["m1"]["latent"] == "tanh"
    assert config_hash(bounded, m1) != config_hash(cfg, m1)
    assert config_hash(bounded, m2) == config_hash(cfg, m2)
    assert config_hash(replace(cfg, m1_latent="linear"), m1) == config_hash(cfg, m1)
    with pytest.raises(ValueError, match="m1_latent"):
        replace(cfg, m1_latent="sigmoid")


def test_run_fit_m1_builds_the_pinn_with_the_configured_latent(monkeypatch):
    """Fake training (no train() call): run_fit passes m1_latent to PINN.init."""

    class Stop(Exception):
        pass

    seen = []

    def fake_train(model, batch, weights, train_cfg, key):
        seen.append(model.latent_mode)
        raise Stop

    monkeypatch.setattr(s1_study, "dataset_for", lambda cfg_, spec_: None)
    monkeypatch.setattr(s1_study, "batch_from_synthetic", lambda ds_: None)
    monkeypatch.setattr(s1_study, "train", fake_train)
    spec = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    for mode in ("linear", "tanh"):
        with pytest.raises(Stop):
            s1_study.run_fit(replace(smoke_config(), m1_latent=mode), spec)
    assert seen == ["linear", "tanh"]


def test_select_start_picks_the_lowest_finite_rms_and_ties_to_the_lowest_index():
    assert select_start([0.3, 0.1, 0.2]) == 1
    assert select_start([0.3, 0.1, 0.1]) == 1
    assert select_start([float("nan"), 0.4, 0.2]) == 2
    assert select_start([float("nan"), float("inf"), None]) == 0
    assert select_start([0.5]) == 0
    assert select_start([float("inf"), 0.7]) == 1


def test_m2v2_starts_are_the_prior_draw_and_the_profile_mle_starts():
    starts = s1_study.m2v2_starts(smoke_config(), 0)  # smoke: 2 starts
    assert len(starts) == 2
    expected = (prior_physics(0), sample_prior_physics(start_key(0, 1)))
    for got, want in zip(starts, expected, strict=True):
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert len(s1_study.m2v2_starts(full_config(), 0)) == 3


def test_run_fit_m2v2_selects_by_residual_rms_before_reading_the_truth(monkeypatch):
    """Fake training (no Diffrax): start 1 has the lowest final RMS, start 2 is non-finite."""
    cfg = replace(smoke_config(), m2v2_n_starts=3)
    spec = RunSpec("main", "fit", "M2v2", "well_specified", 0.05, 0)
    truth = default_theta_true()
    ds = SimpleNamespace(theta_true=truth, rho_target=np.asarray(truth.rho))
    rms = iter([0.09, 0.06, float("nan")])
    wall = iter([1.0, 2.0, 3.0])
    trained = []
    order = []

    def fake_train_ode(cfg_, spec_, batch, physics):
        trained.append(physics)
        log = SimpleNamespace(wall_time_s=next(wall), steps=[0, 1], objective=[2.0, 1.0], normalized={})
        return SimpleNamespace(noise_s=next(rms), physics=physics), log

    real_select, real_truth = s1_study.select_start, s1_study.variant_truth
    monkeypatch.setattr(s1_study, "dataset_for", lambda cfg_, spec_: ds)
    monkeypatch.setattr(s1_study, "batch_from_synthetic", lambda ds_: SimpleNamespace(mask=np.ones(40)))
    monkeypatch.setattr(s1_study, "_train_ode", fake_train_ode)
    monkeypatch.setattr(s1_study, "select_start", lambda v: order.append("select") or real_select(v))
    monkeypatch.setattr(s1_study, "variant_truth", lambda d, c: order.append("truth") or real_truth(d, c))
    record = s1_study._jsonable(s1_study.run_fit(cfg, spec))
    json.dumps(record, allow_nan=False)
    assert order == ["select", "truth"]  # the truth is read only after the start is selected
    diag = record["diagnostics"]
    assert diag["selected_start"] == 1 and diag["selection"] == "min_final_scored_rms"
    assert diag["scored_residual_rms_s"] == pytest.approx(0.06)
    assert diag["residual_ratio"] == pytest.approx(1.2)
    assert [s["index"] for s in diag["starts"]] == [0, 1, 2]
    assert diag["starts"][2]["scored_residual_rms_s"] is None  # NaN, stored as null
    assert [s["fit_s"] for s in diag["starts"]] == [1.0, 2.0, 3.0]
    assert record["estimates"] == pytest.approx(theta_values(trained[1], cfg.compounds))
    assert diag["starts"][1]["estimates"] == record["estimates"]
    assert diag["starts"][1]["rel_error"] == pytest.approx(record["rel_error"])
    assert record["wall_time_s"]["fit"] == pytest.approx(6.0)
    assert record["n_scored"] == 40
    assert "3 prior draws" in record["records"][0]["conditions"]


# --- (5) run_study with stub runners ---


def _stub_record(status="ok", fit_s=1.0):
    entry = MetricRecord("stub quantity, s", "stub population", "stub conditions", "tests", 1.0, "Observed")
    return {"status": status, "records": [asdict(entry)], "wall_time_s": {"fit": fit_s, "total": fit_s}}


class Stub:
    """Runner stub: records calls, can fail or return nonfinite for given run ids, sets M2 fit times."""

    def __init__(self, fail=(), status=None, m2_fit_s=None):
        self.calls = []
        self.fail = set(fail)
        self.status = dict(status or {})
        self.m2_fit_s = dict(m2_fit_s or {})

    def __call__(self, cfg, spec):
        self.calls.append(spec.run_id)
        if spec.run_id in self.fail:
            raise RuntimeError(f"boom {spec.run_id}")
        fit_s = self.m2_fit_s.get(spec.seed, 1.0) if spec.part == "main" and spec.model == "M2" else 1.0
        return _stub_record(self.status.get(spec.run_id, "ok"), fit_s)


def _tiny_cfg():
    return replace(
        smoke_config(),
        seeds_main=(0, 1),
        seeds_misspec=(0,),
        misspec_variants=("wrong_phi", "traffic"),
        out_dir="exp/s1",
    )


def test_run_study_writes_one_record_per_run_and_resumes(tmp_path):
    cfg = _tiny_cfg()
    stub = Stub()
    summary = run_study(cfg, part="main", runner=stub, root=tmp_path, log=_quiet)
    out = tmp_path / cfg.out_dir
    planned = plan_runs(cfg, "main")
    assert len(planned) == 8
    assert {s.model for s in planned if s.kind == "fit"} == {"M1", "M2", "M2v2"}
    assert {p.stem for p in out.glob("*.json")} == {s.run_id for s in planned}
    assert sorted(stub.calls) == sorted(s.run_id for s in planned)
    assert (summary["planned"], summary["run"], summary["skipped"], summary["errors"]) == (8, 8, 0, 0)
    assert not list(out.glob("*.tmp"))
    v2_spec = RunSpec("main", "fit", "M2v2", "well_specified", 0.05, 1)
    v2_record = _read(out, v2_spec.run_id)  # the stub runner writes M2v2 records like any other fit
    assert v2_record["spec"]["model"] == "M2v2" and v2_record["config_hash"] == config_hash(cfg, v2_spec)
    spec = planned[0]
    record = _read(out, spec.run_id)
    assert record["run_id"] == spec.run_id and record["spec"] == asdict(spec)
    assert record["config_hash"] == config_hash(cfg, spec)
    assert record["meta"]["seeds"] == [spec.seed] and record["config_name"] == "smoke"
    assert record["wall_time_s"]["total"] == 1.0

    again = Stub()
    summary = run_study(cfg, part="main", runner=again, root=tmp_path, log=_quiet)
    assert again.calls == []
    assert (summary["run"], summary["skipped"]) == (0, 8)


def test_run_study_refuses_a_changed_config_unless_forced(tmp_path):
    cfg = _tiny_cfg()
    run_study(cfg, part="main", runner=Stub(), root=tmp_path, log=_quiet)
    changed = replace(cfg, m1_train=replace(cfg.m1_train, n_steps=301))
    blocked = Stub()
    with pytest.raises(ValueError, match="main_fit_M1_well_specified_noise0.05_seed0"):
        run_study(changed, part="main", runner=blocked, root=tmp_path, log=_quiet)
    assert blocked.calls == []
    forced = Stub()
    run_study(changed, part="main", force=True, runner=forced, root=tmp_path, log=_quiet)
    m1_ids = sorted(s.run_id for s in plan_runs(cfg, "main") if s.model == "M1")
    assert sorted(forced.calls) == m1_ids
    out = tmp_path / cfg.out_dir
    for spec in plan_runs(cfg, "main"):
        assert _read(out, spec.run_id)["config_hash"] == config_hash(changed, spec)
    assert not list(out.glob("*.tmp"))


def test_run_study_reruns_errors_and_keeps_nonfinite_records(tmp_path):
    cfg = _tiny_cfg()
    fail_id = _rid("main", "profile", None, "well_specified", 0.05, 0)
    nonfinite_id = _rid("main", "fit", "M2", "well_specified", 0.05, 1)
    stub = Stub(fail={fail_id}, status={nonfinite_id: "nonfinite"})
    summary = run_study(cfg, part="main", runner=stub, root=tmp_path, log=_quiet)
    out = tmp_path / cfg.out_dir
    assert summary["errors"] == 1 and summary["run"] == 8
    failed = _read(out, fail_id)
    assert failed["status"] == "error" and "boom" in failed["error"] and failed["records"] == []
    assert _read(out, nonfinite_id)["status"] == "nonfinite"

    rerun = Stub()
    run_study(cfg, part="main", runner=rerun, root=tmp_path, log=_quiet)
    assert rerun.calls == [fail_id]
    assert _read(out, fail_id)["status"] == "ok"
    assert _read(out, nonfinite_id)["status"] == "nonfinite"


class PartialError(RuntimeError):
    """Runner exception that carries the results computed before it, like s1_study.ProfileRunError."""

    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def test_error_records_keep_the_traceback_and_partial_results(tmp_path):
    cfg = _tiny_cfg()
    target = _rid("main", "profile", None, "well_specified", 0.05, 0)
    plain = _rid("main", "profile", None, "well_specified", 0.05, 1)
    message = "run_profile failed in phase profile for k_c[C2]: solver stopped"

    def runner(cfg, spec):
        if spec.run_id == target:
            sse = np.array([1.0, np.nan])
            raise PartialError(message, {"phase": "profile", "param": "k_c[C2]", "sse": sse})
        if spec.run_id == plain:
            raise RuntimeError("x" * 400 + "\nsecond line")
        return _stub_record()

    lines = []
    summary = run_study(cfg, part="main", runner=runner, root=tmp_path, log=lines.append)
    out = tmp_path / cfg.out_dir
    assert summary["errors"] == 2
    record = _read(out, target)
    assert record["status"] == "error" and record["records"] == []
    assert record["error"] == f"PartialError: {message}"
    assert "Traceback" in record["traceback"] and "PartialError" in record["traceback"]
    assert record["partial"] == {"phase": "profile", "param": "k_c[C2]", "sse": [1.0, None]}
    line = next(x for x in lines if x.startswith(target))
    assert "  error  " in line and f"PartialError: {message}" in line

    other = _read(out, plain)
    assert other["partial"] is None and other["error"].endswith("second line")
    line = next(x for x in lines if x.startswith(plain))
    assert "\n" not in line and "RuntimeError: xxx" in line and "second line" not in line  # 300 chars
    ok = _read(out, _rid("main", "fit", "M1", "well_specified", 0.05, 0))
    assert ok["status"] == "ok" and "traceback" not in ok and "partial" not in ok


def test_error_records_survive_a_partial_that_is_not_json(tmp_path):
    """A runner that raises ProfilePointError itself leaves a ProfileResult in .partial."""
    cfg = _tiny_cfg()
    target = _rid("main", "profile", None, "well_specified", 0.05, 0)
    nan = np.array([np.nan])
    result = ProfileResult(nan, nan, nan[:, None], np.zeros(1, bool), np.zeros(1, bool), np.zeros(1, int))

    def runner(cfg, spec):
        if spec.run_id == target:
            raise ProfilePointError("profile of index 0 failed at grid point 0 (value 1): boom", result)
        return _stub_record()

    summary = run_study(cfg, part="main", runner=runner, root=tmp_path, log=_quiet)
    assert summary["errors"] == 1 and summary["run"] == len(plan_runs(cfg, "main"))
    record = _read(tmp_path / cfg.out_dir, target)
    assert record["status"] == "error" and record["error"].startswith("ProfilePointError: ")
    assert isinstance(record["partial"], str) and record["partial"].startswith("ProfileResult(")


def test_stale_error_records_never_block_but_stale_ok_records_do(tmp_path):
    cfg = _tiny_cfg()
    fail_id = _rid("main", "profile", None, "well_specified", 0.05, 0)
    ok_id = _rid("main", "fit", "M1", "well_specified", 0.05, 0)
    run_study(cfg, part="main", runner=Stub(fail={fail_id}), root=tmp_path, log=_quiet)
    out = tmp_path / cfg.out_dir

    def make_stale(run_id):
        record = _read(out, run_id)
        record["config_hash"] = "0" * 64
        (out / f"{run_id}.json").write_text(json.dumps(record), encoding="utf-8")

    make_stale(fail_id)
    rerun = Stub()
    run_study(cfg, part="main", runner=rerun, root=tmp_path, log=_quiet)  # no ValueError
    assert rerun.calls == [fail_id]
    assert _read(out, fail_id)["status"] == "ok"
    profile_spec = RunSpec("main", "profile", None, "well_specified", 0.05, 0)
    assert _read(out, fail_id)["config_hash"] == config_hash(cfg, profile_spec)

    (out / f"{fail_id}.json").unlink()
    run_study(cfg, part="main", runner=Stub(fail={fail_id}), root=tmp_path, log=_quiet)
    assert _read(out, fail_id)["status"] == "error"
    make_stale(fail_id)
    make_stale(ok_id)
    blocked = Stub()
    with pytest.raises(ValueError, match=rf"{ok_id} \(1 run\(s\)"):
        run_study(cfg, part="main", runner=blocked, root=tmp_path, log=_quiet)
    assert blocked.calls == []
    forced = Stub()
    run_study(cfg, part="main", runner=forced, root=tmp_path, log=_quiet, force=True)
    assert sorted(forced.calls) == sorted([fail_id, ok_id])


def test_shards_are_disjoint_and_cover_the_plan(tmp_path):
    cfg = _tiny_cfg()
    plan = plan_runs(cfg, "main")
    a = {s.run_id for s in shard_runs(plan, (0, 2))}
    b = {s.run_id for s in shard_runs(plan, (1, 2))}
    assert a and b and not (a & b) and a | b == {s.run_id for s in plan}
    assert parse_shard("1/3") == (1, 3)
    for bad in ("3/3", "x", "1/0", "1/2/3", "-1/2"):
        with pytest.raises(ValueError, match="shard"):
            parse_shard(bad)
    run_study(cfg, part="main", shard=(0, 2), runner=Stub(), root=tmp_path / "a", log=_quiet)
    run_study(cfg, part="main", shard=(1, 2), runner=Stub(), root=tmp_path / "b", log=_quiet)
    assert {p.stem for p in (tmp_path / "a" / cfg.out_dir).glob("*.json")} == a
    assert {p.stem for p in (tmp_path / "b" / cfg.out_dir).glob("*.json")} == b


def test_run_study_part_all_runs_main_then_misspec(tmp_path):
    cfg = _tiny_cfg()
    summary = run_study(cfg, part="all", runner=Stub(), root=tmp_path, log=_quiet)
    expected = plan_runs(cfg, "main") + plan_runs(cfg, "misspec", include_m2_misspec=True)
    files = {p.stem for p in (tmp_path / cfg.out_dir).glob("*.json")}
    assert files == {s.run_id for s in expected}
    assert summary["planned"] == len(expected) and summary["m2_misspec"]["include"] is True


# --- (6) M2 inclusion rule ---


@pytest.mark.parametrize(("slowest", "included"), [(599.0, True), (601.0, False)])
def test_misspec_includes_m2_only_if_the_slowest_main_m2_fit_is_under_600_s(tmp_path, slowest, included):
    cfg = _tiny_cfg()
    run_study(cfg, part="main", runner=Stub(m2_fit_s={0: 10.0, 1: slowest}), root=tmp_path, log=_quiet)
    stub = Stub()
    summary = run_study(cfg, part="misspec", runner=stub, root=tmp_path, log=_quiet)
    decision = summary["m2_misspec"]
    assert decision["include"] is included
    assert decision["max_fit_s"] == slowest
    assert decision["max_run_id"] == _rid("main", "fit", "M2", "well_specified", 0.05, 1)
    assert set(stub.calls) == {s.run_id for s in plan_runs(cfg, "misspec", included)}
    assert any("_fit_M2_" in c for c in stub.calls) is included


def test_misspec_without_main_m2_records_asks_for_part_main_first(tmp_path):
    cfg = _tiny_cfg()
    with pytest.raises(ValueError, match="run part main first"):
        run_study(cfg, part="misspec", runner=Stub(), root=tmp_path, log=_quiet)
    # One shard of part main leaves a main M2 fit record missing: M2 is undecided, the M1 runs proceed.
    run_study(cfg, part="main", shard=(0, 2), runner=Stub(), root=tmp_path, log=_quiet)
    stub = Stub()
    summary = run_study(cfg, part="misspec", runner=stub, root=tmp_path, log=_quiet)
    assert summary["m2_misspec"]["include"] is None
    assert set(stub.calls) == {s.run_id for s in plan_runs(cfg, "misspec", False)}


def test_a_failing_main_m2_fit_does_not_block_the_m1_misspec_runs(tmp_path):
    cfg = _tiny_cfg()
    m2_id = _rid("main", "fit", "M2", "well_specified", 0.05, 1)
    run_study(cfg, part="main", runner=Stub(fail={m2_id}), root=tmp_path, log=_quiet)
    first = Stub()
    summary = run_study(cfg, part="misspec", runner=first, root=tmp_path, log=_quiet)
    assert summary["m2_misspec"]["include"] is None and summary["m2_misspec"]["missing"] == [m2_id]
    m1_plan = {s.run_id for s in plan_runs(cfg, "misspec", False)}
    assert set(first.calls) == m1_plan and not any("_fit_M2_" in c for c in first.calls)
    assert "undecided" in render_report(load_records(tmp_path / cfg.out_dir), cfg)

    run_study(cfg, part="main", runner=Stub(), root=tmp_path, log=_quiet)  # reruns the failed M2 fit
    second = Stub()
    summary = run_study(cfg, part="misspec", runner=second, root=tmp_path, log=_quiet)
    assert summary["m2_misspec"]["include"] is True
    assert set(second.calls) == {s.run_id for s in plan_runs(cfg, "misspec", True)} - m1_plan


def test_run_study_only_runs_the_named_runs(tmp_path):
    cfg = _tiny_cfg()
    target = _rid("main", "profile", None, "well_specified", 0.05, 1)
    stub = Stub()
    summary = run_study(cfg, part="main", runner=stub, root=tmp_path, log=_quiet, only=[target])
    assert stub.calls == [target] and summary["planned"] == 1
    assert {p.stem for p in (tmp_path / cfg.out_dir).glob("*.json")} == {target}
    rest = Stub()
    run_study(cfg, part="main", runner=rest, root=tmp_path, log=_quiet)
    assert target not in rest.calls and len(rest.calls) == len(plan_runs(cfg, "main")) - 1
    with pytest.raises(ValueError, match="only"):
        run_study(cfg, part="main", runner=Stub(), root=tmp_path, log=_quiet, only=["no_such_run"])
    with pytest.raises(ValueError, match="only"):
        run_study(cfg, part="main", runner=Stub(), root=tmp_path, log=_quiet, only=[])


# --- (7) four-field records ---


def test_written_records_carry_four_fields_and_a_status(tmp_path):
    cfg = _tiny_cfg()
    failing = Stub(fail={_rid("main", "fit", "M1", "well_specified", 0.05, 1)})
    run_study(cfg, part="all", runner=failing, root=tmp_path, log=_quiet)
    paths = list((tmp_path / cfg.out_dir).glob("*.json"))
    assert paths
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] in RUN_STATUSES
        for entry in record["records"]:
            assert all(entry[name] for name in FOUR_FIELDS)
            assert entry["status"] in STATUSES


def test_runner_output_without_four_fields_is_written_as_error(tmp_path):
    def bad_runner(cfg, spec):
        record = _stub_record()
        record["records"][0]["quantity"] = ""
        return record

    cfg = _tiny_cfg()
    run_study(cfg, part="main", runner=bad_runner, root=tmp_path, log=_quiet)
    for path in (tmp_path / cfg.out_dir).glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] == "error" and "quantity" in record["error"]


def test_build_fit_record_errors_flags_and_four_fields():
    cfg = smoke_config()
    spec = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    truth = theta_values(default_theta_true(), cfg.compounds)
    assert len(truth) == 13 and truth["rho_c[C3]"] == pytest.approx(0.06)
    estimates = {n: v * 1.1 for n, v in truth.items()}
    estimates["rho_c[C3]"] = truth["rho_c[C3]"] * 1.2
    diagnostics = {"scored_residual_rms_s": 0.06, "residual_ratio": 1.2, "collapse_flags": []}
    record = build_fit_record(
        cfg,
        spec,
        estimates=estimates,
        truth=truth,
        n_scored=300,
        diagnostics=diagnostics,
        convergence={"objective_rel_change_last20": -1e-4},
        wall_time_s={"fit": 1.0, "total": 2.0},
    )
    assert record["status"] == "ok"
    assert record["rel_error"]["k_c[C2]"] == pytest.approx(0.1)
    assert record["rel_error"]["rho_c[C3]"] == pytest.approx(0.2)
    assert record["recovered"]["k_c[C2]"] and record["recovered"]["rho_c[C2]"]
    assert not record["recovered"]["rho_c[C3]"]
    assert "gamma_c[C2]" not in record["recovered"] and "kappa" not in record["recovered"]
    assert len(record["records"]) == 13 + 2
    for entry in record["records"]:
        MetricRecord(**entry)
        assert all(entry[name] for name in FOUR_FIELDS) and entry["status"] == "Observed"
    json.dumps(s1_study._jsonable(record), allow_nan=False)

    estimates["kappa"] = float("nan")
    broken = build_fit_record(
        cfg,
        spec,
        estimates=estimates,
        truth=truth,
        n_scored=300,
        diagnostics=diagnostics,
        convergence={},
        wall_time_s={"fit": 1.0, "total": 2.0},
    )
    assert broken["status"] == "nonfinite" and broken["rel_error"]["kappa"] is None
    assert json.loads(json.dumps(s1_study._jsonable(broken), allow_nan=False))["estimates"]["kappa"] is None


def test_build_profile_record_status_and_records():
    cfg = smoke_config()
    spec = RunSpec("main", "profile", None, "well_specified", 0.05, 0)
    interval = {"lo": 0.05, "hi": 0.07, "lo_bounded": True, "hi_bounded": True, "threshold": 3.841}
    profiles = {"rho_c[C3]": {"shape": "sharp", "interval": interval, "covers_truth": True}}
    kwargs = dict(
        mle={"values": {"rho_c[C3]": 0.06}, "sse": 1.0},
        profiles=profiles,
        truth={"rho_c[C3]": 0.06},
        sse_truth=1.01,
        n_obs=200,
        wall_time_s={"total": 1.0},
    )
    record = build_profile_record(cfg, spec, spectrum={"smallest_relative": 1e-3}, **kwargs)
    assert record["status"] == "ok" and len(record["records"]) == 3
    for entry in record["records"]:
        assert all(entry[name] for name in FOUR_FIELDS) and entry["status"] in STATUSES
    assert build_profile_record(cfg, spec, spectrum=None, **kwargs)["status"] == "nonfinite"


def _fake_profile_problem(monkeypatch, log_shift=None):
    """Patch run_profile's data and likelihood with a 13-coordinate least-squares problem (no Diffrax).

    Residuals 0.3 (v - v_min) plus one constant 1, so the free minimum is v_min with SSE 1 and n_obs =
    100. v_min is the truth, with log_shift (layout name -> shift in that coordinate) added. The objective
    is NaN where k_c[C2] < 0.6 x truth (a stand-in for an ODE solve that stopped).
    """
    truth = default_theta_true()
    layout = ThetaLayout.from_compounds(smoke_config().compounds)
    v0 = jnp.asarray(layout.to_vector(truth))
    shift = np.zeros(layout.size)
    for name, value in (log_shift or {}).items():
        shift[layout.names.index(name)] = value
    v_min = v0 + jnp.asarray(shift)
    i_k = layout.names.index("k_c[C2]")
    ds = SimpleNamespace(theta_true=truth, rho_target=np.asarray(truth.rho))

    def fake_least_squares(batch, layout_, base, *, rtol, atol):
        def residual_fn(v):
            r = jnp.concatenate([0.3 * (v - v_min), jnp.ones(1, dtype=v.dtype)])
            return jnp.where(v[i_k] < v0[i_k] + np.log(0.6), jnp.nan, r)

        def sse_fn(v):
            return jnp.sum(residual_fn(v) ** 2)

        return sse_fn, residual_fn, 100

    monkeypatch.setattr(s1_study, "dataset_for", lambda cfg, spec: ds)
    monkeypatch.setattr(s1_study, "batch_from_synthetic", lambda ds_: None)
    monkeypatch.setattr(s1_study, "init_physics", lambda spec: truth)
    monkeypatch.setattr(s1_study, "ode_least_squares", fake_least_squares)


def test_run_profile_records_failed_points_and_keeps_running(x64, monkeypatch):
    _fake_profile_problem(monkeypatch)
    spec = RunSpec("main", "profile", None, "well_specified", 0.05, 0)
    record = s1_study._jsonable(s1_study.run_profile(smoke_config(), spec))
    json.dumps(record, allow_nan=False)
    assert record["status"] == "ok"
    mle = record["mle"]
    assert mle["start_failed"] == [False] and mle["at_bound"] == [] and len(mle["n_failed_evals"]) == 1
    assert record["sse_truth"] == pytest.approx(1.0)

    entry = record["profiles"]["k_c[C2]"]  # grid 0.5, 0.75, 1, 1.33, 2 x MLE; 0.5 x is in the NaN region
    assert entry["failed"] == [True, False, False, False, False]
    assert entry["success"][0] is False and entry["n_failed_evals"][0] >= 1
    assert entry["sse"][0] is None and entry["deviance"][0] is None
    assert all(d is not None for d in entry["deviance"][1:])
    assert entry["at_bound"] == [[]] * 5
    interval = entry["interval"]
    assert interval["lo_failed"] is True and interval["lo_bounded"] is False
    assert interval["lo"] == pytest.approx(0.75 * entry["mle"])
    assert interval["hi_bounded"] is True and interval["hi_failed"] is False
    assert entry["shape"] == "flat" and entry["covers_truth"] is True
    # Interval method v2: the open side is untouched, the bounded side is refined by real solves. The
    # profile SSE is 1 + 0.09 d^2 at log distance d from the MLE (n_obs 100), so the exact crossing is
    # d = sqrt(expm1(3.841 / 100) / 0.09), which linear interpolation on the grid undershoots.
    assert entry["interval_method"] == "v2_refined"
    assert entry["refinement"]["lo"]["reason"] == "unbounded" and entry["refinement"]["lo"]["n_evals"] == 0
    hi = entry["refinement"]["hi"]
    assert hi["refined"] is True and hi["reason"] is None and hi["n_evals"] > 2
    assert len(hi["points"]["sse"]) == hi["n_evals"] and hi["value"] == pytest.approx(interval["hi"])
    exact_hi = entry["mle"] * np.exp(np.sqrt(np.expm1(interval["threshold"] / 100) / 0.09))
    assert interval["hi"] == pytest.approx(exact_hi, rel=1e-3)
    linear = entry["interval_linear"]
    assert linear["hi"] < 0.995 * exact_hi and linear["lo_failed"] is True
    assert record["wall_time_s"]["refine"] > 0

    other = record["profiles"]["rho_c[C2]"]
    assert not any(other["failed"]) and other["interval"]["lo_bounded"] and other["interval"]["hi_bounded"]
    assert other["sse_min"] == pytest.approx(1.0)


def _refined_side(coord, n_evals):
    return RefinedSide(
        coord=coord,
        refined=True,
        reason=None,
        n_evals=n_evals,
        n_unconverged=1,
        coords=np.full(n_evals, coord),
        sse=np.ones(n_evals),
        deviance=np.full(n_evals, 3.841),
        success=np.array([True] * (n_evals - 1) + [False]),
    )


def test_profile_entry_uses_the_refined_interval_and_keeps_the_linear_one_as_superseded():
    """Neighbour deviance 26 next to the minimum: the linear interval misses the truth, the refined one
    (fabricated here) covers it, and shape and coverage must follow the refined one."""
    natural = 0.05 * np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    coord = np.log(natural)
    deviance = np.array([30.0, 26.0, 0.0, 26.0, 30.0])
    thetas = np.zeros((5, 13))
    thetas[:, 0] = coord
    prof = ProfileResult(
        grid=coord,
        loss=np.exp(deviance / 100),  # sse_min 1, n_obs 100
        thetas=thetas,
        success=np.ones(5, dtype=bool),
        failed=np.zeros(5, dtype=bool),
        n_failed_evals=np.zeros(5, dtype=int),
    )
    common = dict(
        group="rho",
        index=0,
        natural=natural,
        coord=coord,
        mle_value=0.05,
        truth=0.06,
        sse_fit=1.0,
        n_obs=100,
        bounds=[(-50.0, 50.0)] * 13,
        names=ThetaLayout.from_compounds(("C2", "C3", "C4")).names,
    )
    linear = s1_study._profile_entry(prof, **common)
    assert linear["interval_method"] == "v1_linear"
    assert "interval_linear" not in linear and "refinement" not in linear
    assert linear["interval"]["hi"] < 0.06 and linear["covers_truth"] is False

    coarse = likelihood_interval(coord, deviance)
    refined_iv = replace(coarse, lo=float(np.log(0.042)), hi=float(np.log(0.0615)))
    refined = (refined_iv, _refined_side(refined_iv.lo, 7), _refined_side(refined_iv.hi, 8))
    entry = s1_study._profile_entry(prof, **common, refined=refined)
    assert entry["interval_method"] == "v2_refined"
    assert entry["interval"]["lo"] == pytest.approx(0.042)
    assert entry["interval"]["hi"] == pytest.approx(0.0615)
    assert entry["interval"]["lo_bounded"] and entry["interval"]["hi_bounded"]
    assert entry["interval_linear"] == linear["interval"]
    assert entry["covers_truth"] is True and entry["shape"] == "sharp"
    hi = entry["refinement"]["hi"]
    assert hi["refined"] is True and hi["n_evals"] == 8 and hi["n_unconverged"] == 1
    assert hi["value"] == pytest.approx(0.0615)
    assert entry["refinement"]["lo"]["points"]["value"] == pytest.approx([0.042] * 7)
    json.dumps(s1_study._jsonable(entry), allow_nan=False)


def test_run_profile_keeps_the_mle_and_profiles_inside_the_optimizer_box(x64, monkeypatch):
    """The free minimum of kappa (100) lies outside S1C_BOUNDS (kappa at most 30): the box must hold it."""
    _fake_profile_problem(monkeypatch, log_shift={"kappa": float(np.log(100.0))})
    spec = RunSpec("main", "profile", None, "well_specified", 0.05, 0)
    record = s1_study._jsonable(s1_study.run_profile(smoke_config(), spec))
    assert record["status"] == "ok"
    kappa_hi = S1C_BOUNDS["kappa"][1]
    mle = record["mle"]
    assert mle["values"]["kappa"] == pytest.approx(kappa_hi, rel=1e-8)
    assert mle["at_bound"] == ["kappa"]
    entry = record["profiles"]["rho_c[C2]"]
    assert entry["at_bound"] == [["kappa"]] * len(entry["grid"])
    assert not any(entry["failed"])


def test_run_profile_errors_name_the_phase_and_keep_partial_results(x64, monkeypatch):
    _fake_profile_problem(monkeypatch)
    spec = RunSpec("main", "profile", None, "well_specified", 0.05, 0)
    real_profile = s1_study.profile_likelihood
    i_k = ThetaLayout.from_compounds(smoke_config().compounds).names.index("k_c[C2]")

    def failing_profile(loss_fn, theta_hat, index, grid, **kwargs):
        result = real_profile(loss_fn, theta_hat, index, grid, **kwargs)
        if index == i_k:
            raise ProfilePointError("profile of index 0 failed at grid point 4 (value 1): boom", result)
        return result

    monkeypatch.setattr(s1_study, "profile_likelihood", failing_profile)
    with pytest.raises(s1_study.ProfileRunError) as info:
        s1_study.run_profile(smoke_config(), spec)
    exc = info.value
    assert str(exc).startswith("run_profile failed in phase profile for k_c[C2]: ProfilePointError: ")
    assert isinstance(exc.__cause__, ProfilePointError)
    partial = exc.partial
    json.dumps(partial, allow_nan=False)
    assert partial["phase"] == "profile" and partial["param"] == "k_c[C2]"
    assert partial["mle"]["sse"] == pytest.approx(1.0) and partial["spectrum"] is not None
    assert sorted(partial["profiles"]) == ["rho_c[C2]", "rho_c[C3]", "rho_c[C4]"]  # rho goes first
    point = partial["point"]
    assert point["failed"] == [True, False, False, False, False] and len(point["sse"]) == 5
    k_hat = partial["mle"]["values"]["k_c[C2]"]
    assert point["grid"] == pytest.approx([k_hat * f for f in smoke_config().grid_factors])

    def no_data(cfg, spec):
        raise ValueError("bad data")

    monkeypatch.setattr(s1_study, "dataset_for", no_data)
    with pytest.raises(s1_study.ProfileRunError, match="phase dataset: ValueError: bad data") as info:
        s1_study.run_profile(smoke_config(), spec)
    assert info.value.partial["mle"] is None and info.value.partial["profiles"] == {}


def test_profile_grid_and_objective_change_helpers():
    cfg = smoke_config()
    natural, coord = profile_grid(cfg, "rho", 0.06)
    np.testing.assert_allclose(natural, 0.06 * np.asarray(cfg.grid_factors))
    np.testing.assert_allclose(coord, np.log(natural))
    natural, coord = profile_grid(full_config(), "h_opt", 0.9)
    assert natural[-1] == pytest.approx(0.98) and np.all(np.diff(natural) > 0)
    np.testing.assert_allclose(coord, np.log(natural / (1 - natural)))
    with pytest.raises(ValueError, match="center"):
        profile_grid(cfg, "k", 0.0)
    assert objective_rel_change([10.0, 8.0]) == pytest.approx(-0.2)
    assert objective_rel_change([10.0, 5.0, 4.0, 3.9, 4.0, 3.8]) == pytest.approx(-0.05)
    assert objective_rel_change([1.0]) is None


def test_load_records_skips_unreadable_and_tmp_files(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"run_id": "a", "status": "ok"}), encoding="utf-8")
    (tmp_path / "b.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "c.json.tmp").write_text(json.dumps({"run_id": "c"}), encoding="utf-8")
    assert set(load_records(tmp_path)) == {"a"}
    assert load_records(tmp_path / "missing") == {}


# --- report on fabricated records ---

NAMES = ("k_c[C3]", "rho_c[C3]", "gamma_c[C3]", "h_opt_c[C3]", "kappa")
TRUTH = {"k_c[C3]": 0.35, "rho_c[C3]": 0.06, "gamma_c[C3]": 3.0, "h_opt_c[C3]": 0.6, "kappa": 1.0}


def _report_cfg():
    return replace(
        smoke_config(),
        compounds=("C3",),
        seeds_main=(0, 1, 2),
        noise_levels=(0.05, 0.15),
        seeds_misspec=(0, 1, 2),
        profile_params=("rho", "k", "gamma", "h_opt", "kappa"),
    )


def _fit(spec):
    rel_error = dict.fromkeys(NAMES, 0.01)
    return {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "status": "ok",
        "estimates": {n: TRUTH[n] * (1 + rel_error[n]) for n in NAMES},
        "truth": dict(TRUTH),
        "rel_error": rel_error,
        "diagnostics": {"residual_ratio": 1.3, "collapse_flags": [], "scored_residual_rms_s": 0.065},
        "convergence": {"objective_rel_change_last20": -1e-4},
        "wall_time_s": {"fit": 2.0, "total": 3.0},
        "records": [],
    }


def _profile(spec):
    profiles = {}
    for n in NAMES:
        interval = {"lo": 0.9 * TRUTH[n], "hi": 1.1 * TRUTH[n], "lo_bounded": True, "hi_bounded": True}
        profiles[n] = {
            "shape": "sharp",
            "covers_truth": True,
            "truth": TRUTH[n],
            "mle": TRUTH[n],
            "interval": {**interval, "threshold": 3.841},
        }
    return {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "status": "ok",
        "profiles": profiles,
        "spectrum": {"smallest_relative": 1e-3, "weakest_loadings": dict.fromkeys(NAMES, 0.1)},
        "mle": {"sse": 1.0},
        "sse_truth": 1.01,
        "wall_time_s": {"total": 5.0},
        "records": [],
    }


def _records(cfg):
    specs = plan_runs(cfg, "main") + plan_runs(cfg, "misspec", include_m2_misspec=True)
    return {s.run_id: _fit(s) if s.kind == "fit" else _profile(s) for s in specs}


def _mfit(model, variant, seed):
    return _rid("misspec", "fit", model, variant, 0.05, seed)


def _mprof(variant, seed):
    return _rid("misspec", "profile", None, variant, 0.05, seed)


def _main_fit(model, noise, seed):
    return _rid("main", "fit", model, "well_specified", noise, seed)


def _main_prof(noise, seed):
    return _rid("main", "profile", None, "well_specified", noise, seed)


def test_classify_labels_follow_the_declared_rules():
    cfg = _report_cfg()
    recs = _records(cfg)
    recs[_main_fit("M2", 0.05, 1)]["rel_error"]["rho_c[C3]"] = 0.2  # beyond the 15% O6 tolerance
    for seed in (0, 1, 2):
        recs[_main_prof(0.05, seed)]["profiles"]["gamma_c[C3]"]["shape"] = "flat"
    recs[_main_prof(0.05, 1)]["profiles"]["h_opt_c[C3]"]["shape"] = "wide"
    for seed in (0, 1):
        recs[_main_prof(0.15, seed)]["profiles"]["k_c[C3]"]["shape"] = "flat"
    labels = classify(recs, cfg)
    assert labels[("rho_c[C3]", 0.05, "M1")]["label"] == "identifiable"
    assert labels[("rho_c[C3]", 0.05, "M2")]["label"] == "weakly_identifiable"
    assert labels[("rho_c[C3]", 0.05, "M2")]["recovered"] == [True, False, True]
    assert labels[("gamma_c[C3]", 0.05, "profile")]["label"] == "non_identifiable"
    assert labels[("gamma_c[C3]", 0.05, "profile")]["recovered"] is None
    assert ("gamma_c[C3]", 0.05, "M1") not in labels and ("rho_c[C3]", 0.05, "profile") not in labels
    assert labels[("h_opt_c[C3]", 0.05, "profile")]["label"] == "weakly_identifiable"
    assert labels[("kappa", 0.05, "profile")]["label"] == "identifiable"
    assert labels[("k_c[C3]", 0.15, "M1")]["label"] == "non_identifiable"
    assert labels[("k_c[C3]", 0.15, "M2")]["label"] == "non_identifiable"
    assert labels[("k_c[C3]", 0.05, "M1")]["label"] == "identifiable"

    del recs[_main_fit("M1", 0.15, 1)]
    recs[_main_prof(0.05, 2)]["status"] = "error"
    labels = classify(recs, cfg)
    assert labels[("rho_c[C3]", 0.15, "M1")]["label"] == "missing"
    assert labels[("rho_c[C3]", 0.15, "M2")]["label"] == "identifiable"
    assert all(v["label"] == "missing" for k, v in labels.items() if k[1] == 0.05)


def test_misspec_criterion_1_uses_one_and_a_half_times_the_o6_tolerance():
    cfg = _report_cfg()
    recs = _records(cfg)
    for seed in (0, 1, 2):
        recs[_mfit("M1", "student_t", seed)]["rel_error"].update({"rho_c[C3]": 0.22, "k_c[C3]": -0.37})
        recs[_mfit("M1", "wrong_phi", seed)]["rel_error"]["rho_c[C3]"] = 0.9  # judged by criterion 4
    recs[_mfit("M1", "drop_early", 1)]["rel_error"]["k_c[C3]"] = 0.38
    recs[_mfit("M1", "traffic", 0)]["rel_error"]["rho_c[C3]"] = -0.23
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    assert out["models"] == ["M1", "M2"] and out["m2_decision"]["include"] is True
    student = out["by_variant"]["student_t"]["M1"]["criterion_1"]
    assert student["rho_c[C3]"]["outcome"] == "pass" and student["k_c[C3]"]["outcome"] == "pass"
    assert student["rho_c[C3]"]["limit"] == pytest.approx(0.225)
    assert student["k_c[C3]"]["limit"] == pytest.approx(0.375)
    drop = out["by_variant"]["drop_early"]["M1"]["criterion_1"]
    assert drop["k_c[C3]"]["outcome"] == "fail" and drop["rho_c[C3]"]["outcome"] == "pass"
    assert out["by_variant"]["traffic"]["M1"]["criterion_1"]["rho_c[C3]"]["outcome"] == "fail"
    assert out["by_variant"]["traffic"]["M1"]["summary"]["criterion_1"] == "fail"
    wrong = out["by_variant"]["wrong_phi"]["M1"]["criterion_1"]
    assert wrong["rho_c[C3]"]["outcome"] == "governed_by_criterion_4"
    assert wrong["k_c[C3]"]["outcome"] == "pass"
    assert out["by_variant"]["student_t"]["M2"]["criterion_1"]["rho_c[C3]"]["outcome"] == "pass"
    assert out["silent_failures"] == []  # every failing run has residual ratio 1.3


def test_misspec_criteria_1_and_2_skip_parameters_not_identifiable_in_the_main_study():
    cfg = _report_cfg()
    recs = _records(cfg)
    for seed in (0, 1, 2):
        recs[_main_prof(0.05, seed)]["profiles"]["k_c[C3]"]["shape"] = "flat"
    recs[_main_prof(0.05, 0)]["profiles"]["gamma_c[C3]"]["shape"] = "wide"
    recs[_main_fit("M2", 0.05, 0)]["rel_error"]["rho_c[C3]"] = 0.5
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    m1 = out["by_variant"]["student_t"]["M1"]
    assert m1["criterion_1"]["k_c[C3]"]["outcome"] == "not_applicable"
    assert m1["criterion_2"]["k_c[C3]"]["outcome"] == "not_applicable"
    assert m1["criterion_2"]["gamma_c[C3]"]["outcome"] == "not_applicable"
    assert m1["criterion_1"]["rho_c[C3]"]["outcome"] == "pass"
    assert m1["criterion_2"]["rho_c[C3]"]["outcome"] == "pass"
    m2 = out["by_variant"]["student_t"]["M2"]
    assert m2["criterion_1"]["rho_c[C3]"]["outcome"] == "not_applicable"  # M2 missed rho on seed 0
    assert m2["criterion_2"]["rho_c[C3]"]["outcome"] == "not_applicable"
    assert m1["criterion_2"]["kappa"]["outcome"] == "pass"


def test_misspec_criterion_2_needs_coverage_on_two_of_three_seeds():
    cfg = _report_cfg()
    recs = _records(cfg)
    recs[_mprof("student_t", 2)]["profiles"]["rho_c[C3]"]["covers_truth"] = False
    for seed in (0, 1):
        recs[_mprof("traffic", seed)]["profiles"]["rho_c[C3]"]["covers_truth"] = False
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    student = out["by_variant"]["student_t"]["M1"]["criterion_2"]["rho_c[C3]"]
    assert student["outcome"] == "pass" and student["covered"] == [True, True, False]
    assert student["needed"] == 2
    for model in ("M1", "M2"):
        assert out["by_variant"]["traffic"][model]["criterion_2"]["rho_c[C3]"]["outcome"] == "fail"


def test_misspec_criterion_3_flags_low_residual_ratio_and_collapse():
    cfg = _report_cfg()
    recs = _records(cfg)
    recs[_mfit("M1", "piecewise_wear", 1)]["diagnostics"]["residual_ratio"] = 1.1
    recs[_mfit("M1", "student_t", 0)]["diagnostics"]["collapse_flags"] = ["k_c[C3]"]
    recs[_mfit("M1", "drop_early", 2)]["diagnostics"]["residual_ratio"] = 1.0  # ratio not required here
    recs[_mfit("M2", "traffic", 2)]["diagnostics"]["residual_ratio"] = 1.19
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    outcome = {
        (variant, model): result["criterion_3"]["outcome"]
        for variant, per_model in out["by_variant"].items()
        for model, result in per_model.items()
    }
    assert outcome[("piecewise_wear", "M1")] == "fail"
    assert outcome[("piecewise_wear", "M2")] == "pass"
    assert outcome[("student_t", "M1")] == "fail"
    assert outcome[("drop_early", "M1")] == "pass"
    assert outcome[("traffic", "M1")] == "pass"
    assert outcome[("traffic", "M2")] == "fail"


def test_misspec_criterion_4_compares_the_rho_shift_with_phi_used_minus_phi_true():
    shift = MISSPEC_VARIANTS["wrong_phi"].rho_shift
    assert shift == pytest.approx(0.01)
    cfg = _report_cfg()
    recs = _records(cfg)
    for model in ("M1", "M2"):
        for seed in (0, 1, 2):
            recs[_main_fit(model, 0.05, seed)]["estimates"]["rho_c[C3]"] = 0.060
            recs[_mfit(model, "wrong_phi", seed)]["estimates"]["rho_c[C3]"] = 0.0705  # shift 0.0105
    recs[_mfit("M2", "wrong_phi", 2)]["estimates"]["rho_c[C3]"] = 0.074  # shift 0.014, 40% off
    recs[_main_fit("M1", 0.15, 0)]["estimates"]["rho_c[C3]"] = 1.0  # other noise level: not used
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    m1 = out["by_variant"]["wrong_phi"]["M1"]["criterion_4"]
    assert m1["outcome"] == "pass" and m1["expected_shift"] == pytest.approx(shift)
    np.testing.assert_allclose(m1["per_compound"]["rho_c[C3]"]["deltas"], [0.0105] * 3, atol=1e-12)
    m2 = out["by_variant"]["wrong_phi"]["M2"]["criterion_4"]
    assert m2["outcome"] == "fail"
    assert out["by_variant"]["traffic"]["M1"]["criterion_4"]["outcome"] == "not_applicable"


def test_silent_failure_is_a_wrong_fit_without_any_diagnostic_flag():
    cfg = _report_cfg()
    recs = _records(cfg)
    quiet = recs[_mfit("M1", "student_t", 2)]
    quiet["rel_error"]["rho_c[C3]"] = 0.3
    quiet["diagnostics"]["residual_ratio"] = 1.0
    flagged_ratio = recs[_mfit("M1", "drop_early", 0)]
    flagged_ratio["rel_error"]["rho_c[C3]"] = 0.3
    flagged_collapse = recs[_mfit("M1", "traffic", 1)]
    flagged_collapse["rel_error"]["k_c[C3]"] = 0.5
    flagged_collapse["diagnostics"].update(residual_ratio=1.0, collapse_flags=["k_c[C3]"])
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    found = [(f["variant"], f["model"], f["seed"], f["param"]) for f in out["silent_failures"]]
    assert found == [("student_t", "M1", 2, "rho_c[C3]")]
    assert out["silent_failures"][0]["run_id"] == _mfit("M1", "student_t", 2)


def test_a_wrong_phi_shift_without_any_flag_is_a_silent_failure():
    cfg = _report_cfg()
    recs = _records(cfg)
    for model in ("M1", "M2"):
        for seed in (0, 1, 2):
            recs[_main_fit(model, 0.05, seed)]["estimates"]["rho_c[C3]"] = 0.060
            recs[_mfit(model, "wrong_phi", seed)]["estimates"]["rho_c[C3]"] = 0.070  # shift 0.010: pass
    recs[_mfit("M1", "wrong_phi", 1)]["estimates"]["rho_c[C3]"] = 0.0605  # shift 0.0005: fail
    recs[_mfit("M1", "wrong_phi", 1)]["diagnostics"]["residual_ratio"] = 1.05  # and no flag
    recs[_mfit("M2", "wrong_phi", 0)]["estimates"]["rho_c[C3]"] = 0.0605  # fails, but ratio 1.3 flags it
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    assert out["by_variant"]["wrong_phi"]["M1"]["criterion_4"]["outcome"] == "fail"
    assert out["by_variant"]["wrong_phi"]["M2"]["criterion_4"]["outcome"] == "fail"
    found = [(f["variant"], f["model"], f["seed"], f["param"]) for f in out["silent_failures"]]
    assert found == [("wrong_phi", "M1", 1, "rho_c[C3]")]
    assert out["silent_failures"][0]["criterion"] == 4
    text = render_report(recs, cfg)
    assert "rho_c[C3] fails criterion 4 with no diagnostic flag" in text


def test_unconverged_profile_points_are_counted_and_marked():
    grid = [0.03, 0.045, 0.054, 0.06, 0.066, 0.075, 0.09]
    interval = {"lo": 0.05, "hi": 0.07, "lo_bounded": True, "hi_bounded": True, "threshold": 3.841}
    entry = {"grid": grid, "success": [True] * 7, "interval": interval, "shape": "sharp"}
    assert unconverged_points(entry) == (0, False)
    assert unconverged_points({**entry, "success": [True] * 6 + [False]}) == (1, False)  # beyond the bracket
    assert unconverged_points({**entry, "success": [True] * 5 + [False, True]}) == (1, True)  # hi bracket
    assert unconverged_points({**entry, "success": [True, False] + [True] * 5}) == (1, True)  # lo bracket
    assert unconverged_points({**entry, "success": [True] * 3 + [False] + [True] * 3}) == (1, True)  # inside
    assert unconverged_points({"shape": "sharp"}) == (None, False)
    assert unconverged_points({**entry, "success": [True] * 6 + [None], "interval": None}) == (1, False)

    cfg = _report_cfg()
    recs = _records(cfg)
    bad = {**entry, "success": [True] * 3 + [False] + [True] * 3, "truth": 0.06, "mle": 0.06}
    recs[_main_prof(0.05, 0)]["profiles"]["rho_c[C3]"].update(bad)
    recs[_mprof("traffic", 1)]["profiles"]["rho_c[C3]"].update(bad)
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    c2 = out["by_variant"]["traffic"]["M1"]["criterion_2"]["rho_c[C3]"]
    assert c2["shapes"] == ["sharp", "sharp", "sharp"]
    assert c2["rests_on_unconverged"] == [False, True, False]
    text = render_report(recs, cfg)
    assert "1, interval rests on one" in text and "sharp*" in text
    assert "shapes sharp, sharp*, sharp" in text


def test_failed_and_at_bound_profile_points_are_counted_and_shown():
    grid = [0.03, 0.045, 0.054, 0.06, 0.066, 0.075, 0.09]
    interval = {"lo": 0.05, "hi": 0.07, "lo_bounded": True, "hi_bounded": True, "threshold": 3.841}
    clean = {"grid": grid, "success": [True] * 7, "failed": [False] * 7, "at_bound": [[]] * 7}
    entry = {**clean, "interval": interval, "shape": "sharp", "truth": 0.06, "mle": 0.06}
    assert failed_points(entry) == 0 and at_bound_points(entry) == 0
    at_bound = [[], ["k_c[C3]"], [], [], [], [], ["k_c[C3]", "kappa"]]
    bad = {**entry, "success": [False] + [True] * 6, "failed": [True] + [False] * 6, "at_bound": at_bound}
    assert failed_points(bad) == 1 and at_bound_points(bad) == 2
    assert unconverged_points(bad) == (0, False)  # a failed point is not counted as unconverged
    mixed = {**bad, "success": [False, True, True, False, True, True, True]}
    assert unconverged_points(mixed) == (1, True)
    assert failed_points({"grid": grid}) is None and at_bound_points({"grid": grid}) is None
    assert failed_points({**entry, "failed": [True]}) is None
    assert at_bound_points({**entry, "at_bound": [[]]}) is None
    assert failed_points(None) is None and at_bound_points(None) is None

    cfg = _report_cfg()
    recs = _records(cfg)
    rid = _main_prof(0.05, 0)
    recs[rid]["profiles"]["rho_c[C3]"].update(bad)
    failed_side = {**interval, "lo": 0.045, "lo_bounded": False, "lo_failed": True, "hi_failed": False}
    recs[rid]["profiles"]["rho_c[C3]"]["interval"] = failed_side
    recs[rid]["mle"]["at_bound"] = ["gamma_c[C3]"]
    text = render_report(recs, cfg)
    rows = [line for line in text.splitlines() if line.startswith(f"| {rid} | ")]
    assert rows[0].endswith("| gamma_c[C3] |")  # spectrum and MLE table: MLE coordinates on the box edge
    row = next(r for r in rows if r.startswith(f"| {rid} | rho_c[C3] |"))
    assert "| [open at failed point 0.0450, 0.0700] |" in row  # a failed side shows the last finite point
    assert row.endswith("| 0 | 1 | 2 |")  # unconverged (failed points excluded), failed, at the box edge
    assert next(r for r in rows if r.startswith(f"| {rid} | kappa |")).endswith("| n/a | n/a | n/a |")
    assert "| Unconverged points | Failed points | At box edge |" in text


def test_missing_records_give_missing_never_pass():
    cfg = _report_cfg()
    recs = _records(cfg)
    del recs[_mfit("M1", "student_t", 1)]
    del recs[_mprof("traffic", 0)]
    recs[_mfit("M1", "drop_early", 0)]["status"] = "error"
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    student = out["by_variant"]["student_t"]["M1"]
    assert {e["outcome"] for e in student["criterion_1"].values()} == {"missing"}
    assert student["criterion_3"]["outcome"] == "missing"
    assert student["summary"]["criterion_1"] == "missing"
    assert out["by_variant"]["drop_early"]["M1"]["criterion_3"]["outcome"] == "missing"
    traffic = out["by_variant"]["traffic"]["M1"]["criterion_2"]
    assert {e["outcome"] for e in traffic.values()} == {"missing"}

    recs = _records(cfg)
    recs[_main_fit("M1", 0.05, 2)]["status"] = "error"
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    wrong = out["by_variant"]["wrong_phi"]["M1"]
    assert wrong["criterion_4"]["outcome"] == "missing"
    assert wrong["criterion_1"]["k_c[C3]"]["outcome"] == "missing"  # the main-study label is missing

    del recs[_main_fit("M2", 0.05, 0)]
    out = evaluate_misspec(recs, classify(recs, cfg), cfg)
    assert out["m2_decision"]["include"] is None and out["models"] == ["M1"]


def test_p1_evidence_is_mechanical_and_reports_each_trigger():
    cfg = _report_cfg()
    recs = _records(cfg)
    for spec in plan_runs(cfg, "main"):
        if spec.kind == "fit":  # well-specified fits that reach the noise level
            recs[spec.run_id]["diagnostics"]["residual_ratio"] = 1.0
    base = p1_evidence(recs, classify(recs, cfg), cfg)
    assert base["note"] == "mechanical trigger check, not the decision"
    assert base["r1"]["status"] == "trigger_condition_not_met"
    assert base["r2"]["status"] == "indicator_absent" and base["r2"]["need_seeds"] == 3
    assert base["e5"]["status"] == "not_refuted"

    recs[_main_prof(0.15, 0)]["profiles"]["h_opt_c[C3]"]["shape"] = "flat"
    recs[_main_prof(0.15, 1)]["profiles"]["h_opt_c[C3]"]["shape"] = "flat"
    recs[_main_prof(0.15, 2)]["profiles"]["k_c[C3]"]["shape"] = "wide"
    for seed in (0, 1, 2):
        recs[_main_fit("M1", 0.05, seed)]["rel_error"]["k_c[C3]"] = 0.6  # recovery failure, no stall
    recs[_main_prof(0.05, 1)]["spectrum"]["smallest_relative"] = 1e-7
    ev = p1_evidence(recs, classify(recs, cfg), cfg)
    assert ev["r1"]["status"] == "trigger_condition_met"
    assert [e["param"] for e in ev["r1"]["non_identifiable"]] == ["h_opt_c[C3]"]
    assert {e["param"] for e in ev["r1"]["weakly_identifiable"]} == {"k_c[C3]"}
    assert ev["r2"]["status"] == "indicator_absent"  # recovery failure alone is not a stall
    assert ev["e5"]["status"] == "refuted"

    for seed in (0, 1, 2):  # plateau (objective change -1e-4) above the noise level: a stall
        recs[_main_fit("M1", 0.15, seed)]["diagnostics"]["residual_ratio"] = 1.5
    recs[_main_fit("M1", 0.15, 0)]["convergence"]["normalized"] = {"data": [1.0, 0.2, 0.19], "wear": [1.0]}
    ev = p1_evidence(recs, classify(recs, cfg), cfg)
    assert ev["r2"]["status"] == "indicator_present"
    row = next(r for r in ev["r2"]["rows"] if r["run_id"] == _main_fit("M1", 0.15, 0))
    assert row["stalled"] is True and row["final_normalized"] == {"data": 0.19, "wear": 1.0}
    assert f"| {_main_fit('M1', 0.15, 0)} | M1 | 0.15 | 0 | ok | yes | " in render_report(recs, cfg)

    del recs[_main_fit("M2", 0.15, 1)]  # an M2 record of a stalled seed is missing
    assert p1_evidence(recs, classify(recs, cfg), cfg)["r2"]["status"] == "incomplete"

    smoke = p1_evidence({}, classify({}, smoke_config()), smoke_config())
    assert smoke["r1"]["status"] == "not_evaluable"
    assert smoke["r2"]["status"] == "incomplete" and smoke["e5"]["status"] == "incomplete"

    smoke = p1_evidence({}, classify({}, smoke_config()), smoke_config())
    assert smoke["r1"]["status"] == "not_evaluable"
    assert smoke["r2"]["status"] == "incomplete" and smoke["e5"]["status"] == "incomplete"


HEADINGS = (
    "S1b recovery",
    "S1c identifiability",
    "S1d misspecification",
    "Wall time",
    "P1 trigger evidence",
    "Missing runs",
    "How to run",
)


def test_render_report_has_every_section_and_the_declared_text(tmp_path):
    cfg = _report_cfg()
    recs = _records(cfg)
    text = render_report(recs, cfg)
    for heading in HEADINGS:
        assert f"\n## {heading}\n" in text, heading
    assert len(MISSPEC_CRITERIA_V1) == 4
    for criterion in MISSPEC_CRITERIA_V1:
        assert criterion in text
    assert "stays within 1.5× its `O6` tolerance (`ρ_c` 22.5%, `k_c` 37.5%) on all 3 seeds" in text
    assert "covers its variant truth on at least 2 of 3 seeds" in text
    assert "rises above `noise_s` by at least 20% for `piecewise_wear` and `traffic`" in text
    assert "is within 30% of `φ_used − φ_true` for each compound" in text
    assert P1_DECISION_SENTENCE in text
    assert "not the decision" in text
    assert "Smoke numbers do not answer S1" in text
    assert _main_fit("M1", 0.05, 0) in text and _main_prof(0.15, 2) in text
    assert "Slowest main-study M2 fit: 2.0 s (run main_fit_M2_well_specified_noise0.05_seed0)" in text
    assert "None: every planned run has a finished record." in text
    assert "\u2014" not in text and "\u2013" not in text  # no em or en dashes

    full = render_report(recs, replace(cfg, name="full"))
    assert "Smoke numbers do not answer S1" not in full and "--config full" in full

    del recs[_main_fit("M1", 0.05, 1)]
    assert f"- {_main_fit('M1', 0.05, 1)}: no record" in render_report(recs, cfg)

    empty = render_report({}, smoke_config())
    for heading in HEADINGS:
        assert f"\n## {heading}\n" in empty, heading
    assert "no record" in empty and "Smoke numbers do not answer S1" in empty

    path = write_report(tmp_path / "reports" / "s1.md", text)
    assert text.endswith("\n") and path.read_text(encoding="utf-8") == text
    assert not list((tmp_path / "reports").glob("*.tmp"))


def test_report_shows_the_interval_method_the_superseded_linear_interval_and_refinement_counts():
    cfg = _report_cfg()
    recs = _records(cfg)
    plain = render_report(recs, cfg)
    assert "Interval method per main profile record: v2 refined 0, v1 linear 6." in plain
    assert "mix interval methods" not in plain
    rid = _main_prof(0.05, 0)
    lost = {"refined": False, "reason": "failed_solve", "n_evals": 2, "n_unconverged": 1}
    recs[rid]["profiles"]["rho_c[C3]"].update(
        interval_method="v2_refined",
        interval_linear={**recs[rid]["profiles"]["rho_c[C3]"]["interval"], "lo": 0.057, "hi": 0.0582},
        refinement={"lo": {"refined": True, "reason": None, "n_evals": 7, "n_unconverged": 0}, "hi": lost},
    )
    text = render_report(recs, cfg)
    assert "Interval method per main profile record: v2 refined 1, v1 linear 5." in text
    assert "**Warning**: the profile records mix interval methods" in text
    header = "| Interval method | v1 linear interval (Superseded) | Refinement | Unconverged points |"
    assert f"| Covers truth {header}" in text
    rows = [line for line in text.splitlines() if line.startswith(f"| {rid} | ")]
    row = next(r for r in rows if r.startswith(f"| {rid} | rho_c[C3] |"))
    refinement = "lo 7 evals, hi kept v1 linear (failed_solve), 1 unconverged"
    assert f"| v2 refined | [0.0570, 0.0582] | {refinement} |" in row
    old = next(r for r in rows if r.startswith(f"| {rid} | kappa |"))  # a record from before version 3
    assert "| v1 linear | n/a | n/a |" in old
    assert f"- {INTERVAL_V2_DECLARATION}" in text


def test_report_shows_the_recovery_tally_and_m2_v1_next_to_m2_v2():
    cfg = _report_cfg()
    assert cfg.models_main == ("M1", "M2", "M2v2")
    recs = _records(cfg)
    recs[_main_fit("M2", 0.05, 0)]["rel_error"]["rho_c[C3]"] = 0.5  # v1 misses rho on seed 0
    v2 = recs[_main_fit("M2v2", 0.05, 0)]  # v2 recovers it from its second start
    v2["diagnostics"].update(
        residual_ratio=1.1,
        selected_start=1,
        starts=[{"index": 0, "residual_ratio": 1.9}, {"index": 1, "residual_ratio": 1.1}],
    )
    del recs[_main_fit("M2v2", 0.15, 2)]
    same = recs[_main_fit("M2v2", 0.05, 2)]["diagnostics"]  # start 0 reproduces v1 up to rounding
    same.update(selected_start=0, starts=[{"index": 0, "residual_ratio": 1.3 * (1 + 1e-9)}])
    text = render_report(recs, cfg)
    assert "### Recovered fits per model and noise level" in text
    assert "| Model | Method version | Noise s | Recovered | Missing |" in text
    assert "| M1 | v1 | all | 6 of 6 | 0 |" in text
    assert "| M2 | v1 | 0.05 | 2 of 3 | 0 |" in text and "| M2 | v1 | all | 5 of 6 | 0 |" in text
    assert "| M2v2 | v2 | 0.05 | 3 of 3 | 0 |" in text
    assert "| M2v2 | v2 | 0.15 | 2 of 3 | 1 |" in text and "| M2v2 | v2 | all | 5 of 6 | 1 |" in text
    assert "### `M2` v1 vs `M2` v2" in text
    assert "| 0.05 | 0 | no | 1.300 | rho_c[C3] | yes | 1.100 | none | 1 | 0: 1.900, 1: 1.100 | no |" in text
    assert "| 0.05 | 1 | yes | 1.300 | none | yes | 1.300 | none | n/a | n/a | n/a |" in text
    assert "| 0.05 | 2 | yes | 1.300 | none | yes | 1.300 | none | 0 | 0: 1.300 | yes |" in text
    missing = "| missing | missing | missing | missing | missing | missing |"
    assert f"| 0.15 | 2 | yes | 1.300 | none {missing}" in text
    assert "| v2 residual ratio per start | Start 0 matches v1 |" in text
    rid = _main_fit("M2v2", 0.05, 0)  # start 0 ratio 1.9 against the v1 ratio 1.3
    warning = "**Warning**: `M2v2` start 0 does not reproduce the `M2` v1 residual ratio in 1 record(s)"
    assert f"{warning}: {rid}." in text
    assert f"- {S1B_V1_FREEZE}\n- {M2_V2_DECLARATION}\n" in text
    labels = classify(recs, cfg)
    assert labels[("rho_c[C3]", 0.05, "M2")]["label"] == "weakly_identifiable"
    assert labels[("rho_c[C3]", 0.05, "M2v2")]["label"] == "identifiable"  # labeled separately
    assert labels[("rho_c[C3]", 0.15, "M2v2")]["label"] == "missing"

    v1_cfg = replace(cfg, models_main=("M1", "M2"))
    v1_only = render_report(_records(v1_cfg), v1_cfg)
    assert "`M2` and `M2v2` are not both in models_main of this config." in v1_only


def test_r2_uses_m2_v1_only_and_ignores_m2v2_records():
    """Declaration B: an M2v2 record never stands in for a missing M2 v1 record in the R2 indicator."""
    cfg = _report_cfg()
    recs = _records(cfg)
    for spec in plan_runs(cfg, "main"):
        if spec.kind == "fit":  # well-specified fits that reach the noise level
            recs[spec.run_id]["diagnostics"]["residual_ratio"] = 1.0
    for seed in (0, 1, 2):  # M1 stalls above the noise level on every seed at 0.15 s
        recs[_main_fit("M1", 0.15, seed)]["diagnostics"]["residual_ratio"] = 1.5
    assert p1_evidence(recs, classify(recs, cfg), cfg)["r2"]["status"] == "indicator_present"
    for seed in (0, 1, 2):  # only the M2v2 records of the stalled seeds remain
        del recs[_main_fit("M2", 0.15, seed)]
    r2 = p1_evidence(recs, classify(recs, cfg), cfg)["r2"]
    assert r2["status"] == "incomplete"
    assert {r["model"] for r in r2["rows"]} == {"M1", "M2"}


def test_r1_reads_only_the_v1_models_and_ignores_missing_m2v2_records():
    """Declaration B: M2v2 labels neither enter the R1 entries nor keep R1 incomplete."""
    cfg = _report_cfg()
    assert "M2v2" in cfg.models_main
    recs = _records(cfg)
    for spec in plan_runs(cfg, "main"):
        if spec.kind == "fit":  # well-specified fits that reach the noise level
            recs[spec.run_id]["diagnostics"]["residual_ratio"] = 1.0
    for seed in cfg.seeds_main:  # no M2v2 fit at the realistic noise level has run yet
        del recs[_main_fit("M2v2", 0.15, seed)]
    labels = classify(recs, cfg)
    assert labels[("k_c[C3]", 0.15, "M2v2")]["label"] == "missing"
    r1 = p1_evidence(recs, labels, cfg)["r1"]
    assert r1["status"] == "trigger_condition_not_met"
    assert {e["basis"] for e in r1["entries"]} == {"M1", "M2", "profile"}
    del recs[_main_fit("M2", 0.15, 0)]  # a missing v1 record still keeps R1 incomplete
    assert p1_evidence(recs, classify(recs, cfg), cfg)["r1"]["status"] == "incomplete"


def test_report_quotes_the_declared_readings_and_plan_t5_holds_them_verbatim():
    text = render_report(_records(_report_cfg()), _report_cfg())
    assert len(S1C_SOLVER_DECLARATIONS) == 2
    s1b = (S1B_V1_FREEZE, M2_V2_DECLARATION)
    readings = (*MISSPEC_READING_V1, R2_DECLARATION, *S1C_SOLVER_DECLARATIONS, INTERVAL_V2_DECLARATION, *s1b)
    readings = (*readings, M1_DIAG_DECLARATION)
    for reading in readings:
        assert reading in text
        assert "\u2014" not in reading
    assert f"(PLAN T5, verbatim): {M1_DIAG_DECLARATION} " in text  # P1 section bullet
    assert "`reports/s1_m1_diag.md`" in text
    assert "How this report applies them" not in text  # every reading is declared in PLAN T5
    plan = s1_study.REPO_ROOT / "PLAN.md"
    if not plan.is_file():  # PLAN.md is a local planning document, absent from clean checkouts
        pytest.skip("PLAN.md is not in this checkout")
    plan_text = plan.read_text(encoding="utf-8")
    declared = (*MISSPEC_CRITERIA_V1, *MISSPEC_READING_V1, R2_DECLARATION, *S1C_SOLVER_DECLARATIONS)
    for text_block in (*declared, INTERVAL_V2_DECLARATION, *s1b, M1_DIAG_DECLARATION):
        assert text_block in plan_text, text_block[:60]
    for block in (INTERVAL_V2_DECLARATION, *s1b, M1_DIAG_DECLARATION):
        assert f"  - {block}\n" in plan_text  # its own bullet of the T5 declarations
    # Order in PLAN T5: failed ODE solves, then A (S1b v1 outcome), B (M2 v2), C (interval method v2),
    # D (M1 exploratory diagnostics).
    blocks = (S1C_SOLVER_DECLARATIONS[1], *s1b, INTERVAL_V2_DECLARATION, M1_DIAG_DECLARATION)
    positions = [plan_text.index(b) for b in blocks]
    assert positions == sorted(positions)
    assert (
        "with endpoints by linear interpolation between grid points. This endpoint rule is interval method "
        "v1, superseded by **Interval method v2** below for every reported interval." in plan_text
    )


# --- slow: the real runner end to end ---


@pytest.mark.slow
def test_study_end_to_end_on_a_tiny_config(x64, tmp_path):
    base = smoke_config()
    cfg = replace(
        base,
        n_stints=6,
        m1_train=replace(base.m1_train, n_steps=20),
        m2_train=replace(base.m2_train, n_steps=10),
        m2v2_n_starts=2,
        profile_params=("rho",),
        grid_factors=(0.75, 1.0, 1.33),
        n_starts=1,
        seeds_main=(0,),
        seeds_misspec=(0,),
        misspec_variants=("wrong_phi",),
        out_dir="exp/s1",
        report_path="rep/s1.md",
    )
    summary = run_study(cfg, part="all", root=tmp_path, log=_quiet)
    decision = summary["m2_misspec"]
    assert decision["include"] in (True, False)
    planned = plan_runs(cfg, "main") + plan_runs(cfg, "misspec", bool(decision["include"]))
    records = load_records(tmp_path / cfg.out_dir)
    assert set(records) == {s.run_id for s in planned}
    for run_id, record in records.items():
        assert record["status"] in ("ok", "nonfinite"), (run_id, record.get("error"))
        assert record["wall_time_s"]["total"] > 0
    v2 = records[_main_fit("M2v2", 0.05, 0)]
    assert len(v2["diagnostics"]["starts"]) == 2 and v2["diagnostics"]["selected_start"] in (0, 1)
    text = render_report(records, cfg)
    path = write_report(tmp_path / cfg.report_path, text)
    assert "## S1b recovery" in path.read_text(encoding="utf-8")
    again = run_study(cfg, part="all", root=tmp_path, log=_quiet)
    assert again["run"] == 0 and again["skipped"] == len(planned)
