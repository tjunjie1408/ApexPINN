"""Tests for the S1 study runner and report (PLAN T5). Fast tests use stub runners and fabricated
records only: they never call train() or compile Diffrax. The end-to-end test is marked slow."""

import json
from dataclasses import asdict, replace

import jax
import numpy as np
import pytest

from apexpinn.contracts import FOUR_FIELDS, STATUSES, MetricRecord
from apexpinn.eval.s1_report import (
    MISSPEC_CRITERIA_V1,
    MISSPEC_READING_V1,
    P1_DECISION_SENTENCE,
    R2_DECLARATION,
    classify,
    evaluate_misspec,
    load_records,
    p1_evidence,
    render_report,
    unconverged_points,
    write_report,
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
    shard_runs,
    smoke_config,
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
    assert cfg.models_main == ("M1", "M2")
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
    assert len(main) == 30
    assert sum(s.kind == "profile" for s in main) == 10
    fits = [s for s in main if s.kind == "fit"]
    assert len(fits) == 20 and {s.model for s in fits} == {"M1", "M2"}
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
    assert "main_profile_ode_well_specified_noise0.15_seed4" in ids
    assert "misspec_profile_ode_traffic_noise0.05_seed2" in ids
    assert "misspec_fit_M2_wrong_phi_noise0.05_seed1" in ids

    smoke = smoke_config()
    assert len(plan_runs(smoke, "main")) == 3
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
    assert len(planned) == 6
    assert {p.stem for p in out.glob("*.json")} == {s.run_id for s in planned}
    assert sorted(stub.calls) == sorted(s.run_id for s in planned)
    assert (summary["planned"], summary["run"], summary["skipped"], summary["errors"]) == (6, 6, 0, 0)
    assert not list(out.glob("*.tmp"))
    spec = planned[0]
    record = _read(out, spec.run_id)
    assert record["run_id"] == spec.run_id and record["spec"] == asdict(spec)
    assert record["config_hash"] == config_hash(cfg, spec)
    assert record["meta"]["seeds"] == [spec.seed] and record["config_name"] == "smoke"
    assert record["wall_time_s"]["total"] == 1.0

    again = Stub()
    summary = run_study(cfg, part="main", runner=again, root=tmp_path, log=_quiet)
    assert again.calls == []
    assert (summary["run"], summary["skipped"]) == (0, 6)


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
    assert summary["errors"] == 1 and summary["run"] == 6
    failed = _read(out, fail_id)
    assert failed["status"] == "error" and "boom" in failed["error"] and failed["records"] == []
    assert _read(out, nonfinite_id)["status"] == "nonfinite"

    rerun = Stub()
    run_study(cfg, part="main", runner=rerun, root=tmp_path, log=_quiet)
    assert rerun.calls == [fail_id]
    assert _read(out, fail_id)["status"] == "ok"
    assert _read(out, nonfinite_id)["status"] == "nonfinite"


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


def test_report_quotes_the_declared_readings_and_plan_t5_holds_them_verbatim():
    text = render_report(_records(_report_cfg()), _report_cfg())
    for reading in (*MISSPEC_READING_V1, R2_DECLARATION):
        assert reading in text
        assert "\u2014" not in reading
    assert "How this report applies them" not in text  # every reading is declared in PLAN T5
    plan = s1_study.REPO_ROOT / "PLAN.md"
    if not plan.is_file():  # PLAN.md is a local planning document, absent from clean checkouts
        pytest.skip("PLAN.md is not in this checkout")
    plan_text = plan.read_text(encoding="utf-8")
    for text_block in (*MISSPEC_CRITERIA_V1, *MISSPEC_READING_V1, R2_DECLARATION):
        assert text_block in plan_text, text_block[:60]


# --- slow: the real runner end to end ---


@pytest.mark.slow
def test_study_end_to_end_on_a_tiny_config(x64, tmp_path):
    base = smoke_config()
    cfg = replace(
        base,
        n_stints=6,
        m1_train=replace(base.m1_train, n_steps=20),
        m2_train=replace(base.m2_train, n_steps=10),
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
    text = render_report(records, cfg)
    path = write_report(tmp_path / cfg.report_path, text)
    assert "## S1b recovery" in path.read_text(encoding="utf-8")
    again = run_study(cfg, part="all", root=tmp_path, log=_quiet)
    assert again["run"] == 0 and again["skipped"] == len(planned)
