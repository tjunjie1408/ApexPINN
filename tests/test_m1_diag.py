"""Tests for the exploratory M1 diagnostics (PLAN T5, `M1` exploratory diagnostics): the bounded PINN
latent, the variant configs, run_diag with a stub runner, the report and scripts/run_m1_diag.py. Fast:
no train() call."""

import importlib.util
import json
from dataclasses import fields, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexpinn.eval.m1_diag_report import DIAG_BANNER, HEADER, render_m1_diag_report
from apexpinn.eval.s1_report import M1_DIAG_DECLARATION, P1_DECISION_SENTENCE, stalled
from apexpinn.models.pinn import LATENT_MODES, PINN, TAU_SCALE
from apexpinn.models.stints import CTX_DIM
from apexpinn.train.m1_diag import (
    DIAG_OUT_DIR,
    DIAG_REPORT,
    VARIANTS,
    diag_specs,
    run_diag,
    summary_line,
    variant_config,
)
from apexpinn.train.s1_study import RunSpec, config_hash, full_config, gated_names

REPO = Path(__file__).resolve().parents[1]
RUN_M1_DIAG = REPO / "scripts" / "run_m1_diag.py"
RID0 = "main_fit_M1_well_specified_noise0.05_seed0"


def _quiet(_message):
    return None


# --- PINN latent_mode ---


def _scaled(model: PINN, factor: float) -> PINN:
    """model with every network weight and bias multiplied by factor (larger network outputs)."""
    net = jax.tree_util.tree_map(lambda x: x * factor if eqx.is_inexact_array(x) else x, model.net)
    return eqx.tree_at(lambda m: m.net, model, net)


def test_pinn_linear_latent_is_the_v1_ansatz():
    k_net, k_ctx = jax.random.split(jax.random.key(0))
    model = PINN.init(k_net, width=16, depth=2)
    assert model.latent_mode == "linear" and LATENT_MODES == ("linear", "tanh")
    ctx = jax.random.normal(k_ctx, (CTX_DIM,))
    for tau in (0.0, 3.5, 40.0, 500.0):
        t = jnp.asarray(tau)
        out = model.net(jnp.concatenate([jnp.reshape(t / TAU_SCALE, (1,)), ctx]))
        h, w = model.latent(tau, ctx)
        np.testing.assert_array_equal(np.asarray(h), np.asarray(t * out[0]))
        np.testing.assert_array_equal(np.asarray(w), np.asarray(t * jax.nn.softplus(out[1])))


def test_pinn_tanh_latent_is_bounded_with_an_exact_zero_initial_condition():
    k_net, k_ctx = jax.random.split(jax.random.key(1))
    ctx = jax.random.normal(k_ctx, (CTX_DIM,))
    linear = _scaled(PINN.init(k_net, width=16, depth=2), 5.0)
    bounded = _scaled(PINN.init(k_net, width=16, depth=2, latent_mode="tanh"), 5.0)
    assert bounded.latent_mode == "tanh"
    taus = jnp.asarray([0.0, 0.25, 0.5, 5.0, 40.0, 200.0, 1000.0])
    h_lin, w_lin = jax.vmap(lambda t: linear.latent(t, ctx))(taus)
    h, w = jax.vmap(lambda t: bounded.latent(t, ctx))(taus)
    h, h_lin = np.asarray(h), np.asarray(h_lin)
    assert h[0] == 0.0 and float(w[0]) == 0.0  # h(0) = w(0) = 0 exactly
    assert np.all(np.isfinite(h)) and np.all(np.abs(h) <= 1.0)
    assert np.max(np.abs(h_lin)) > 1.0  # the linear latent is unbounded, so the mode is not ignored
    inside = np.abs(h_lin) < 5.0
    assert inside.sum() >= 2 and np.all(np.abs(h[inside]) < 1.0)
    np.testing.assert_allclose(h, np.tanh(h_lin), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(w), np.asarray(w_lin))  # w is the same in both modes


def test_pinn_rejects_an_unknown_latent_mode():
    with pytest.raises(ValueError, match="latent_mode"):
        PINN.init(jax.random.key(0), latent_mode="sigmoid")


# --- variant configs ---


def test_each_variant_changes_only_its_own_field():
    base = full_config()
    changes = {
        "budget_20k": {"m1_train": replace(base.m1_train, n_steps=20000, log_every=1000)},
        "physics_off": {"m1_weights": replace(base.m1_weights, thermal=0.0, wear=0.0, mono=0.0)},
        "physics_x10": {"m1_weights": replace(base.m1_weights, thermal=10.0, wear=10.0)},
        "bounded_h": {"m1_latent": "tanh"},
    }
    assert list(VARIANTS) == list(changes)
    common = {"models_main", "noise_levels", "seeds_main", "out_dir", "report_path", "name"}
    m1 = RunSpec("main", "fit", "M1", "well_specified", 0.05, 0)
    hashes = {config_hash(base, m1)}
    for name, change in changes.items():
        cfg = variant_config(name)
        expected = replace(
            base,
            models_main=("M1",),
            noise_levels=(0.05,),
            seeds_main=(0,),
            out_dir=f"experiments/s1_m1_diag/{name}",
            report_path="reports/s1_m1_diag.md",
            name=f"m1_diag_{name}",
            **change,
        )
        assert cfg == expected
        differing = {f.name for f in fields(cfg) if getattr(cfg, f.name) != getattr(base, f.name)}
        assert differing == common | set(change), name
        hashes.add(config_hash(cfg, m1))
    assert len(hashes) == 1 + len(changes)  # every variant changes the M1 hash, none equals v1
    budget = variant_config("budget_20k").m1_train
    assert (budget.n_steps, budget.log_every, budget.n_warm, budget.n_ramp) == (20000, 1000, 200, 800)
    assert variant_config("physics_x10").m1_weights.mono == base.m1_weights.mono
    with pytest.raises(ValueError, match="variant"):
        variant_config("nope")


def test_each_variant_plans_only_the_m1_fit_per_seed():
    for name in VARIANTS:
        cfg = variant_config(name, (0, 1))
        ids = [s.run_id for s in diag_specs(cfg)]
        assert ids == [RID0, "main_fit_M1_well_specified_noise0.05_seed1"]


def test_run_diag_writes_only_the_variant_folders(tmp_path):
    calls = []

    def stub(cfg, spec):
        calls.append((cfg.name, cfg.m1_latent, spec.run_id))
        return {"status": "ok", "records": []}

    summaries = run_diag(seeds=(0,), root=tmp_path, log=_quiet, runner=stub)
    assert list(summaries) == list(VARIANTS)
    assert sorted(c[0] for c in calls) == sorted(f"m1_diag_{n}" for n in VARIANTS)
    assert {c[2] for c in calls} == {RID0}
    assert ("m1_diag_bounded_h", "tanh", RID0) in calls
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json"))
    assert written == sorted(f"{DIAG_OUT_DIR}/{n}/{RID0}.json" for n in VARIANTS)
    for name in VARIANTS:
        record = json.loads((tmp_path / DIAG_OUT_DIR / name / f"{RID0}.json").read_text(encoding="utf-8"))
        assert record["config_name"] == f"m1_diag_{name}"
        assert summaries[name]["planned"] == summaries[name]["run"] == 1
    again = run_diag(["bounded_h"], (0,), root=tmp_path, log=_quiet, runner=stub)
    assert again["bounded_h"]["skipped"] == 1 and again["bounded_h"]["run"] == 0
    with pytest.raises(ValueError, match="variants"):
        run_diag(["nope"], root=tmp_path, log=_quiet, runner=stub)


def test_run_diag_refuses_one_variant_and_still_runs_and_logs_the_others(tmp_path):
    def stub(cfg, spec):
        return {"status": "ok", "records": []}

    run_diag(["physics_off"], (0,), root=tmp_path, log=_quiet, runner=stub)
    path = tmp_path / DIAG_OUT_DIR / "physics_off" / f"{RID0}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, "config_hash": "stale"}), encoding="utf-8")
    lines = []
    summaries = run_diag(seeds=(0,), root=tmp_path, log=lines.append, runner=stub)
    assert list(summaries) == list(VARIANTS)
    assert "config changed" in summaries["physics_off"]["refused"]
    for name in ("budget_20k", "physics_x10", "bounded_h"):  # before and after the refused variant
        assert summaries[name]["run"] == 1 and "refused" not in summaries[name]
        assert summary_line(name, summaries[name]) in lines
    stop = summary_line("physics_off", summaries["physics_off"])
    header = f"variant physics_x10: {VARIANTS['physics_x10'][0]}"
    assert stop in lines and lines.index(stop) < lines.index(header)  # logged when it stopped
    assert json.loads(path.read_text(encoding="utf-8"))["config_hash"] == "stale"  # not overwritten


# --- report ---


def _fit(cfg, k_err, ratio, change):
    rel = {n: (k_err if n.startswith("k_c") else 0.01) for n in gated_names(cfg)}
    return {
        "status": "ok",
        "spec": {"seed": 0},
        "rel_error": rel,
        "diagnostics": {"residual_ratio": ratio, "thermal_residual_rms": 0.25},
        "convergence": {
            "objective_rel_change_last20": change,
            "normalized": {"data": [1.0, 0.3], "thermal": [1.0, 0.04]},
        },
        "wall_time_s": {"fit": 12.5},
    }


def _cells(text, label):
    line = next(row for row in text.splitlines() if row.startswith(f"| {label} |"))
    return [c.strip() for c in line.strip().strip("|").split("|")]


def test_render_m1_diag_report_shows_banner_declaration_missing_rows_and_the_r2_stall_flag():
    cfg = full_config()
    n = len(gated_names(cfg))
    v1 = {RID0: _fit(cfg, k_err=-0.95, ratio=1.5, change=-0.001)}  # stalls: change > -1%, ratio >= 1.2
    diag = {
        "budget_20k": {RID0: _fit(cfg, k_err=-0.10, ratio=1.0, change=-0.05)},  # no stall, k within 25%
        "physics_x10": {RID0: {"status": "error", "error": "boom"}},
        "bounded_h": {RID0: _fit(cfg, k_err=0.30, ratio=1.2, change=-0.009)},  # stalls at both boundaries
    }  # physics_off has no record
    text = render_m1_diag_report(diag, v1, cfg)
    assert DIAG_BANNER == (
        "Exploratory M1 diagnostics. Not S1b evidence; no S1b result, label or method changes from these "
        "runs."
    )
    assert f"> {DIAG_BANNER}" in text
    assert f"- {M1_DIAG_DECLARATION}" in text
    assert P1_DECISION_SENTENCE in text
    assert "\u2014" not in text
    stall = HEADER.index("R2 stall flag")
    base = _cells(text, "baseline v1")
    assert len(base) == len(HEADER)
    assert base[stall] == "yes" and stalled(v1[RID0]) is True
    assert base[HEADER.index("Gated recovered")] == f"{n // 2} of {n}"
    assert "-95.0%" in base[HEADER.index("`k_c` rel. error")]
    assert base[HEADER.index("Source")] == f"`experiments/s1/{RID0}.json`"
    budget = _cells(text, "budget_20k")
    assert budget[stall] == "no" and budget[HEADER.index("Gated recovered")] == f"{n} of {n}"
    assert budget[HEADER.index("Steps")] == "20000"
    assert budget[HEADER.index("Source")] == f"`{DIAG_OUT_DIR}/budget_20k/{RID0}.json`"
    bounded = _cells(text, "bounded_h")
    assert bounded[stall] == "yes" and bounded[HEADER.index("Latent")] == "tanh"
    missing = _cells(text, "physics_off")
    assert missing[HEADER.index("Status")] == "missing" and missing[stall] == "missing"
    assert missing[HEADER.index("Weights")] == "thermal 0, wear 0, mono 0"
    errored = _cells(text, "physics_x10")
    assert errored[HEADER.index("Status")] == "error" and errored[stall] == "n/a"
    assert "- Seed 0, `baseline v1`: residual ratio below 1.2: no; any `k_c` within 25%: no;" in text
    assert "- Seed 0, `budget_20k`: residual ratio below 1.2: yes; any `k_c` within 25%: yes;" in text
    assert "- Seed 0, `physics_off`: residual ratio below 1.2: missing;" in text


def _load_run_m1_diag():
    """Load the script, then restore x64: its module-level update must not leak into later tests."""
    previous = jax.config.x64_enabled
    try:
        spec = importlib.util.spec_from_file_location("run_m1_diag", RUN_M1_DIAG)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        jax.config.update("jax_enable_x64", previous)
    return module


def test_run_m1_diag_report_only_writes_the_report_on_an_empty_root(tmp_path):
    script = _load_run_m1_diag()
    assert script.main(["--report-only", "--root", str(tmp_path)]) == 0
    text = (tmp_path / DIAG_REPORT).read_text(encoding="utf-8")
    assert DIAG_BANNER in text and M1_DIAG_DECLARATION in text
    assert _cells(text, "baseline v1")[HEADER.index("Status")] == "missing"
    assert not (tmp_path / "experiments").exists()


def test_run_m1_diag_exit_codes(monkeypatch, tmp_path):
    script = _load_run_m1_diag()
    ok = {"planned": 1, "run": 1, "skipped": 0, "errors": 0, "wall_time_s": 0.1}
    monkeypatch.setattr(script, "run_diag", lambda *a, **k: {"bounded_h": ok})
    assert script.main(["--root", str(tmp_path)]) == 0
    monkeypatch.setattr(script, "run_diag", lambda *a, **k: {"bounded_h": {**ok, "errors": 1}})
    assert script.main(["--root", str(tmp_path)]) == 1
    refused = {"bounded_h": {**ok, "errors": 1}, "physics_off": {"refused": "config changed"}}
    monkeypatch.setattr(script, "run_diag", lambda *a, **k: refused)
    assert script.main(["--root", str(tmp_path)]) == 2

    def refuse(*args, **kwargs):
        raise ValueError("config changed")

    monkeypatch.setattr(script, "run_diag", refuse)
    assert script.main(["--root", str(tmp_path)]) == 2
    assert (tmp_path / DIAG_REPORT).is_file()  # the report is rebuilt on every call
