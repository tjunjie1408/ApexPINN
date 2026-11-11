"""Pure arithmetic/provenance and mocked no-fit checks. Never execute a calibration fit."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from apexpinn.eval import calibration as cal


def row(fold="F5", role="calibration"):
    means = [-1.0, 0.0, 1.0, 2.0, 3.0]
    scales = [0.5] * 5
    return {
        "model": "M2/raw",
        "fold": fold,
        "event_id": "2024_example",
        "stint_id": "s",
        "t0": 3,
        "horizon": 5,
        "phi": 0.06,
        "y_true": 1.2,
        "y_pred": 1.0,
        "sigma_s": float(np.sqrt(np.mean(np.square(scales)) + np.var(means))),
        "member_means_s": means,
        "member_sigma_s": scales,
        "conditions": {"data_role": role, "outer_fold": "F3", "model": "M2/raw"},
        "instrument": {"model_hash": "five-fixed-models", "normalization_numeric_sha256": "norm"},
    }


def test_three_way_split_and_role_rejection():
    d = SimpleNamespace(folds={f"F{i}": [f"c{i}"] for i in range(1, 7)})
    p = cal.split_plan(d, "F3")
    assert p["validation_fold"] == "F4" and p["calibration_fold"] == "F5"
    assert p["tuning_circuits"] == ["c1", "c2", "c6"]
    assert p["final_circuits"] == ["c1", "c2", "c4", "c6"]
    cal.validate_calibration_rows([row()], p, horizons=(5,), minimum_events=1)
    with pytest.raises(ValueError, match="calibration"):
        cal.validate_calibration_rows([row("F3", "outer_test")], p, horizons=(5,), minimum_events=1)
    with pytest.raises(ValueError, match="horizon"):
        cal.validate_calibration_rows([row()], p, horizons=(1, 5), minimum_events=1)


def test_event_weighting_and_weighted_quantile_are_explicit_arithmetic():
    np.testing.assert_allclose(cal.event_weights(["a", "a", "b"]), [0.25, 0.25, 0.5])
    assert cal.weighted_quantile([1, 5, 9], [0.9, 0.05, 0.05], 0.9) == 1
    with pytest.raises(ValueError):
        cal.weighted_quantile([1, 2], [1, -1], 0.9)


def test_fixed_transform_preserves_mean_exact_mixture_and_inputs():
    r = row("F3", "outer_test")
    original = copy.deepcopy(r)
    result = cal.rescale_row(r, 2.0, recipe_sha256="recipe")
    assert result["y_pred"] == original["y_pred"] and result["y_true"] == original["y_true"]
    assert result["sigma_s"] ** 2 == pytest.approx(4 * original["sigma_s"] ** 2)
    assert np.mean(result["member_means_s"]) == pytest.approx(original["y_pred"])
    assert result["sigma_s"] ** 2 == pytest.approx(
        np.mean(np.square(result["member_sigma_s"])) + np.var(result["member_means_s"])
    )
    assert r == original
    with pytest.raises(ValueError, match="positive"):
        cal.rescale_row(r, 0, recipe_sha256="recipe")


def test_calibrator_fitting_stops_at_remote_guard(monkeypatch):
    def blocked():
        raise RuntimeError("Colab only")

    monkeypatch.setattr(cal, "_remote_guard", blocked)
    with pytest.raises(RuntimeError, match="Colab only"):
        cal.fit_calibrator([], {}, {})


def test_training_consumers_cannot_receive_calibration_or_test_labels():
    import pandas as pd

    from apexpinn.train.calibration_study import select_rows

    frame = pd.DataFrame(
        {
            "dataset_split": ["development"] * 6,
            "circuit_id": [f"c{i}" for i in range(1, 7)],
            "label": [1, 2, 3, 4, 5, 6],
        }
    )
    plan = cal.split_plan(SimpleNamespace(folds={f"F{i}": [f"c{i}"] for i in range(1, 7)}), "F3")
    tuning = select_rows(frame, plan["tuning_circuits"] + plan["validation_circuits"])
    final = select_rows(frame, plan["final_circuits"])
    poisoned = frame.copy()
    poisoned.loc[poisoned.circuit_id.isin(["c3", "c5"]), "label"] = 999999
    pd.testing.assert_frame_equal(
        tuning, select_rows(poisoned, plan["tuning_circuits"] + plan["validation_circuits"])
    )
    pd.testing.assert_frame_equal(final, select_rows(poisoned, plan["final_circuits"]))
    frame.loc[0, "dataset_split"] = "holdout"
    with pytest.raises(ValueError, match="holdout"):
        select_rows(frame, plan["final_circuits"])


def test_prediction_cache_reuses_verified_outputs_and_refuses_corruption(tmp_path):
    from apexpinn.train.calibration_study import _prediction_cache

    folder = tmp_path / "predictions"
    expected = [{"prediction": 1.0}]
    assert _prediction_cache(folder, {"source": "fixed"}, lambda: (expected, [])) == expected
    assert (
        _prediction_cache(folder, {"source": "fixed"}, lambda: pytest.fail("prediction replay")) == expected
    )
    (folder / "forecasts.jsonl").write_text('{"prediction": 99}\n')
    with pytest.raises(ValueError, match="changed"):
        _prediction_cache(folder, {"source": "fixed"}, lambda: pytest.fail("corrupt cache replay"))


def test_new_runner_guard_precedes_model_io(tmp_path, monkeypatch):
    from apexpinn.train import calibration_study as study

    def blocked():
        raise RuntimeError("Colab only")

    monkeypatch.setattr(study, "_remote_guard", blocked)
    with pytest.raises(RuntimeError, match="Colab only"):
        study.run_clean(
            tmp_path / "method", tmp_path / "config", tmp_path / "gold", tmp_path / "out", "F3", "mid"
        )
    assert not (tmp_path / "out").exists()


def test_recipe_rejects_a_different_ensemble_or_normalization():
    r = row("F3", "outer_test")
    recipe = {
        "plan": {"outer_fold": "F3"},
        "model_hash": r["instrument"]["model_hash"],
        "normalization_numeric_sha256": "norm",
        "phi": 0.06,
        "factors": {"5": 0.5},
    }
    assert cal.apply_calibrator([r], recipe)[0]["sigma_s"] == pytest.approx(r["sigma_s"] / 2)
    for field in ("model_hash", "normalization_numeric_sha256"):
        altered = copy.deepcopy(r)
        altered["instrument"][field] = "changed"
        with pytest.raises(ValueError, match="bound"):
            cal.apply_calibrator([altered], recipe)


def test_mocked_twenty_trial_budget_keeps_failures_and_reuses_finished_slots(tmp_path, monkeypatch):
    from dataclasses import dataclass

    import apexpinn.data.gold as gold
    import apexpinn.eval.prepared as prepared
    import apexpinn.models.ensemble as ensemble
    import apexpinn.train.baseline_study as baseline
    import apexpinn.train.physics_study as physics
    from apexpinn.cloud.artifacts import digest, write_json
    from apexpinn.train.calibration_study import _run_trials

    @dataclass
    class FakeFold:
        frame: object
        stats: object
        source: str
        model_hash: str
        inner_validation_fold: str

    import apexpinn.eval.protocol as protocol

    monkeypatch.setattr(protocol, "EvaluationFold", FakeFold)
    monkeypatch.setattr(protocol, "ProtocolConfig", lambda *args: None)
    monkeypatch.setattr(gold, "build_gold", lambda *args: "mock frame")
    monkeypatch.setattr(baseline, "frame_digest", lambda *args: "fixed")
    monkeypatch.setattr(prepared, "prepare_evaluation", lambda *args, **kwargs: "mock plan")
    monkeypatch.setattr(physics, "configurations", lambda *args: [{"id": i} for i in range(20)])
    calls = []

    def mock_fit(base, name, params, variant, seed, batch, folder, run_id):
        calls.append(params["id"])
        if params["id"] == 0:
            raise ValueError("mocked failed budget slot")
        folder.mkdir(parents=True)
        (folder / "model.eqx").write_bytes(b"mock model, no execution")
        fit = {"model_sha256": digest(folder / "model.eqx"), "elapsed_s": 0.0}
        write_json(folder / "fit.json", fit)
        return SimpleNamespace(state_digest=lambda: "mock model"), fit

    monkeypatch.setattr(physics, "fit_once", mock_fit)
    monkeypatch.setattr(ensemble, "PhysicsMember", lambda model: model)
    monkeypatch.setattr(
        prepared, "run_prepared", lambda *args, **kwargs: [SimpleNamespace(y_true=1, y_pred=2)]
    )
    args = (
        {"selection": "minimum_inner_h5_RMSE_then_trial_index"},
        {},
        None,
        None,
        SimpleNamespace(phi=0.06),
        None,
        {"outer_fold": "F3", "validation_fold": "F4"},
        tmp_path,
        {"mock": True},
        0,
    )
    selected, trials = _run_trials(*args)
    assert calls == list(range(20)) and selected == {"id": 1}
    assert len(trials) == 20 and trials[0]["status"] == "failed"
    calls.clear()
    assert _run_trials(*args)[0] == selected
    assert calls == []


def test_saved_score_report_recovers_without_model_imports(tmp_path, monkeypatch):
    import builtins
    import json

    from apexpinn.eval.calibration_report import render_report

    original_import = builtins.__import__

    def forbid_models(name, *args, **kwargs):
        if name == "jax" or name.startswith(("jax.", "apexpinn.models", "apexpinn.train")):
            pytest.fail("report recovery imported model execution")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_models)
    monkeypatch.setenv("MPLBACKEND", "module://unavailable_colab_inline_backend")
    metrics = [
        {
            "conditions": {"model": model, "horizon_actual_laps": 5},
            "quantity": {"metric": metric},
            "population": {"n_forecasts": 4},
            "value": value,
        }
        for model in ("M2/raw", "M2/calibrated")
        for metric, value in (
            ("rmse", 1.0),
            ("crps", 0.7),
            ("coverage_50", 0.5),
            ("coverage_90", 0.75),
            ("width_90", 3.0),
        )
    ]
    saved = {
        "schema": "apexpinn.calibration_evaluation/v1",
        "status": "EVALUATION_COMPLETE",
        "files_sha256": {},
        "recipe": {"factors": {"5": 1.2}, "populations": {"5": {"origins": 4, "events": ["fixture"]}}},
        "plan": {"outer_fold": "F3", "validation_fold": "F4", "calibration_fold": "F5"},
        "metrics": metrics,
        "circuit_metrics": [{"circuit": "fixture", **m} for m in metrics],
        "missing_test_circuits": [],
        "limits": "Fixture scores only; no model execution.",
    }
    (tmp_path / "evaluation.json").write_text(json.dumps(saved))
    assert render_report(tmp_path)["status"] == "REPORT_COMPLETE"
    assert (tmp_path / "calibration_intervals.png").stat().st_size > 0
    assert "Fixture scores only" in (tmp_path / "calibration_report.md").read_text()
