"""Execution-only replay must preserve the original protocol and its exact populations."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from apexpinn.eval.prepared import prepare_evaluation, run_prepared
from apexpinn.eval.protocol import run_protocol


def test_prepared_population_matches_original_records_and_audit():
    from test_eval import ToyForecaster, evaluation_fixture

    folds, cfg = evaluation_fixture()
    data = folds["F1"]
    audit = []
    expected = run_protocol(ToyForecaster(error=0.1), {"F1": data}, cfg, audit=audit)
    plan = prepare_evaluation("F1", data, cfg)
    replay_audit = []
    actual = run_prepared(ToyForecaster(error=0.1), data, cfg, plan, audit=replay_audit)
    for left, right in zip(expected, actual, strict=True):
        a, b = asdict(left), asdict(right)
        b["instrument"].pop("input_execution")
        assert a == b
    for left, right in zip(audit, replay_audit, strict=True):
        a, b = asdict(left), asdict(right)
        b["instrument"].pop("input_execution")
        assert a == b


def test_replay_reuses_readonly_inputs_and_validates_identity(monkeypatch):
    from test_eval import ToyForecaster, evaluation_fixture

    from apexpinn.eval import protocol

    folds, cfg = evaluation_fixture()
    calls = []
    original = protocol.slice_at_origin

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(protocol, "slice_at_origin", capture)
    plan = prepare_evaluation("F1", folds["F1"], cfg)
    count = len(calls)
    assert count > 0
    for _ in range(3):
        run_prepared(ToyForecaster(), folds["F1"], cfg, plan, audit=[])
    assert len(calls) == count
    with pytest.raises(ValueError, match="identity"):
        run_prepared(ToyForecaster(), replace(folds["F1"], gold_hash="different"), cfg, plan, audit=[])
    with pytest.raises(ValueError, match="identity"):
        run_prepared(ToyForecaster(), folds["F1"], replace(cfg, phi=0.03), plan, audit=[])
    changed_stats = replace(folds["F1"].normalization, training_q_mean=0.99)
    with pytest.raises(ValueError, match="identity"):
        run_prepared(ToyForecaster(), replace(folds["F1"], normalization=changed_stats), cfg, plan, audit=[])

    class Mutating(ToyForecaster):
        def forecast(self, history, horizon, scenario):
            history.y[0] = 0.0

    with pytest.raises(ValueError, match="read-only"):
        run_prepared(Mutating(), folds["F1"], cfg, plan, audit=[])


@pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3"])
def test_fitted_baseline_replay_is_bitwise_identical(name):
    import jax
    from test_baselines import baseline_fixture
    from test_eval import evaluation_fixture

    from apexpinn.models.baselines import B0, B1, B2, B3

    folds, cfg = evaluation_fixture(model_name=name)
    batch, _, _ = baseline_fixture()
    key = jax.random.key(2026)
    constructors = {"B0": B0(), "B1": B1(n_estimators=8), "B2": B2(), "B3": B3.init(key, n_steps=20)}
    model = constructors[name].fit(batch, key)
    data = replace(folds["F1"], model_hash=model.state_digest())
    expected = run_protocol(model, {"F1": data}, cfg, audit=[])
    plan = prepare_evaluation("F1", data, cfg)
    actual = run_prepared(model, data, cfg, plan, audit=[])
    assert [r.key for r in expected] == [r.key for r in actual]
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(
            [a.y_pred, a.sigma_s, *a.member_means_s, *a.member_sigma_s],
            [b.y_pred, b.sigma_s, *b.member_means_s, *b.member_sigma_s],
        )
    primary = run_prepared(model, data, cfg, plan, audit=[], horizons=(5,))
    assert [r.key for r in primary] == [r.key for r in expected if r.horizon == 5]
    assert [(r.y_pred, r.sigma_s, r.member_means_s) for r in primary] == [
        (r.y_pred, r.sigma_s, r.member_means_s) for r in expected if r.horizon == 5
    ]


def test_primary_preparation_preserves_declared_population_and_rejects_other_horizons():
    from test_eval import ToyForecaster, evaluation_fixture

    folds, cfg = evaluation_fixture()
    data = folds["F1"]
    plan = prepare_evaluation("F1", data, cfg, horizons=(5,))
    actual = run_prepared(ToyForecaster(error=0.25), data, cfg, plan, audit=[])
    expected = run_protocol(ToyForecaster(error=0.25), {"F1": data}, cfg, audit=[])
    assert [r.key for r in actual] == [r.key for r in expected if r.horizon == 5]
    assert [r.y_pred for r in actual] == [r.y_pred for r in expected if r.horizon == 5]
    with pytest.raises(ValueError, match="horizons"):
        run_prepared(ToyForecaster(), data, cfg, plan, audit=[], horizons=(1,))
