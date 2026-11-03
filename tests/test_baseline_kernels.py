"""Declared numerical equivalence checks, not another real-data tuning study."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from apexpinn.models.baseline_kernels import CompiledBaseline, fit_kalman, nll_components, training_arrays
from apexpinn.models.baselines import B1, B2, B3, _kalman_weights


@pytest.fixture(autouse=True)
def declared_float64_execution():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def test_compiled_fit_rejects_float32_before_optimization():
    from test_baselines import baseline_fixture

    batch, _, _ = baseline_fixture()
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="x64"):
        fit_kalman(B2(), batch)


def test_compiled_likelihood_matches_original_with_original_lap_gaps():
    import equinox as eqx
    from test_baselines import baseline_fixture

    batch, _, _ = baseline_fixture()
    mask = np.asarray(batch.mask).copy()
    mask[:, [4, 9]] = False
    batch = eqx.tree_at(lambda b: b.mask, batch, jax.numpy.asarray(mask))
    times, values, lengths = training_arrays(batch)
    for variances in ((0.005, 0.0004, 0.01), (0.02, 0.001, 0.2), (1e-8, 1e-8, 1e-6)):
        expected = [
            _kalman_weights(t[:n], y[:n], *variances)[2]
            for t, y, n in zip(times, values, lengths, strict=True)
        ]
        actual = np.asarray(nll_components(jax.numpy.asarray(variances), times, values, lengths))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-7)


def test_compiled_kalman_fit_preserves_fixture_forecasts():
    from test_baselines import baseline_fixture

    batch, history, scenario = baseline_fixture()
    key = jax.random.key(7)
    original = B2().fit(batch, key)
    compiled = fit_kalman(B2(), batch)
    for horizon in (1, 3, 5, 10):
        a, b = original.forecast(history, horizon, scenario), compiled.forecast(history, horizon, scenario)
        np.testing.assert_allclose(a.mean_s, b.mean_s, rtol=1e-8, atol=1e-7)
        np.testing.assert_allclose(a.sigma_s, b.sigma_s, rtol=1e-8, atol=1e-7)


@pytest.mark.parametrize("name", ["B1", "B2", "B3"])
def test_compiled_forecast_and_cache_preserve_values_and_fit_state(name):
    from test_baselines import baseline_fixture

    batch, history, scenario = baseline_fixture()
    key = jax.random.key(8)
    constructors = {"B1": B1(n_estimators=8), "B2": B2(), "B3": B3.init(key, n_steps=20)}
    fitted = constructors[name].fit(batch, key)
    wrapped = CompiledBaseline(fitted)
    before = wrapped.state_digest()
    for horizon in (1, 3, 5, 10, 3):
        expected, actual = (
            fitted.forecast(history, horizon, scenario),
            wrapped.forecast(history, horizon, scenario),
        )
        np.testing.assert_allclose(actual.mean_s, expected.mean_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.sigma_s, expected.sigma_s, rtol=1e-12, atol=1e-12)
        assert wrapped.state_digest() == before
    changed = replace(history, y=history.y + 0.5)
    changed_prediction = wrapped.forecast(changed, 5, scenario)
    np.testing.assert_allclose(
        changed_prediction.mean_s, fitted.forecast(changed, 5, scenario).mean_s, rtol=1e-12, atol=1e-12
    )
