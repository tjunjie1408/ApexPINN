"""Compiled execution of the declared baselines; model equations and optimizer settings stay unchanged."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from apexpinn.contracts import PredictionOutput
from apexpinn.models.baselines import B1, B2, B3, _anchored, _mlp_features, _rows, _tabular_features
from apexpinn.models.stints import Scenario, StintBatch, StintHistory, validate_horizon


def training_arrays(train: StintBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau, y, groups, _, _, _ = _rows(train)
    lengths = np.bincount(groups)
    width = int(lengths.max())
    times, values = np.zeros((len(lengths), width)), np.zeros((len(lengths), width))
    offset = 0
    for i, size in enumerate(lengths):
        times[i, :size], values[i, :size] = tau[offset : offset + size], y[offset : offset + size]
        offset += size
    return times, values, lengths


def _transition(distance, variances):
    level, slope, _ = variances
    d = distance
    one, zero = jnp.ones_like(d), jnp.zeros_like(d)
    transition = jnp.stack((jnp.stack((one, d), -1), jnp.stack((zero, one), -1)), -2)
    q00 = d * level + slope * d * (d - 1) * (2 * d - 1) / 6
    q01 = slope * d * (d - 1) / 2
    q11 = d * slope
    covariance = jnp.stack((jnp.stack((q00, q01), -1), jnp.stack((q01, q11), -1)), -2)
    return transition, covariance


@eqx.filter_jit
def nll_components(variances, times, values, lengths):
    """Original first-two-observation weights/Joseph filter, scanned over padded independent stints."""
    n, width = values.shape
    noise = variances[2]
    d = times[:, 1] - times[:, 0]
    _, process = _transition(d, variances)
    weights = jnp.zeros((n, 2, width), dtype=values.dtype)
    weights = weights.at[:, 0, 1].set(1).at[:, 1, 0].set(-1 / d).at[:, 1, 1].set(1 / d)
    p00, p01 = jnp.full_like(d, noise), noise / d
    p11 = (2 * noise + process[:, 0, 0]) / d**2 + process[:, 1, 1] - 2 * process[:, 0, 1] / d
    covariance = jnp.stack((jnp.stack((p00, p01), -1), jnp.stack((p01, p11), -1)), -2)

    def step(carry, index):
        weight, cov, likelihood = carry
        active = index < lengths
        distance = jnp.where(active, times[:, index] - times[:, index - 1], 1)
        transition, process = _transition(distance, variances)
        weight = transition @ weight
        cov = (transition @ cov) @ jnp.swapaxes(transition, -1, -2) + process
        innovation = values[:, index] - jnp.sum(weight[:, 0] * values, axis=1)
        variance = cov[:, 0, 0] + noise
        gain = cov[:, :, 0] / variance[:, None]
        contribution = 0.5 * (jnp.log(variance) + innovation**2 / variance)
        likelihood += jnp.where(active, contribution, 0)
        basis = (jnp.arange(width) == index).astype(values.dtype)
        weight += gain[:, :, None] * (basis - weight[:, 0])[:, None, :]
        correction = jnp.eye(2) - gain[:, :, None] * jnp.array([[1.0, 0.0]])
        cov = (correction @ cov) @ jnp.swapaxes(correction, -1, -2)
        cov += noise * gain[:, :, None] * gain[:, None, :]
        return (weight, cov, likelihood), None

    (_, _, result), _ = jax.lax.scan(
        step, (weights, covariance, jnp.zeros(n, dtype=values.dtype)), jnp.arange(2, width)
    )
    return result


def fit_kalman(model: B2, train: StintBatch) -> B2:
    if not jax.config.x64_enabled:
        raise ValueError("compiled execution requires jax_enable_x64=True before model use")
    times, values, lengths = (jnp.asarray(x) for x in training_arrays(train))

    def objective(log_variances):
        # Preserve the original host-side stint summation order and SciPy finite-difference optimizer.
        components = np.asarray(nll_components(jnp.asarray(np.exp(log_variances)), times, values, lengths))
        return sum(components.tolist())

    start = np.log(np.maximum([model.level_variance, model.slope_variance, model.observation_variance], 1e-8))
    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=[(np.log(1e-10), np.log(100))] * 3,
        options={"maxiter": model.maxiter},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"Kalman training likelihood did not converge: {result.message}")
    level, slope, noise = map(float, np.exp(result.x))
    return replace(
        model,
        level_variance=level,
        slope_variance=slope,
        observation_variance=noise,
        parameters_source="training_innovation_likelihood",
    )


@eqx.filter_jit
def _mlp_values(net, features):
    return jax.vmap(net)(features)[:, 0]


@dataclass(frozen=True)
class CompiledBaseline:
    """Own execution caches separately from immutable fitted parameters and their state digest."""

    base: B1 | B2 | B3
    context_features: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    predictions: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def fit(self, train: StintBatch, key: jax.Array) -> CompiledBaseline:
        fitted = fit_kalman(self.base, train) if isinstance(self.base, B2) else self.base.fit(train, key)
        return CompiledBaseline(fitted)

    def state_digest(self) -> str:
        return self.base.state_digest()

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        if not jax.config.x64_enabled:
            raise ValueError("compiled execution requires jax_enable_x64=True before model use")
        horizon = validate_horizon(horizon)
        if isinstance(self.base, B2):
            key = (
                history.stint_id,
                history.t0,
                history.tau.tobytes(),
                history.y.tobytes(),
                history.q.tobytes(),
                history.ctx,
                history.race_lap_t0,
                scenario,
            )
            if key not in self.predictions or len(self.predictions[key].mean_s) < horizon:
                prediction = self.base.forecast(history, max(10, horizon), scenario)
                for value in (
                    prediction.mean_s,
                    prediction.member_means_s,
                    prediction.sigma_s,
                    prediction.member_sigma_s,
                ):
                    if value is not None:
                        value.setflags(write=False)
                self.predictions[key] = prediction
            p = self.predictions[key]
            return PredictionOutput(
                p.mean_s[:horizon],
                p.member_means_s[:, :horizon],
                p.sigma_s[:horizon],
                None,
                None,
                member_sigma_s=p.member_sigma_s[:, :horizon],
            )
        key = (history.ctx, jax.config.x64_enabled)
        if key not in self.context_features:
            self.context_features[key] = np.asarray(history.ctx.features())
        tau = np.concatenate((history.tau, np.arange(history.t0 + 1, history.t0 + horizon + 1)))
        ctx = np.tile(self.context_features[key], (len(tau), 1))
        if isinstance(self.base, B1):
            if self.base.booster is None:
                return self.base.forecast(history, horizon, scenario)
            q = np.concatenate((history.q, np.full(horizon, scenario.q_bar)))
            shape = self.base.booster.predict(
                _tabular_features(tau, ctx, q, np.full(len(tau), history.ctx.compound_idx))
            )
        else:
            shape = np.asarray(_mlp_values(self.base.net, jnp.asarray(_mlp_features(tau, ctx))))
        return _anchored(
            history, horizon, shape[: len(history.tau)], shape[len(history.tau) :], self.base.noise_s
        )
