"""B0-B3: non-physics forecasts with shared history-only offset anchoring (PLAN T12)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from apexpinn.contracts import PredictionOutput
from apexpinn.models.pinn import TAU_SCALE
from apexpinn.models.stints import (
    AGE_SCALE_LAPS,
    CTX_DIM,
    TRACK_TEMP_REF_C,
    TRACK_TEMP_SCALE_C,
    Scenario,
    StintBatch,
    StintHistory,
    anchor_offset,
    build_prediction,
    validate_horizon,
)
from apexpinn.physics.residuals import N_COMPOUNDS

SIGMA_FLOOR = 1e-6


def _digest(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(str((part.dtype.str, part.shape)).encode())
            digest.update(part.tobytes())
        else:
            digest.update(json.dumps(part, sort_keys=True).encode())
    return digest.hexdigest()


def _rows(batch: StintBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(batch.mask, dtype=bool)
    if np.any(mask.sum(axis=1) < 3):
        raise ValueError("baseline training requires at least three scored laps per stint")
    tau = np.asarray(batch.tau, dtype=float)
    if not np.array_equal(tau, np.arange(1, len(tau) + 1)):
        raise ValueError("StintBatch must retain original integer lap counters 1..K")
    groups, columns = np.nonzero(mask)
    ctx = np.asarray(batch.ctx, dtype=float)[groups]
    targets = np.asarray(batch.y, dtype=float)[groups, columns]
    q = np.asarray(batch.q, dtype=float)[groups, columns + 1]
    compounds = np.asarray(batch.compound_idx, dtype=int)[groups]
    if not all(np.isfinite(v).all() for v in (ctx, targets, q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("training rows/context/q must be finite and q normalized to [0,1]")
    if np.any((compounds < 0) | (compounds >= N_COMPOUNDS)):
        raise ValueError("unsupported training compound")
    return tau[columns], targets, groups, ctx, q, compounds


def _center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return values - np.bincount(groups, weights=values)[groups] / np.bincount(groups)[groups]


def _noise(y: np.ndarray, shape: np.ndarray, groups: np.ndarray) -> float:
    return max(float(np.sqrt(np.mean(_center(y - shape, groups) ** 2))), SIGMA_FLOOR)


def _anchored(
    history: StintHistory, horizon: int, historical: np.ndarray, future: np.ndarray, noise_s: float
) -> PredictionOutput:
    validate_horizon(horizon)
    return build_prediction(historical[None, :], future[None, :], history.y, noise_s)


@dataclass(frozen=True)
class B0:
    """Within-stint compound linear slopes, with pooled-training fallback for unseen C1-C5."""

    slopes: tuple[float, ...] = (0.0,) * N_COMPOUNDS
    seen_compounds: tuple[bool, ...] = (False,) * N_COMPOUNDS
    noise_s: float = 0.1
    fitted: bool = False

    def fit(self, train: StintBatch, key: jax.Array) -> B0:
        tau, y, groups, _, _, compounds = _rows(train)
        x, z = _center(tau, groups), _center(y, groups)
        denominator = float(x @ x)
        if denominator <= 0:
            raise ValueError("no within-stint lap variation")
        pooled = float(x @ z / denominator)
        slopes = []
        seen = []
        for c in range(N_COMPOUNDS):
            take = compounds == c
            d = float(x[take] @ x[take])
            slopes.append(float(x[take] @ z[take] / d) if d > 0 else pooled)
            seen.append(d > 0)
        shape = np.array(slopes)[compounds] * tau
        return B0(tuple(slopes), tuple(seen), _noise(y, shape, groups), True)

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        horizon = validate_horizon(horizon)
        if not self.fitted:
            raise ValueError("B0 must be fitted on training stints")
        slope = self.slopes[history.ctx.compound_idx]
        return _anchored(
            history,
            horizon,
            slope * history.tau,
            slope * np.arange(history.t0 + 1, history.t0 + horizon + 1),
            self.noise_s,
        )

    def state_digest(self) -> str:
        return _digest("B0", asdict(self))


def _tabular_features(tau: np.ndarray, ctx: np.ndarray, q: np.ndarray, compounds: np.ndarray) -> np.ndarray:
    age = ctx[:, N_COMPOUNDS] * AGE_SCALE_LAPS + tau - 1
    temperature = ctx[:, N_COMPOUNDS + 2] * TRACK_TEMP_SCALE_C + TRACK_TEMP_REF_C
    return np.column_stack((tau, age, compounds, temperature, q))


@dataclass(frozen=True)
class B1:
    """LightGBM with nuisance stint offsets profiled out by centered residual gradients."""

    n_estimators: int = 100
    num_leaves: int = 15
    learning_rate: float = 0.05
    min_data_in_leaf: int = 5
    booster: Any = None
    noise_s: float = 0.1

    def fit(self, train: StintBatch, key: jax.Array) -> B1:
        import lightgbm as lgb

        tau, y, groups, ctx, q, c = _rows(train)
        features = _tabular_features(tau, ctx, q, c)

        def objective(prediction, dataset):
            residual = _center(np.asarray(prediction) - y, groups)
            hessian = 1 - 1 / np.bincount(groups)[groups]
            return residual, hessian

        seed = int(np.asarray(jax.random.key_data(key), dtype=np.uint32).sum(dtype=np.uint64) % (2**31 - 1))
        params = {
            "objective": objective,
            "metric": "None",
            "verbosity": -1,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "seed": seed,
            "data_random_seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_pre_filter": False,
        }
        dataset = lgb.Dataset(features, label=y, categorical_feature=[2])
        booster = lgb.train(params, dataset, num_boost_round=self.n_estimators)
        return replace(self, booster=booster, noise_s=_noise(y, booster.predict(features), groups))

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        horizon = validate_horizon(horizon)
        if self.booster is None:
            raise ValueError("B1 must be fitted on training stints")
        future_tau = np.arange(history.t0 + 1, history.t0 + horizon + 1, dtype=float)
        tau = np.concatenate((history.tau, future_tau))
        q = np.concatenate((history.q, np.full(horizon, scenario.q_bar)))
        ctx = np.tile(np.asarray(history.ctx.features()), (len(tau), 1))
        values = self.booster.predict(
            _tabular_features(tau, ctx, q, np.full(len(tau), history.ctx.compound_idx))
        )
        return _anchored(
            history, horizon, values[: len(history.tau)], values[len(history.tau) :], self.noise_s
        )

    def state_digest(self) -> str:
        if self.booster is None:
            raise ValueError("B1 is not fitted")
        return _digest("B1", self.booster.model_to_string(), self.noise_s)


def _transition(distance: int, level: float, slope: float) -> tuple[np.ndarray, np.ndarray]:
    d = float(distance)
    transition = np.array([[1.0, d], [0.0, 1.0]])
    covariance = np.array(
        [
            [d * level + slope * d * (d - 1) * (2 * d - 1) / 6, slope * d * (d - 1) / 2],
            [slope * d * (d - 1) / 2, d * slope],
        ]
    )
    return transition, covariance


def _kalman_weights(
    tau: np.ndarray, y: np.ndarray, level: float, slope: float, noise: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Diffuse first-two-observation initialization; return filtered level weights and last-state weights."""
    n = len(tau)
    if n < 2 or np.any(np.diff(tau) <= 0) or np.any(tau % 1):
        raise ValueError("Kalman history requires at least two increasing original integer laps")
    distance = int(tau[1] - tau[0])
    _, process = _transition(distance, level, slope)
    weights = np.zeros((2, n))
    weights[0, 1] = 1
    weights[1, 0], weights[1, 1] = -1 / distance, 1 / distance
    covariance = np.array(
        [
            [noise, noise / distance],
            [
                noise / distance,
                (2 * noise + process[0, 0]) / distance**2 + process[1, 1] - 2 * process[0, 1] / distance,
            ],
        ]
    )
    filtered = np.zeros((n, n))
    filtered[0, 0] = 1
    filtered[1] = weights[0]
    likelihood = 0.0
    for i in range(2, n):
        transition, process = _transition(int(tau[i] - tau[i - 1]), level, slope)
        weights = transition @ weights
        covariance = transition @ covariance @ transition.T + process
        innovation = float(y[i] - weights[0] @ y)
        variance = float(covariance[0, 0] + noise)
        if variance > 0:
            gain = covariance[:, 0] / variance
            likelihood += 0.5 * (np.log(variance) + innovation**2 / variance)
            basis = np.zeros(n)
            basis[i] = 1
            weights += gain[:, None] * (basis - weights[0])[None, :]
            correction = np.eye(2) - gain[:, None] * np.array([[1.0, 0.0]])
            covariance = correction @ covariance @ correction.T + noise * np.outer(gain, gain)
        elif abs(innovation) > 1e-9:
            raise ValueError("observations incompatible with the specified noiseless trend")
        filtered[i] = weights[0]
    return filtered, weights, likelihood


def _level_covariance(left: np.ndarray, right: np.ndarray, level: float, slope: float) -> np.ndarray:
    t, u = left[:, None], right[None, :]
    m = np.minimum(t, u)
    sum_j = m * (m - 1) / 2
    sum_j2 = m * (m - 1) * (2 * m - 1) / 6
    return level * m + slope * (m * (t - 1) * (u - 1) - (t + u - 2) * sum_j + sum_j2)


@dataclass(frozen=True)
class B2:
    """Discrete trend Kalman model with prediction and anchor covariance propagated together."""

    level_variance: float = 0.005
    slope_variance: float = 0.0004
    observation_variance: float = 0.01
    maxiter: int = 100
    parameters_source: str = "specified"

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(v) and v >= 0
            for v in (self.level_variance, self.slope_variance, self.observation_variance)
        ):
            raise ValueError("Kalman variances must be finite and nonnegative")

    def fit(self, train: StintBatch, key: jax.Array) -> B2:
        tau, y, groups, _, _, _ = _rows(train)
        histories = [(tau[groups == i], y[groups == i]) for i in np.unique(groups)]

        def objective(log_variances):
            level, slope, noise = np.exp(log_variances)
            return sum(_kalman_weights(t, z, level, slope, noise)[2] for t, z in histories)

        start = np.log(
            np.maximum([self.level_variance, self.slope_variance, self.observation_variance], 1e-8)
        )
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(np.log(1e-10), np.log(100))] * 3,
            options={"maxiter": self.maxiter},
        )
        if not result.success or not np.isfinite(result.fun):
            raise ValueError(f"Kalman training likelihood did not converge: {result.message}")
        level, slope, noise = map(float, np.exp(result.x))
        return replace(
            self,
            level_variance=level,
            slope_variance=slope,
            observation_variance=noise,
            parameters_source="training_innovation_likelihood",
        )

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        horizon = validate_horizon(horizon)
        tau = np.asarray(history.tau, dtype=float)
        if tau[-1] != history.t0:
            raise ValueError("Kalman history must end at t0")
        fitted, last, _ = _kalman_weights(
            tau, history.y, self.level_variance, self.slope_variance, self.observation_variance
        )
        historical = fitted @ history.y
        offset, _ = anchor_offset(history.y, historical)
        future = np.arange(history.t0 + 1, history.t0 + horizon + 1, dtype=float)
        unanchored = np.array([np.array([1.0, t - history.t0]) @ last for t in future])
        offset_weights = np.mean(np.eye(len(tau)) - fitted, axis=0)
        weights = unanchored + offset_weights
        times = tau - tau[0]
        future_times = future - tau[0]
        covariance = _level_covariance(times, times, self.level_variance, self.slope_variance)
        covariance += np.eye(len(tau)) * self.observation_variance
        cross = _level_covariance(future_times, times, self.level_variance, self.slope_variance)
        variance = np.diag(
            _level_covariance(future_times, future_times, self.level_variance, self.slope_variance)
        )
        variance = (
            variance + self.observation_variance + np.einsum("ij,jk,ik->i", weights, covariance, weights)
        )
        variance -= 2 * np.sum(weights * cross, axis=1)
        mean = unanchored @ history.y + offset
        sigma = np.sqrt(np.maximum(variance, SIGMA_FLOOR**2))
        return PredictionOutput(mean, mean[None, :], sigma, None, None, member_sigma_s=sigma[None, :])

    def state_digest(self) -> str:
        return _digest("B2", asdict(self))


def _mlp_features(tau: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    return np.column_stack((tau / TAU_SCALE, ctx))


@eqx.filter_jit
def _mlp_step(net, state, x, y, groups, counts, learning_rate):
    @eqx.filter_value_and_grad
    def loss(model):
        residual = jax.vmap(model)(x)[:, 0] - y
        center = jax.ops.segment_sum(residual, groups, num_segments=counts.shape[0]) / counts
        return jnp.mean((residual - center[groups]) ** 2)

    value, gradient = loss(net)
    optimizer = optax.adam(learning_rate)
    updates, state = optimizer.update(gradient, state, eqx.filter(net, eqx.is_inexact_array))
    return eqx.apply_updates(net, updates), state, value


class B3(eqx.Module):
    """Scalar physics-free MLP with M1's exact network inputs and default hidden width/depth."""

    net: eqx.nn.MLP
    noise_s: float
    n_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        key: jax.Array,
        *,
        width: int = 32,
        depth: int = 2,
        n_steps: int = 500,
        learning_rate: float = 0.01,
    ) -> B3:
        if n_steps < 1 or learning_rate <= 0:
            raise ValueError("MLP steps and learning rate must be positive")
        net = eqx.nn.MLP(1 + CTX_DIM, 1, width, depth, activation=jnp.tanh, key=key)
        return cls(net, 0.1, n_steps, learning_rate)

    def fit(self, train: StintBatch, key: jax.Array) -> B3:
        tau, y, groups, ctx, _, _ = _rows(train)
        x = jnp.asarray(_mlp_features(tau, ctx))
        targets = jnp.asarray(y)
        ids = jnp.asarray(groups)
        counts = jnp.asarray(np.bincount(groups))
        net = self.net
        state = optax.adam(self.learning_rate).init(eqx.filter(net, eqx.is_inexact_array))
        for _ in range(self.n_steps):
            net, state, value = _mlp_step(
                net, state, x, targets, ids, counts, jnp.asarray(self.learning_rate)
            )
        if not np.isfinite(float(value)):
            raise ValueError("MLP training produced nonfinite loss")
        shape = np.asarray(jax.vmap(net)(x))[:, 0]
        return eqx.tree_at(lambda m: (m.net, m.noise_s), self, (net, _noise(y, shape, groups)))

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        horizon = validate_horizon(horizon)
        tau = np.concatenate((history.tau, np.arange(history.t0 + 1, history.t0 + horizon + 1)))
        ctx = np.tile(np.asarray(history.ctx.features()), (len(tau), 1))
        values = np.asarray(jax.vmap(self.net)(jnp.asarray(_mlp_features(tau, ctx))))[:, 0]
        return _anchored(
            history, horizon, values[: len(history.tau)], values[len(history.tau) :], self.noise_s
        )

    def state_digest(self) -> str:
        leaves = jax.tree_util.tree_leaves(eqx.filter(self.net, eqx.is_inexact_array))
        return _digest("B3", self.noise_s, self.n_steps, self.learning_rate, *[np.asarray(v) for v in leaves])


@dataclass(frozen=True)
class BaselineEnsemble:
    """Equal-weight mixture of separately fitted baseline members, with explicit component scales."""

    members: tuple[Any, ...]

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        if not self.members:
            raise ValueError("ensemble must have members")
        predictions = [m.forecast(history, horizon, scenario) for m in self.members]
        if any(p.member_means_s.shape[0] != 1 for p in predictions):
            raise ValueError("nested baseline ensembles are not supported")
        means = np.stack([p.mean_s for p in predictions])
        scales = np.stack([p.sigma_s for p in predictions])
        return PredictionOutput(
            means.mean(axis=0),
            means,
            np.sqrt(np.mean(scales**2, axis=0) + np.var(means, axis=0)),
            None,
            None,
            member_sigma_s=scales,
        )

    def state_digest(self) -> str:
        return _digest("baseline_ensemble", [m.state_digest() for m in self.members])
