"""Model-agnostic trainer for M1 (PINN) and M2 (ODE model): loss weights, curriculum, and the Adam loop.

See PROJECT.md section 4.6 (loss, normalization by the value at init, curriculum, collocation points) and
PLAN.md T4. The trainer only calls model.loss_terms, model.batch_shape and reads model.uses_collocation and
model.noise_s, so this module must not import apexpinn.models.

Training offsets: the per-stint offset b_s is a free trainable array owned by the trainer. It starts at the
closed-form value (masked mean of y minus the initial lap-time shape) and is returned in TrainLog.b_train.
At forecast time b_s comes only from offset anchoring on history laps (PROJECT section 4.5).

Compilation: the jitted step (_update) and log evaluation (_evaluate) live at module level. Normalization
coefficients and the curriculum scale are passed in as arrays, so train() calls with the same model
structure, batch shapes and static settings reuse the compiled functions instead of tracing new closures.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.typing import ArrayLike

# Ramped by the curriculum (0 during warm-up, then linear to 1). 'data', 'state', 'reg' are constant.
PHYSICS_TERMS = ("thermal", "wear", "mono")


def _check_int(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _check_float(name: str, value: object, *, positive: bool) -> None:
    ok = isinstance(value, int | float | np.floating) and not isinstance(value, bool) and np.isfinite(value)
    if not ok or value < 0 or (positive and value == 0):
        rule = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {rule}, got {value!r}")


@dataclass(frozen=True)
class LossWeights:
    """Weights lambda of the loss terms returned by model.loss_terms (PROJECT section 4.6).

    Terms with weight 0 are skipped by the trainer. 'state' is only used when true states are observed
    (synthetic S1a); train() raises if state > 0 and the batch has no true states.
    """

    data: float = 1.0
    thermal: float = 1.0
    wear: float = 1.0
    mono: float = 0.1
    reg: float = 1e-4
    state: float = 0.0

    def __post_init__(self) -> None:
        for f in fields(self):
            _check_float(f.name, getattr(self, f.name), positive=f.name == "data")


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer and curriculum settings. Hashable, so it can be a static field on an eqx.Module.

    n_warm data-only steps come first, then lambda_T and lambda_W ramp linearly over n_ramp steps.
    Collocation points per stint (n_colloc) cover tau in [0, T_s + colloc_extension]. Each loss term is
    divided by its value at init, floored as in init_normalizers: 'data' by norm_floor, every other term by
    max(norm_floor, norm_rel_floor * data value at init). grad_clip=None disables global-norm clipping.
    """

    n_steps: int = 2000
    learning_rate: float = 1e-3
    n_warm: int = 200
    n_ramp: int = 500
    n_colloc: int = 32
    colloc_extension: float = 10.0
    log_every: int = 100
    float64: bool = False
    norm_floor: float = 1e-12
    norm_rel_floor: float = 1e-3
    grad_clip: float | None = 1.0

    def __post_init__(self) -> None:
        _check_int("n_steps", self.n_steps, 1)
        _check_int("n_warm", self.n_warm, 0)
        _check_int("n_ramp", self.n_ramp, 0)
        _check_int("n_colloc", self.n_colloc, 1)
        _check_int("log_every", self.log_every, 1)
        _check_float("learning_rate", self.learning_rate, positive=True)
        _check_float("colloc_extension", self.colloc_extension, positive=False)
        _check_float("norm_floor", self.norm_floor, positive=True)
        _check_float("norm_rel_floor", self.norm_rel_floor, positive=False)
        if not isinstance(self.float64, bool):
            raise ValueError(f"float64 must be a bool, got {self.float64!r}")
        if self.grad_clip is not None:
            _check_float("grad_clip", self.grad_clip, positive=True)


def curriculum_scale(step: ArrayLike, n_warm: int, n_ramp: int) -> jax.Array:
    """Multiplier of the physics terms at a step: 0 for step < n_warm, then linear to 1 over n_ramp steps.

    For step >= n_warm the value is clip((step - n_warm + 1) / n_ramp, 0, 1), or 1 if n_ramp == 0.
    step may be a traced integer array (or an array of steps), so this runs under jit.
    """
    _check_int("n_warm", n_warm, 0)
    _check_int("n_ramp", n_ramp, 0)
    step = jnp.asarray(step)
    if n_ramp == 0:
        ramp = jnp.ones((), dtype=float)
    else:
        ramp = jnp.clip((step - n_warm + 1) / n_ramp, 0.0, 1.0)
    return jnp.asarray(jnp.where(step < n_warm, 0.0, ramp), dtype=float)


def sample_collocation(key: jax.Array, n_laps: ArrayLike, n_colloc: int, extension: float) -> jax.Array:
    """Stratified uniform collocation points of shape (n, n_colloc) (PROJECT 4.6, GLOSSARY).

    Stint s gets tau_j = (j + u_j) / n_colloc * (n_laps[s] + extension), u_j ~ U(0, 1), so the points
    cover [0, T_s + extension] one per stratum and are almost surely non-integer.
    """
    _check_int("n_colloc", n_colloc, 1)
    n_laps = jnp.asarray(n_laps)
    if n_laps.ndim != 1:
        raise ValueError(f"n_laps must be 1-D (one length per stint), got shape {n_laps.shape}")
    dtype = jnp.result_type(float)
    u = jax.random.uniform(key, (n_laps.shape[0], n_colloc), dtype=dtype)
    strata = jnp.arange(n_colloc, dtype=dtype)
    length = n_laps.astype(dtype) + jnp.asarray(extension, dtype=dtype)
    return (strata[None, :] + u) / n_colloc * length[:, None]


def init_normalizers(
    raw0: Mapping[str, float], active: Sequence[str], norm_floor: float, norm_rel_floor: float
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Normalizers of the active terms from their raw values at init, and the terms that were floored.

    'data' is divided by max(raw0['data'], norm_floor). Every other term is divided by
    max(raw0[term], floor) with floor = max(norm_floor, norm_rel_floor * raw0['data']). The relative floor
    keeps a term that is almost 0 at init (mono, or state near the truth) from getting a coefficient of
    weight / 1e-8; a term whose init value is below its floor is reported in the second return value.
    The mapping is continuous in raw0, so 0 and 1e-11 give the same normalizer.
    """
    _check_float("norm_floor", norm_floor, positive=True)
    _check_float("norm_rel_floor", norm_rel_floor, positive=False)
    if "data" not in raw0:
        raise ValueError(f"raw0 must contain a 'data' term, got {sorted(raw0)}")
    missing = [t for t in active if t not in raw0]
    if missing:
        raise ValueError(f"active terms missing from raw0: {missing}")
    relative = max(norm_floor, norm_rel_floor * float(raw0["data"]))
    normalizers: dict[str, float] = {}
    floored: list[str] = []
    for term in active:
        value = float(raw0[term])
        floor = norm_floor if term == "data" else relative
        if value > floor:
            normalizers[term] = value
        else:
            normalizers[term] = floor
            floored.append(term)
    return normalizers, tuple(floored)


@dataclass(frozen=True, eq=False)  # eq=False: elementwise array == has no single truth value
class TrainLog:
    """Training trace. Entries are logged at step 0, every log_every steps and at the last step, before
    that step's update.

    terms holds every raw term returned by model.loss_terms. normalized, grad_norms, normalizers and
    coefficients (weight / normalizer, the factor applied inside the objective) hold the active terms
    (weight > 0) only; grad_norms is the L2 norm over all trainable leaves of the gradient of the
    normalized, unweighted term. objective is the value returned by the jitted update step, the quantity
    Adam differentiates. total is the same sum recomputed on the host from the logged raw terms, so the
    two agree up to float rounding. floored_terms are active terms whose init value was at or below their
    floor (see init_normalizers) and were divided by the floor instead.
    """

    steps: np.ndarray  # (n_log,) int
    terms: dict[str, np.ndarray]
    normalized: dict[str, np.ndarray]
    grad_norms: dict[str, np.ndarray]
    objective: np.ndarray
    total: np.ndarray
    curriculum: np.ndarray
    normalizers: dict[str, float]
    coefficients: dict[str, float]
    floored_terms: tuple[str, ...]
    b_train: np.ndarray  # (n_stints,) final training offsets
    wall_time_s: float


@dataclass(frozen=True)
class _Spec:
    """Static settings of the jitted functions. Hashable, so filter_jit caches on it."""

    active: tuple[str, ...]
    uses_collocation: bool
    n_colloc: int
    colloc_extension: float


def _to_float64(tree: Any) -> Any:
    """Cast every floating array leaf to float64 (integer, bool and non-array leaves are kept)."""
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, tree)


@functools.lru_cache(maxsize=32)
def _optimizer(learning_rate: float, grad_clip: float | None) -> optax.GradientTransformation:
    """Adam, after global-norm clipping if grad_clip is set. Cached, so equal settings give the same
    object and the jitted update is not traced again."""
    transforms = [optax.adam(learning_rate)]
    if grad_clip is not None:
        transforms.insert(0, optax.clip_by_global_norm(grad_clip))
    return optax.chain(*transforms)


@eqx.filter_jit
def _closed_form_offsets(model: Any, data: Any) -> jax.Array:
    """b_s = masked mean over scored laps of (y - g), with g the model's lap-time shape at data.tau."""
    g = model.batch_shape(data)
    resid = jnp.where(data.mask, data.y - g, 0.0)
    count = jnp.sum(data.mask.astype(resid.dtype), axis=1)
    return jnp.sum(resid, axis=1) / jnp.maximum(count, 1.0)


def _colloc_at(spec: _Spec, key: jax.Array, step: jax.Array, n_laps: jax.Array) -> jax.Array | None:
    if not spec.uses_collocation:
        return None
    return sample_collocation(jax.random.fold_in(key, step), n_laps, spec.n_colloc, spec.colloc_extension)


def _raw_terms(p: Any, static: Any, batch: Any, colloc: jax.Array | None) -> dict[str, jax.Array]:
    model, b = eqx.combine(p, static)
    return model.loss_terms(b, batch, colloc)


def _weighted_objective(
    raw: Mapping[str, jax.Array], coefs: Mapping[str, jax.Array], scale: jax.Array
) -> jax.Array:
    """sum over active terms of coef * c * raw, with c = scale for PHYSICS_TERMS and 1 otherwise."""
    total = jnp.zeros((), dtype=scale.dtype)
    for term, coef in coefs.items():
        c = scale if term in PHYSICS_TERMS else 1.0
        total = total + coef * c * raw[term]
    return total


@eqx.filter_jit
def _evaluate(p: Any, static: Any, batch: Any, key: jax.Array, step: jax.Array, spec: _Spec):
    """Raw terms and the gradient norm of each raw active term, with step's collocation sample.

    One forward pass (jax.vjp of the term dict) is shared by all terms; each term is one pullback.
    """
    colloc = _colloc_at(spec, key, step, batch.n_laps)
    raw, pullback = jax.vjp(lambda q: _raw_terms(q, static, batch, colloc), p)
    norms = {}
    zeros = {t: jnp.zeros_like(v) for t, v in raw.items()}
    for term in spec.active:
        (grads,) = pullback({**zeros, term: jnp.ones_like(raw[term])})
        norms[term] = optax.tree.norm(grads)
    return raw, norms


@eqx.filter_jit
def _update(
    p: Any,
    static: Any,
    opt_state: Any,
    batch: Any,
    key: jax.Array,
    step: jax.Array,
    scale: jax.Array,
    coefs: dict[str, jax.Array],
    spec: _Spec,
    optimizer: optax.GradientTransformation,
):
    """One Adam step. Returns (objective before the update, new params, new optimizer state)."""
    colloc = _colloc_at(spec, key, step, batch.n_laps)

    def objective(q: Any) -> jax.Array:
        return _weighted_objective(_raw_terms(q, static, batch, colloc), coefs, scale)

    value, grads = jax.value_and_grad(objective)(p)
    updates, opt_state = optimizer.update(grads, opt_state, p)
    return value, eqx.apply_updates(p, updates), opt_state


def train(
    model: Any, data: Any, weights: LossWeights, cfg: TrainConfig, key: jax.Array
) -> tuple[Any, TrainLog]:
    """Fit model (PINN or ODEModel) to the StintBatch data with Adam. Returns (fitted model, TrainLog).

    The objective at step i is sum over active terms of weight * c_i * raw / normalizer, where c_i is
    curriculum_scale(i) for PHYSICS_TERMS and 1 otherwise, and the normalizer is the raw value at init
    (step 0, same collocation sample), floored as in init_normalizers. Trainable leaves are the inexact
    arrays of the model and the per-stint offsets b. noise_s is not in the loss, so Adam leaves it unchanged
    until it is set to sqrt(final data MSE) after the loop.
    """
    if not isinstance(weights, LossWeights):
        raise ValueError(f"weights must be a LossWeights, got {type(weights).__name__}")
    if not isinstance(cfg, TrainConfig):
        raise ValueError(f"cfg must be a TrainConfig, got {type(cfg).__name__}")
    if weights.state > 0 and getattr(data, "h_true", None) is None:
        raise ValueError(
            "weights.state > 0 needs true states in the batch: build it with "
            "batch_from_synthetic(ds, observe_states=True), or set state=0"
        )
    if cfg.float64:
        if not jax.config.x64_enabled:
            raise ValueError(
                "float64 mode requires jax_enable_x64: call jax.config.update('jax_enable_x64', True) first"
            )
        model = _to_float64(model)
        data = _to_float64(data)

    start = time.perf_counter()
    k_colloc, _ = jax.random.split(key)
    b0 = _closed_form_offsets(model, data)
    dtype = b0.dtype
    params, static = eqx.partition((model, b0), eqx.is_inexact_array)
    spec = _Spec(
        active=(),
        uses_collocation=bool(model.uses_collocation),
        n_colloc=int(cfg.n_colloc),
        colloc_extension=float(cfg.colloc_extension),
    )

    def raw_terms_at(p: Any, batch: Any, k: jax.Array, step: jax.Array) -> dict[str, jax.Array]:
        return _raw_terms(p, static, batch, _colloc_at(spec, k, step, batch.n_laps))

    step0 = jnp.asarray(0, dtype=jnp.int32)
    names = tuple(eqx.filter_eval_shape(raw_terms_at, params, data, k_colloc, step0))
    unknown = [t for t in names if t not in {f.name for f in fields(LossWeights)}]
    if unknown:
        raise ValueError(f"model.loss_terms returned terms without a LossWeights field: {unknown}")
    if "data" not in names:
        raise ValueError(f"model.loss_terms must return a 'data' term, got {list(names)}")
    spec = replace(spec, active=tuple(t for t in names if getattr(weights, t) > 0))

    log_steps: list[int] = []
    log_raw: list[dict[str, float]] = []
    log_gnorm: list[dict[str, float]] = []
    log_objective: list[float] = []

    def record(step: int, p: Any) -> dict[str, float]:
        raw, norms = _evaluate(p, static, data, k_colloc, jnp.asarray(step, dtype=jnp.int32), spec)
        values = {t: float(v) for t, v in raw.items()}
        log_steps.append(step)
        log_raw.append(values)
        log_gnorm.append({t: float(v) for t, v in norms.items()})
        return values

    raw0 = record(0, params)
    normalizers, floored = init_normalizers(raw0, spec.active, cfg.norm_floor, cfg.norm_rel_floor)
    coefficients = {t: getattr(weights, t) / normalizers[t] for t in spec.active}
    coefs = {t: jnp.asarray(c, dtype=dtype) for t, c in coefficients.items()}
    scales = np.asarray(curriculum_scale(jnp.arange(cfg.n_steps), cfg.n_warm, cfg.n_ramp), dtype=np.float64)

    optimizer = _optimizer(float(cfg.learning_rate), None if cfg.grad_clip is None else float(cfg.grad_clip))
    opt_state = optimizer.init(params)

    last = cfg.n_steps - 1
    for i in range(cfg.n_steps):
        logged = i % cfg.log_every == 0 or i == last
        if logged and i > 0:
            record(i, params)
        value, params, opt_state = _update(
            params,
            static,
            opt_state,
            data,
            k_colloc,
            jnp.asarray(i, dtype=jnp.int32),
            jnp.asarray(scales[i], dtype=dtype),
            coefs,
            spec,
            optimizer,
        )
        if logged:
            log_objective.append(float(value))

    final_raw, _ = _evaluate(params, static, data, k_colloc, jnp.asarray(cfg.n_steps, dtype=jnp.int32), spec)
    fitted, b_train = eqx.combine(params, static)
    data_mse = max(float(final_raw["data"]), cfg.norm_floor)
    noise_s = jnp.asarray(np.sqrt(data_mse), dtype=fitted.noise_s.dtype)
    fitted = eqx.tree_at(lambda m: m.noise_s, fitted, noise_s)
    b_train = np.asarray(jax.block_until_ready(b_train))
    wall = time.perf_counter() - start

    log_scales = scales[np.asarray(log_steps, dtype=np.int64)]
    terms = {t: np.array([r[t] for r in log_raw]) for t in names}
    normalized = {t: terms[t] / normalizers[t] for t in spec.active}
    total = np.zeros(len(log_steps))
    for t in spec.active:
        c = log_scales if t in PHYSICS_TERMS else 1.0
        total = total + getattr(weights, t) * c * normalized[t]
    log = TrainLog(
        steps=np.asarray(log_steps, dtype=np.int64),
        terms=terms,
        normalized=normalized,
        grad_norms={t: np.array([g[t] for g in log_gnorm]) / normalizers[t] for t in spec.active},
        objective=np.asarray(log_objective, dtype=np.float64),
        total=total,
        curriculum=log_scales,
        normalizers=normalizers,
        coefficients=coefficients,
        floored_terms=floored,
        b_train=b_train,
        wall_time_s=wall,
    )
    return fitted, log
