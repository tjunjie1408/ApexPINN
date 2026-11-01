"""Shared stint types for every forecaster: context, history, scenario, training batch, offset anchoring.

See PROJECT.md section 4.5 (forecasting protocol, offset anchoring, scenario heat load), section 5.2
(Forecaster protocol and PredictionOutput), and PLAN.md section 4 (StintHistory schema).

On real data, data.guards.slice_at_origin (PLAN T9) is the only producer of StintHistory and StintBatch.
The SyntheticDataset adapters here mirror its contract for synthetic studies: the history never contains
laps after the origin t0, and y_true is returned separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from apexpinn.contracts import PredictionOutput
from apexpinn.physics.residuals import N_COMPOUNDS
from apexpinn.physics.synthetic import SyntheticDataset

TRACK_TEMP_REF_C = 35.0
TRACK_TEMP_SCALE_C = 10.0
AGE_SCALE_LAPS = 10.0
CTX_DIM = N_COMPOUNDS + 4  # one-hot compound, then age, fresh, track temperature, circuit q mean
# O5 k_min (configs/decisions_v1.yaml o5_protocol.k_min). Every valid origin t0 >= k_min has seen the heat
# load knots tau = 0..k_min, so the synthetic circuit_q_mean uses only those knots (PROJECT 4.5).
SYNTHETIC_K_MIN = 3


def ctx_features(
    compound_idx: ArrayLike,
    age_at_start: ArrayLike,
    fresh: ArrayLike,
    track_temp_start: ArrayLike,
    circuit_q_mean: ArrayLike,
) -> jax.Array:
    """Normalized context vector(s) of shape (..., CTX_DIM). Inputs broadcast over a leading stint axis.

    Layout: one-hot compound (N_COMPOUNDS), age_at_start / AGE_SCALE_LAPS, float(fresh),
    (track_temp_start - TRACK_TEMP_REF_C) / TRACK_TEMP_SCALE_C, circuit_q_mean.
    """
    onehot = jnp.asarray(jax.nn.one_hot(jnp.asarray(compound_idx), N_COMPOUNDS), dtype=float)
    extras = jnp.stack(
        jnp.broadcast_arrays(
            jnp.asarray(age_at_start, dtype=float) / AGE_SCALE_LAPS,
            jnp.asarray(fresh, dtype=float),
            (jnp.asarray(track_temp_start, dtype=float) - TRACK_TEMP_REF_C) / TRACK_TEMP_SCALE_C,
            jnp.asarray(circuit_q_mean, dtype=float),
        ),
        axis=-1,
    )
    lead = jnp.broadcast_shapes(onehot.shape[:-1], extras.shape[:-1])
    return jnp.concatenate(
        [jnp.broadcast_to(onehot, (*lead, N_COMPOUNDS)), jnp.broadcast_to(extras, (*lead, 4))], axis=-1
    )


def _finite_float(name: str, value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return out


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def validate_horizon(horizon: object) -> int:
    """Return horizon as an int, or raise ValueError unless it is an integer >= 1."""
    return _check_int("horizon", horizon, 1)


@dataclass(frozen=True)
class StintContext:
    """Per-stint context known at the start of the stint (PLAN section 4)."""

    compound_idx: int  # index into residuals.COMPOUNDS
    age_at_start: float  # tire age in laps at the first stint lap
    fresh: bool
    track_temp_start: float  # degrees C
    circuit_q_mean: float  # circuit-level mean heat load proxy

    def __post_init__(self) -> None:
        idx = self.compound_idx
        if isinstance(idx, bool) or not isinstance(idx, int | np.integer) or not 0 <= idx < N_COMPOUNDS:
            raise ValueError(f"compound_idx must be an integer in [0, {N_COMPOUNDS}), got {idx!r}")
        object.__setattr__(self, "compound_idx", int(idx))
        age = _finite_float("age_at_start", self.age_at_start)
        if age < 0:
            raise ValueError(f"age_at_start must be >= 0, got {age}")
        object.__setattr__(self, "age_at_start", age)
        object.__setattr__(self, "fresh", bool(self.fresh))
        object.__setattr__(self, "track_temp_start", _finite_float("track_temp_start", self.track_temp_start))
        object.__setattr__(self, "circuit_q_mean", _finite_float("circuit_q_mean", self.circuit_q_mean))

    def features(self) -> jax.Array:
        """Context vector of shape (CTX_DIM,)."""
        return ctx_features(
            self.compound_idx, self.age_at_start, self.fresh, self.track_temp_start, self.circuit_q_mean
        )


@dataclass(frozen=True, eq=False)  # eq=False: elementwise array == has no single truth value
class StintHistory:
    """Scored laps of one stint up to the forecast origin t0 (PLAN section 4 schema plus t0).

    data.guards.slice_at_origin (PLAN T9) is the only real-data producer. The history never contains laps
    after t0: construction raises ValueError if any tau > t0. tau is the stint lap (1-based), so gaps are
    laps removed by the population filters (PROJECT section 4.5).
    """

    stint_id: str
    t0: int  # forecast origin, stint lap
    tau: np.ndarray  # (n_hist,) scored stint laps, strictly increasing, 1 <= tau <= t0
    y: np.ndarray  # (n_hist,) target at those laps, in seconds
    q: np.ndarray  # (n_hist,) heat load proxy at those laps
    ctx: StintContext
    race_lap_t0: int

    def __post_init__(self) -> None:
        t0 = _check_int("t0", self.t0, 1)
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "race_lap_t0", _check_int("race_lap_t0", self.race_lap_t0, 0))
        if not isinstance(self.ctx, StintContext):
            raise ValueError(f"ctx must be a StintContext, got {type(self.ctx).__name__}")
        arrays = {}
        for name in ("tau", "y", "q"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] < 1:
                raise ValueError(f"{name} must be 1-D with at least one lap, got shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")
            arrays[name] = arr
        tau = arrays["tau"]
        if arrays["y"].shape != tau.shape or arrays["q"].shape != tau.shape:
            shapes = ", ".join(str(arr.shape) for arr in arrays.values())
            raise ValueError(f"tau, y, q must have equal shapes, got {shapes}")
        if np.any(np.diff(tau) <= 0):
            raise ValueError("tau must be strictly increasing")
        if tau[0] < 1:
            raise ValueError(f"tau must start at stint lap >= 1, got {tau[0]}")
        if tau[-1] > t0:
            raise ValueError(f"history contains lap tau={tau[-1]} after the origin t0={t0}")
        for name, arr in arrays.items():
            object.__setattr__(self, name, arr)


@dataclass(frozen=True)
class Scenario:
    """Future heat load assumed after t0 (PROJECT section 4.5)."""

    q_bar: float

    def __post_init__(self) -> None:
        q_bar = _finite_float("q_bar", self.q_bar)
        if not 0.0 <= q_bar <= 1.0:
            raise ValueError(f"q_bar must be in [0, 1], got {q_bar}")
        object.__setattr__(self, "q_bar", q_bar)

    @classmethod
    def from_history(cls, history: StintHistory, n_last: int = 3) -> Scenario:
        """q_bar = mean of q over history laps tau > t0 - n_last (PROJECT 4.5: mean(q_{t0-2..t0})).

        If the filters removed every lap in that window, the q of the last history lap is used.
        """
        n_last = _check_int("n_last", n_last, 1)
        window = history.tau > history.t0 - n_last
        q = history.q[window] if np.any(window) else history.q[-1:]
        return cls(q_bar=float(np.mean(q)))


class StintBatch(eqx.Module):
    """Padded training stints. Row s is stint s, lap column k - 1 holds stint lap k (tau = k).

    n_laps[s] is the last stint lap T_s and defines the collocation domain [0, T_s + colloc_extension].
    h_true and w_true are true latent states, only available for synthetic S1a.

    Padding contract (for data.guards in PLAN T9): y, h_true, w_true past n_laps are unused and may be 0.
    q knots past n_laps are also unused: PINN.loss_terms holds q at knot n_laps over (T_s, T_s + extension],
    and the M2 solve is causal, so laps tau <= T_s never see later knots. Any finite padding is valid.
    """

    compound_idx: jax.Array  # (n,) int
    ctx: jax.Array  # (n, CTX_DIM)
    tau: jax.Array  # (K,) float, 1..K
    mask: jax.Array  # (n, K) bool, scored laps
    y: jax.Array  # (n, K) target, padding unused
    q: jax.Array  # (n, K + 1) heat load knots at tau = 0..K
    n_laps: jax.Array  # (n,) int
    h_true: jax.Array | None = None  # (n, K)
    w_true: jax.Array | None = None  # (n, K)

    def __check_init__(self) -> None:
        n = self.compound_idx.shape[0] if self.compound_idx.ndim == 1 else -1
        k = self.tau.shape[0] if self.tau.ndim == 1 else -1
        if n < 1 or k < 1:
            shapes = f"{self.compound_idx.shape}, {self.tau.shape}"
            raise ValueError(f"compound_idx and tau must be non-empty 1-D, got {shapes}")
        expected = {
            "ctx": (n, CTX_DIM),
            "mask": (n, k),
            "y": (n, k),
            "q": (n, k + 1),
            "n_laps": (n,),
            "h_true": (n, k),
            "w_true": (n, k),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value is not None and value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if (self.h_true is None) != (self.w_true is None):
            raise ValueError("h_true and w_true must be given together")

    @property
    def n_stints(self) -> int:
        return self.compound_idx.shape[0]


def _synthetic_circuit_q_mean(ds: SyntheticDataset) -> jax.Array:
    """Mean q over knots tau = 0..SYNTHETIC_K_MIN per stint, so the context holds no post-t0 heat load.

    Synthetic q is a per-stint circuit level plus noise, so this is a noisy estimate of the level. Training
    (batch_from_synthetic) and forecasting (history_from_synthetic) use the same value.
    """
    return jnp.mean(ds.q[:, : SYNTHETIC_K_MIN + 1], axis=1)


def batch_from_synthetic(ds: SyntheticDataset, observe_states: bool = False) -> StintBatch:
    """Training batch from a SyntheticDataset: fresh tires, age 0, reference track temperature.

    observe_states=True attaches the true (h, w), for the S1a recovery test only.
    """
    ctx = ctx_features(ds.compound_idx, 0.0, True, TRACK_TEMP_REF_C, _synthetic_circuit_q_mean(ds))
    return StintBatch(
        compound_idx=ds.compound_idx,
        ctx=ctx,
        tau=ds.tau,
        mask=ds.mask,
        y=ds.y,
        q=ds.q,
        n_laps=ds.n_laps,
        h_true=ds.h if observe_states else None,
        w_true=ds.w if observe_states else None,
    )


def history_from_synthetic(
    ds: SyntheticDataset, stint: int, t0: int, horizon: int
) -> tuple[StintHistory, Scenario, np.ndarray]:
    """(history, scenario, y_true) for one synthetic stint at origin t0, mirroring slice_at_origin.

    The history holds the scored laps tau <= t0 only, and t0 must be >= SYNTHETIC_K_MIN (O5 k_min).
    y_true = y at stint laps t0+1..t0+horizon is returned separately and must never be passed to
    Forecaster.forecast. The context uses the same circuit_q_mean as batch_from_synthetic (knots
    0..SYNTHETIC_K_MIN, all at or before t0), so a stint gets the same context at training and forecast time.
    """
    n_stints = ds.compound_idx.shape[0]
    stint = _check_int("stint", stint, 0)
    if stint >= n_stints:
        raise ValueError(f"stint must be < {n_stints}, got {stint}")
    t0 = _check_int("t0", t0, SYNTHETIC_K_MIN)
    horizon = validate_horizon(horizon)
    n_laps = int(ds.n_laps[stint])
    if t0 + horizon > n_laps:
        raise ValueError(f"t0 + horizon = {t0 + horizon} exceeds the stint length {n_laps}")

    tau_all = np.asarray(ds.tau)
    keep = np.asarray(ds.mask[stint]) & (tau_all <= t0)
    tau = tau_all[keep]
    lap = tau.astype(int)
    ctx = StintContext(
        compound_idx=int(ds.compound_idx[stint]),
        age_at_start=0.0,
        fresh=True,
        track_temp_start=TRACK_TEMP_REF_C,
        circuit_q_mean=float(_synthetic_circuit_q_mean(ds)[stint]),
    )
    history = StintHistory(
        stint_id=f"synthetic_{stint}",
        t0=t0,
        tau=tau,
        y=np.asarray(ds.y[stint])[lap - 1],
        q=np.asarray(ds.q[stint])[lap],
        ctx=ctx,
        race_lap_t0=int(ds.race_lap[stint, t0 - 1]),
    )
    y_true = np.asarray(ds.y[stint, t0 : t0 + horizon])
    return history, Scenario.from_history(history), y_true


def anchor_offset(y_hist: ArrayLike, shape_hist: ArrayLike) -> tuple[float, float]:
    """Closed-form stint offset from history laps (PROJECT section 4.5), shared by every model.

    Returns b = mean(y - shape) and the population residual variance mean((y - shape - b)^2).
    T11 may move this into eval/protocol.py and re-export it here.
    """
    y = np.asarray(y_hist, dtype=np.float64)
    shape = np.asarray(shape_hist, dtype=np.float64)
    if y.ndim != 1 or y.shape[0] < 1 or shape.shape != y.shape:
        raise ValueError(f"y_hist and shape_hist must be equal non-empty 1-D, got {y.shape}, {shape.shape}")
    resid = y - shape
    b = float(np.mean(resid))
    return b, float(np.mean((resid - b) ** 2))


def forecast_q_knots(history: StintHistory, horizon: int, scenario: Scenario) -> np.ndarray:
    """Heat load knots at tau = 0..t0+horizon: observed q (linear in gaps, clamped at 0), then q_bar."""
    horizon = validate_horizon(horizon)
    tau = np.arange(history.t0 + horizon + 1, dtype=np.float64)
    past = np.interp(tau, history.tau, history.q)
    return np.where(tau > history.t0, scenario.q_bar, past)


def build_prediction(
    shape_hist: ArrayLike,
    shape_future: ArrayLike,
    y_hist: ArrayLike,
    noise_s: float,
    latent_h: ArrayLike | None = None,
    latent_w_s: ArrayLike | None = None,
) -> PredictionOutput:
    """Anchor each member's shape on the history and combine members (PROJECT sections 4.5, 4.6).

    shape_hist is (E, n_hist) and shape_future is (E, horizon). Member e has offset b_e and residual
    variance var_e from anchor_offset. member_means = b_e + shape_future[e], mean_s is their mean, and
    sigma_s = sqrt(mean_e(max(var_e, noise_s^2) / n_hist) + noise_s^2 + variance of the member means)
    (PROJECT 4.6 "Uncertainty").

    max(var_e, noise_s^2) / n_hist is the variance of the anchored offset b_e, the uncertainty that offset
    anchoring adds. The raw residual variance var_e is not added on top of noise_s^2, because both
    estimate the same observation noise and their sum would be about sqrt(2) too wide (review v1,
    issue 3). The max keeps the term stable when n_hist is tiny (t0 = k_min = 3), where var_e can be
    near zero by chance, and still widens sigma_s when the shape misfits the history (var_e > noise_s^2).
    """
    sh = np.asarray(shape_hist, dtype=np.float64)
    sf = np.asarray(shape_future, dtype=np.float64)
    if sh.ndim != 2 or sf.ndim != 2 or sh.shape[0] != sf.shape[0] or sh.shape[0] < 1:
        raise ValueError(
            f"shape_hist and shape_future must be (E, n_hist) and (E, horizon), got {sh.shape}, {sf.shape}"
        )
    noise = _finite_float("noise_s", noise_s)
    if noise < 0:
        raise ValueError(f"noise_s must be >= 0, got {noise}")
    anchored = [anchor_offset(y_hist, member) for member in sh]
    offsets = np.array([b for b, _ in anchored])
    offset_var = np.maximum(np.array([v for _, v in anchored]), noise**2) / sh.shape[1]
    members = offsets[:, None] + sf
    sigma = np.sqrt(np.mean(offset_var) + noise**2 + np.var(members, axis=0))
    return PredictionOutput(
        mean_s=np.mean(members, axis=0),
        member_means_s=members,
        sigma_s=sigma,
        latent_h=None if latent_h is None else np.asarray(latent_h),
        latent_w_s=None if latent_w_s is None else np.asarray(latent_w_s),
    )


@runtime_checkable
class Forecaster(Protocol):
    """Shared model protocol (PROJECT section 5.2). forecast never receives a target argument."""

    def fit(self, train: StintBatch, key: jax.Array) -> Forecaster: ...

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput: ...
