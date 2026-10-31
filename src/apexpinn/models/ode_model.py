"""M2: the gauge-fixed ODEs integrated with Diffrax, the comparator for M1 (PROJECT.md section 4.7).

Same Theta (residuals.PhysicsParams), same observation equation (residuals.observation), and the same
offset anchoring as M1. (h, w) come from Tsit5 with RecursiveCheckpointAdjoint instead of a network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from apexpinn.contracts import PredictionOutput
from apexpinn.models.pinn import INIT_NOISE_S, default_physics, masked_mean
from apexpinn.models.stints import (
    Scenario,
    StintBatch,
    StintHistory,
    build_prediction,
    forecast_q_knots,
    validate_horizon,
)
from apexpinn.physics.residuals import PhysicsParams, observation, thermal_rhs, wear_rhs
from apexpinn.train.trainer import LossWeights, TrainConfig


def _vector_field(t, y, args):
    physics, compound_idx, q_of_tau = args
    h, _ = y
    dh = thermal_rhs(physics, compound_idx, h, q_of_tau.evaluate(t))
    return dh, wear_rhs(physics, compound_idx, h)


def _diffeqsolve(
    physics: PhysicsParams,
    compound_idx: jax.Array,
    q_knots: jax.Array,
    saveat: diffrax.SaveAt,
    rtol: float,
    atol: float,
    max_steps: int,
) -> diffrax.Solution:
    """Solve both ODEs on tau in [0, K] from h(0) = w(0) = 0, with q linear between knots tau = 0..K.

    The interior knots are jump times, so no step spans a slope change of q and the error estimate stays
    valid (the synthetic generator solves the same way). This also keeps knots past a saved time out of
    the steps that produce it, up to a few ULPs at the knot itself.
    """
    knots = jnp.arange(q_knots.shape[0], dtype=q_knots.dtype)
    q_of_tau = diffrax.LinearInterpolation(ts=knots, ys=q_knots)
    zero = jnp.zeros((), dtype=q_knots.dtype)
    return diffrax.diffeqsolve(
        diffrax.ODETerm(_vector_field),
        diffrax.Tsit5(),
        t0=knots[0],
        t1=knots[-1],
        dt0=0.1,
        y0=(zero, zero),
        args=(physics, compound_idx, q_of_tau),
        saveat=saveat,
        stepsize_controller=diffrax.ClipStepSizeController(
            diffrax.PIDController(rtol=rtol, atol=atol), jump_ts=knots[1:-1]
        ),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
    )


@eqx.filter_jit
def _solve_at(physics, compound_idx, q_knots, ts, rtol, atol, max_steps):
    sol = _diffeqsolve(physics, compound_idx, q_knots, diffrax.SaveAt(ts=ts), rtol, atol, max_steps)
    return sol.ys


def _as_knots(q_knots: ArrayLike) -> jax.Array:
    q_knots = jnp.asarray(q_knots, dtype=float)
    if q_knots.ndim != 1 or q_knots.shape[0] < 2:
        raise ValueError(f"q_knots must be 1-D with at least 2 knots (tau = 0..K), got {q_knots.shape}")
    return q_knots


# ODEModel.forecast pads knot intervals and save times to multiples of this (review v1, issue 8).
FORECAST_PAD = 16


def _round_up(n: int, size: int) -> int:
    return -(-n // size) * size


def _pad_edge(values: np.ndarray, length: int) -> np.ndarray:
    """values (1-D) extended to length by repeating its last entry."""
    return np.concatenate([values, np.full(length - values.shape[0], values[-1], dtype=values.dtype)])


class ODEModel(eqx.Module):
    """M2. Only Theta and the observation noise are fitted; there is no network."""

    physics: PhysicsParams
    noise_s: jax.Array  # scalar observation noise std in seconds; fit sets it from the training residual
    train_cfg: TrainConfig = eqx.field(static=True)
    weights: LossWeights = eqx.field(static=True)
    rtol: float = eqx.field(static=True, default=1e-6)
    atol: float = eqx.field(static=True, default=1e-6)
    max_steps: int = eqx.field(static=True, default=4096)

    uses_collocation: ClassVar[bool] = False

    @classmethod
    def init(
        cls,
        *,
        physics: PhysicsParams | None = None,
        train_cfg: TrainConfig | None = None,
        weights: LossWeights | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> ODEModel:
        """New model. physics defaults to pinn.default_physics(), the same neutral start as M1."""
        for name, value in (("rtol", rtol), ("atol", atol)):
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        return cls(
            physics=default_physics() if physics is None else physics,
            noise_s=jnp.asarray(INIT_NOISE_S, dtype=float),
            train_cfg=TrainConfig() if train_cfg is None else train_cfg,
            weights=LossWeights() if weights is None else weights,
            rtol=float(rtol),
            atol=float(atol),
        )

    def solve(
        self, compound_idx: ArrayLike, q_knots: ArrayLike, ts: ArrayLike
    ) -> tuple[jax.Array, jax.Array]:
        """(h, w) at the 1-D, non-decreasing times ts in [0, K], where K = len(q_knots) - 1."""
        q_knots = _as_knots(q_knots)
        ts = jnp.asarray(ts, dtype=q_knots.dtype)
        if ts.ndim != 1:
            raise ValueError(f"ts must be 1-D, got shape {ts.shape}")
        h, w = _solve_at(
            self.physics, jnp.asarray(compound_idx), q_knots, ts, self.rtol, self.atol, self.max_steps
        )
        return h, w

    def dense_latent(
        self, compound_idx: ArrayLike, q_knots: ArrayLike
    ) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
        """Scalar tau -> (h, w) from the dense Tsit5 solution on [0, K].

        The interpolant is differentiable in tau (jax.jvp), so it plugs into residuals.states_and_rates.
        """
        q_knots = _as_knots(q_knots)
        sol = _diffeqsolve(
            self.physics,
            jnp.asarray(compound_idx),
            q_knots,
            diffrax.SaveAt(dense=True),
            self.rtol,
            self.atol,
            self.max_steps,
        )

        def latent(tau: jax.Array) -> tuple[jax.Array, jax.Array]:
            h, w = sol.evaluate(tau)
            return h, w

        return latent

    def shape(
        self, compound_idx: ArrayLike, q_knots: ArrayLike, tau: ArrayLike
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(g, h, w) at tau, where g is the lap-time shape: observation with b = 0."""
        h, w = self.solve(compound_idx, q_knots, tau)
        return observation(self.physics, compound_idx, h, w, 0.0), h, w

    def batch_shape(self, batch: StintBatch) -> jax.Array:
        """Lap-time shape g (observation with b = 0) at batch.tau for every stint, shape (n, K).

        Used by the trainer for the closed-form initial training offsets.
        """
        return jax.vmap(lambda c, q: self.shape(c, q, batch.tau)[0])(batch.compound_idx, batch.q)

    def loss_terms(
        self, b: jax.Array, batch: StintBatch, colloc_tau: jax.Array | None = None
    ) -> dict[str, jax.Array]:
        """Raw loss terms: 'data' (masked MSE at batch.tau) plus 'state' if batch.h_true is set.

        M2 has no collocation residuals, so colloc_tau is ignored.
        """
        h, w = jax.vmap(lambda c, q: self.solve(c, q, batch.tau))(batch.compound_idx, batch.q)
        y_hat = observation(self.physics, batch.compound_idx[:, None], h, w, b[:, None])
        terms = {"data": masked_mean((y_hat - batch.y) ** 2, batch.mask)}
        if batch.h_true is not None:
            terms["state"] = masked_mean((h - batch.h_true) ** 2 + (w - batch.w_true) ** 2, batch.mask)
        return terms

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        """Anchored forecast for stint laps t0+1..t0+horizon (E = 1).

        q follows the observed history up to t0 and the scenario q_bar after it (forecast_q_knots).
        The knots and save times are padded to multiples of FORECAST_PAD (knots with q_bar, times with the
        last time), so an origin and horizon sweep reuses a few compiled solves instead of one per shape.
        Padding only extends the solve past the last saved time, which the causal ODE never reads back.
        """
        horizon = validate_horizon(horizon)
        q_knots = forecast_q_knots(history, horizon, scenario)
        tau_future = np.arange(history.t0 + 1, history.t0 + horizon + 1, dtype=np.float64)
        ts = np.concatenate([history.tau, tau_future])
        n_ts = ts.shape[0]
        q_pad = _pad_edge(q_knots, _round_up(q_knots.shape[0] - 1, FORECAST_PAD) + 1)
        ts_pad = _pad_edge(ts, _round_up(n_ts, FORECAST_PAD))
        g, h, w = (np.asarray(x)[:n_ts] for x in self.shape(history.ctx.compound_idx, q_pad, ts_pad))
        n_hist = history.tau.shape[0]
        return build_prediction(
            g[None, :n_hist],
            g[None, n_hist:],
            history.y,
            float(self.noise_s),
            latent_h=h[n_hist:],
            latent_w_s=w[n_hist:],
        )

    def fit(self, train: StintBatch, key: jax.Array) -> ODEModel:
        """Train with apexpinn.train.trainer.train using this model's weights and train_cfg."""
        from apexpinn.train.trainer import train as _train

        return _train(self, train, self.weights, self.train_cfg, key)[0]
