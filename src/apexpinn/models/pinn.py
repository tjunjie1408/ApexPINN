"""M1: inverse PINN with hard initial conditions and autodiff rates (PROJECT.md section 4.6).

The latent ansatz h = tau * n_h, w = tau * softplus(n_w) follows PLAN.md section 4, so h(0) = w(0) = 0
exactly. latent_mode 'tanh' bounds the thermal state instead: h = tanh(tau * n_h), in (-1, 1) with
h(0) = 0 exactly; it exists for the exploratory M1 diagnostics (PLAN T5, `M1` exploratory diagnostics)
and w is the same in both modes. Rates come from residuals.states_and_rates (forward-mode autodiff), and
lap time from the observation equation shared with M2 (residuals.observation).
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from apexpinn.contracts import PredictionOutput
from apexpinn.models.stints import (
    CTX_DIM,
    Scenario,
    StintBatch,
    StintHistory,
    build_prediction,
    validate_horizon,
)
from apexpinn.physics.residuals import (
    PhysicsParams,
    observation,
    states_and_rates,
    thermal_residual,
    wear_residual,
)
from apexpinn.train.trainer import LossWeights, TrainConfig

TAU_SCALE = 40.0  # network input is tau / TAU_SCALE, about [0, 1] over the collocation domain
INIT_NOISE_S = 0.1
LATENT_MODES = ("linear", "tanh")  # 'linear': h = tau * n_h (v1); 'tanh': h = tanh(tau * n_h)


def default_physics() -> PhysicsParams:
    """Neutral starting Theta for fitting, not the synthetic truth. T5 passes prior draws via physics=."""
    return PhysicsParams.from_constrained(k=0.3, rho=0.05, gamma=1.0, h_opt=0.5, kappa=0.5)


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean of values over the True entries of mask (0 if mask is empty)."""
    weights = mask.astype(values.dtype)
    return jnp.sum(weights * values) / jnp.maximum(jnp.sum(weights), 1.0)


class PINN(eqx.Module):
    """M1. The network N(tau / TAU_SCALE, ctx) -> (n_h, n_w) is an MLP with tanh activations.

    latent_mode (one of LATENT_MODES) selects the thermal-state ansatz; the default 'linear' is v1.
    """

    net: eqx.nn.MLP
    physics: PhysicsParams
    noise_s: jax.Array  # scalar observation noise std in seconds; fit sets it from the training residual
    train_cfg: TrainConfig = eqx.field(static=True)
    weights: LossWeights = eqx.field(static=True)
    latent_mode: str = eqx.field(static=True, default="linear")

    uses_collocation: ClassVar[bool] = True

    @classmethod
    def init(
        cls,
        key: jax.Array,
        *,
        width: int = 32,
        depth: int = 2,
        physics: PhysicsParams | None = None,
        train_cfg: TrainConfig | None = None,
        weights: LossWeights | None = None,
        latent_mode: str = "linear",
    ) -> PINN:
        """New model. Network dtype follows jax_enable_x64; physics defaults to default_physics()."""
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError(f"width must be an integer >= 1, got {width!r}")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError(f"depth must be an integer >= 0, got {depth!r}")
        if latent_mode not in LATENT_MODES:
            raise ValueError(f"latent_mode must be one of {LATENT_MODES}, got {latent_mode!r}")
        net = eqx.nn.MLP(
            in_size=1 + CTX_DIM, out_size=2, width_size=width, depth=depth, activation=jnp.tanh, key=key
        )
        return cls(
            net=net,
            physics=default_physics() if physics is None else physics,
            noise_s=jnp.asarray(INIT_NOISE_S, dtype=float),
            train_cfg=TrainConfig() if train_cfg is None else train_cfg,
            weights=LossWeights() if weights is None else weights,
            latent_mode=latent_mode,
        )

    def latent(self, tau: ArrayLike, ctx: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """(h, w) at a scalar tau for one context vector of shape (CTX_DIM,).

        h = tau * n_h ('linear') or tanh(tau * n_h) ('tanh'); w = tau * softplus(n_w) in both modes.
        """
        tau = jnp.asarray(tau)
        if tau.ndim != 0:
            raise ValueError(f"tau must be a scalar, got shape {tau.shape}")
        out = self.net(jnp.concatenate([jnp.reshape(tau / TAU_SCALE, (1,)), jnp.asarray(ctx)]))
        h = tau * out[0]
        if self.latent_mode == "tanh":
            h = jnp.tanh(h)
        return h, tau * jax.nn.softplus(out[1])

    def latent_and_rates(
        self, tau: ArrayLike, ctx: ArrayLike
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """(h, w, dh/dtau, dw/dtau) on a 1-D tau grid, via residuals.states_and_rates."""
        return states_and_rates(lambda t: self.latent(t, ctx), tau)

    def shape(
        self, compound_idx: ArrayLike, ctx: ArrayLike, tau: ArrayLike
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(g, h, w) on a 1-D tau grid, where g is the lap-time shape: observation with b = 0."""
        tau = jnp.asarray(tau, dtype=float)
        if tau.ndim != 1:
            raise ValueError(f"tau must be 1-D, got shape {tau.shape}")
        h, w = jax.vmap(lambda t: self.latent(t, ctx))(tau)
        return observation(self.physics, compound_idx, h, w, 0.0), h, w

    def batch_shape(self, batch: StintBatch) -> jax.Array:
        """Lap-time shape g (observation with b = 0) at batch.tau for every stint, shape (n, K).

        Used by the trainer for the closed-form initial training offsets.
        """
        return jax.vmap(lambda c, ctx: self.shape(c, ctx, batch.tau)[0])(batch.compound_idx, batch.ctx)

    def weight_norm_sq(self) -> jax.Array:
        """Sum of squares of the network parameters (physics parameters excluded)."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self.net, eqx.is_inexact_array))
        return jnp.sum(jnp.stack([jnp.sum(leaf**2) for leaf in leaves]))

    def loss_terms(
        self, b: jax.Array, batch: StintBatch, colloc_tau: jax.Array | None
    ) -> dict[str, jax.Array]:
        """Raw (unweighted) loss terms of PROJECT section 4.6, as scalars keyed by name.

        b is the (n,) training offset per stint and colloc_tau the (n, m) collocation points. 'data' is the
        masked mean squared error at batch.tau. 'thermal', 'wear' are mean r_T^2, r_W^2 and 'mono' is
        mean ReLU(-dw/dtau)^2, all at the collocation points, where q is linear between the knots up to
        knot n_laps[s] and held at q[n_laps[s]] beyond it, so padded knots past T_s never enter the loss
        (StintBatch padding contract). 'reg' is the squared norm of the network weights. 'state' is the
        masked mean of (h - h_true)^2 + (w - w_true)^2 and is present only if batch.h_true is set.
        A wear-only R1 variant would supply a different latent/shape and drop 'thermal'.
        """
        if colloc_tau is None:
            raise ValueError("PINN.loss_terms needs collocation points colloc_tau of shape (n, m)")
        colloc_tau = jnp.asarray(colloc_tau, dtype=batch.tau.dtype)
        if colloc_tau.ndim != 2 or colloc_tau.shape[0] != batch.n_stints:
            raise ValueError(f"colloc_tau must have shape ({batch.n_stints}, m), got {colloc_tau.shape}")
        physics = self.physics
        knots = jnp.arange(batch.q.shape[1], dtype=batch.tau.dtype)

        def per_stint(c, ctx, b_s, q_knots, t_col, n_s):
            def latent_fn(t):
                return self.latent(t, ctx)

            h, w = jax.vmap(latent_fn)(batch.tau)
            y_hat = observation(physics, c, h, w, b_s)
            h_c, _, dh_c, dw_c = states_and_rates(latent_fn, t_col)
            q_c = jnp.interp(jnp.minimum(t_col, n_s.astype(t_col.dtype)), knots, q_knots)
            r_t = thermal_residual(physics, c, h_c, dh_c, q_c)
            r_w = wear_residual(physics, c, h_c, dw_c)
            return y_hat, h, w, r_t, r_w, dw_c

        y_hat, h, w, r_t, r_w, dw = jax.vmap(per_stint)(
            batch.compound_idx, batch.ctx, b, batch.q, colloc_tau, batch.n_laps
        )
        terms = {
            "data": masked_mean((y_hat - batch.y) ** 2, batch.mask),
            "thermal": jnp.mean(r_t**2),
            "wear": jnp.mean(r_w**2),
            "mono": jnp.mean(jax.nn.relu(-dw) ** 2),
            "reg": self.weight_norm_sq(),
        }
        if batch.h_true is not None:
            terms["state"] = masked_mean((h - batch.h_true) ** 2 + (w - batch.w_true) ** 2, batch.mask)
        return terms

    def forecast(self, history: StintHistory, horizon: int, scenario: Scenario) -> PredictionOutput:
        """Anchored forecast for stint laps t0+1..t0+horizon (E = 1).

        The M1 latent depends on tau and ctx only (PROJECT section 4.6), so the scenario q_bar does not
        change M1's forecast. It is accepted for the shared Forecaster protocol.
        """
        horizon = validate_horizon(horizon)
        ctx = history.ctx.features()
        c = history.ctx.compound_idx
        tau_future = jnp.arange(history.t0 + 1, history.t0 + horizon + 1, dtype=float)
        g_hist, _, _ = self.shape(c, ctx, history.tau)
        g_future, h_future, w_future = self.shape(c, ctx, tau_future)
        return build_prediction(
            np.asarray(g_hist)[None, :],
            np.asarray(g_future)[None, :],
            history.y,
            float(self.noise_s),
            latent_h=np.asarray(h_future),
            latent_w_s=np.asarray(w_future),
        )

    def fit(self, train: StintBatch, key: jax.Array) -> PINN:
        """Train with apexpinn.train.trainer.train using this model's weights and train_cfg."""
        from apexpinn.train.trainer import train as _train

        return _train(self, train, self.weights, self.train_cfg, key)[0]
