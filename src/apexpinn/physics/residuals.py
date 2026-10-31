"""Gauge-fixed physics parameters, ODE residuals, and the shared observation equation.

See PROJECT.md sections 4.3 (gauge fixing), 4.4 (equations), and 4.6 (autodiff rates).
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

COMPOUNDS = ("C1", "C2", "C3", "C4", "C5")
N_COMPOUNDS = len(COMPOUNDS)
# k_c and rho_c are floor + softplus(raw). Training that drives them onto the floor is parameter collapse
# (GLOSSARY). physics.diagnostics.detect_collapse reports it instead of hiding it.
POSITIVITY_FLOOR = 1e-4


def _inverse_softplus(x: np.ndarray) -> np.ndarray:
    return x + np.log(-np.expm1(-x))


def _per_compound(name: str, value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(N_COMPOUNDS, float(arr))
    if arr.shape != (N_COMPOUNDS,):
        raise ValueError(f"{name} must be a scalar or have shape ({N_COMPOUNDS},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _check_per_compound(name: str, arr: np.ndarray, ok: np.ndarray, rule: str) -> None:
    bad = [f"{COMPOUNDS[i]}={arr[i]}" for i in np.flatnonzero(~ok)]
    if bad:
        raise ValueError(f"{name} must be {rule}, got {', '.join(bad)}")


class PhysicsParams(eqx.Module):
    """Unconstrained physics parameters Theta. Read the constrained values through the properties.

    Per compound (shape (5,), index 0..4 = C1..C5): k_c, rho_c (floor + softplus), gamma_c (softplus),
    h_opt_c (sigmoid). Shared scalar: kappa (softplus).
    """

    raw_k: jax.Array
    raw_rho: jax.Array
    raw_gamma: jax.Array
    raw_h_opt: jax.Array
    raw_kappa: jax.Array

    @property
    def k(self) -> jax.Array:
        """Warm-up rate per lap, > POSITIVITY_FLOOR."""
        return POSITIVITY_FLOOR + jax.nn.softplus(self.raw_k)

    @property
    def rho(self) -> jax.Array:
        """Base wear rate in s/lap, > POSITIVITY_FLOOR."""
        return POSITIVITY_FLOOR + jax.nn.softplus(self.raw_rho)

    @property
    def gamma(self) -> jax.Array:
        """Thermal penalty coefficient in s, >= 0."""
        return jax.nn.softplus(self.raw_gamma)

    @property
    def h_opt(self) -> jax.Array:
        """Optimal thermal state, in (0, 1)."""
        return jax.nn.sigmoid(self.raw_h_opt)

    @property
    def kappa(self) -> jax.Array:
        """Overheating wear multiplier, >= 0."""
        return jax.nn.softplus(self.raw_kappa)

    @classmethod
    def from_constrained(
        cls,
        k: ArrayLike,
        rho: ArrayLike,
        gamma: ArrayLike,
        h_opt: ArrayLike,
        kappa: ArrayLike,
    ) -> PhysicsParams:
        """Build from physical values. Per-compound values may be a scalar (broadcast) or shape (5,)."""
        k_arr = _per_compound("k", k)
        rho_arr = _per_compound("rho", rho)
        gamma_arr = _per_compound("gamma", gamma)
        h_opt_arr = _per_compound("h_opt", h_opt)
        kappa_arr = np.asarray(kappa, dtype=np.float64)
        if kappa_arr.ndim != 0 or not np.isfinite(kappa_arr) or kappa_arr <= 0:
            raise ValueError(f"kappa must be a finite scalar > 0, got {kappa!r}")
        floor_rule = f"> POSITIVITY_FLOOR ({POSITIVITY_FLOOR})"
        _check_per_compound("k", k_arr, k_arr > POSITIVITY_FLOOR, floor_rule)
        _check_per_compound("rho", rho_arr, rho_arr > POSITIVITY_FLOOR, floor_rule)
        _check_per_compound("gamma", gamma_arr, gamma_arr > 0, "> 0")
        _check_per_compound("h_opt", h_opt_arr, (h_opt_arr > 0) & (h_opt_arr < 1), "in (0, 1)")
        def as_jax(x: np.ndarray) -> jax.Array:
            return jnp.asarray(x, dtype=float)  # float32 or float64, following jax_enable_x64

        return cls(
            raw_k=as_jax(_inverse_softplus(k_arr - POSITIVITY_FLOOR)),
            raw_rho=as_jax(_inverse_softplus(rho_arr - POSITIVITY_FLOOR)),
            raw_gamma=as_jax(_inverse_softplus(gamma_arr)),
            raw_h_opt=as_jax(np.log(h_opt_arr) - np.log1p(-h_opt_arr)),
            raw_kappa=as_jax(_inverse_softplus(kappa_arr)),
        )


def states_and_rates(
    latent_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]], tau: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate (h, w, dh/dtau, dw/dtau) at each point of a 1-D tau grid.

    latent_fn maps a scalar tau to scalar (h, w). Rates use forward-mode autodiff (jax.jvp), vmapped over tau,
    so non-integer collocation points need no finite differences.
    """
    tau = jnp.asarray(tau, dtype=float)
    if tau.ndim != 1:
        raise ValueError(f"tau must be 1-D, got shape {tau.shape}")

    def one(t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        (h, w), (dh, dw) = jax.jvp(latent_fn, (t,), (jnp.ones_like(t),))
        return h, w, dh, dw

    return jax.vmap(one)(tau)


def thermal_rhs(params: PhysicsParams, compound_idx: ArrayLike, h: ArrayLike, q: ArrayLike) -> jax.Array:
    """dh/dtau implied by the thermal ODE: k_c * (q - h)."""
    return params.k[compound_idx] * (q - h)


def wear_rhs(params: PhysicsParams, compound_idx: ArrayLike, h: ArrayLike) -> jax.Array:
    """dw/dtau implied by the wear ODE: rho_c * (1 + kappa * ReLU(h - h_opt_c))."""
    return params.rho[compound_idx] * (1.0 + params.kappa * jax.nn.relu(h - params.h_opt[compound_idx]))


def thermal_residual(
    params: PhysicsParams, compound_idx: ArrayLike, h: ArrayLike, dh: ArrayLike, q: ArrayLike
) -> jax.Array:
    """r_T = dh/dtau - k_c * (q - h)."""
    return dh - thermal_rhs(params, compound_idx, h, q)


def wear_residual(params: PhysicsParams, compound_idx: ArrayLike, h: ArrayLike, dw: ArrayLike) -> jax.Array:
    """r_W = dw/dtau - rho_c * (1 + kappa * ReLU(h - h_opt_c))."""
    return dw - wear_rhs(params, compound_idx, h)


def observation(
    params: PhysicsParams, compound_idx: ArrayLike, h: ArrayLike, w: ArrayLike, b: ArrayLike
) -> jax.Array:
    """Predicted target y_hat = b + gamma_c * (h - h_opt_c)^2 + w, in seconds. Shared by M1 and M2."""
    return b + params.gamma[compound_idx] * (h - params.h_opt[compound_idx]) ** 2 + w
