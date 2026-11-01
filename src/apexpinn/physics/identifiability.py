"""Identifiability tooling for the S1 study: profile likelihood, Gauss-Newton spectrum, labels, priors.

See PRD.md section 6.4 (S1b recovery, S1c profiles and spectrum), GLOSSARY.md section 6 (identifiable,
weakly identifiable), and PLAN.md T5 (tooling tests and the declared method).

S1c profiles and spectra use the exact ODE likelihood of the M2 forward model (ode_least_squares), with
the stint offsets b_s concentrated out in closed form. Identifiability is a property of the model family
and the data design, and M1 and M2 share Theta and the observation equation, so this avoids retraining
the M1 network at every profile grid point. Parameters live in log / logit coordinates (ThetaLayout).

The optimizers work in float64 numpy at the scipy boundary. Turn on jax_enable_x64 before calling them,
otherwise JAX evaluates the losses in float32 and the profiles lose precision.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.stats
from jax.typing import ArrayLike

from apexpinn.models.ode_model import ODEModel
from apexpinn.models.stints import StintBatch
from apexpinn.physics.residuals import COMPOUNDS, N_COMPOUNDS, POSITIVITY_FLOOR, PhysicsParams, observation

LossFn = Callable[[jax.Array], jax.Array]
SHAPE_LABELS = ("sharp", "wide", "flat")
SHARP_REL_HALF_WIDTH = 0.5  # GLOSSARY: an interval wider than +-50% of the true value is wide
# Prior ranges for the initialization draw (PLAN T5 method declarations): (low, high, scale).
PRIOR_RANGES: dict[str, tuple[float, float, str]] = {
    "k": (0.1, 1.0, "log"),
    "rho": (0.02, 0.2, "log"),
    "gamma": (1.0, 6.0, "log"),
    "h_opt": (0.4, 0.8, "linear"),
    "kappa": (0.3, 3.0, "log"),
}
_GROUPS = ("k", "rho", "gamma", "h_opt")


def _as_vector(name: str, value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty 1-D vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_grid(grid: ArrayLike, min_points: int = 1) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < min_points:
        raise ValueError(f"grid must be 1-D with at least {min_points} points, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(np.diff(arr) <= 0):
        raise ValueError("grid must be finite and strictly increasing")
    return arr


def _check_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


# --- Gauss-Newton spectrum ---


@dataclass(frozen=True, eq=False)
class Spectrum:
    """Eigen-decomposition of J^T J at theta_hat, where J is the residual Jacobian.

    eigenvalues are sorted descending and are not clipped. relative = max(eigenvalue, 0) / largest is the
    view used for near-zero checks. eigenvectors[:, i] belongs to eigenvalues[i].
    """

    eigenvalues: np.ndarray
    relative: np.ndarray
    eigenvectors: np.ndarray


def gauss_newton_spectrum(residual_fn: LossFn, theta_hat: ArrayLike, *, mode: str = "fwd") -> Spectrum:
    """Spectrum of the Gauss-Newton matrix J^T J, with J = d residual / d theta.

    mode='fwd' builds J with jax.jacfwd (cheap for few parameters). mode='rev' uses jax.jacrev, which the
    exact ODE likelihood needs: ode_least_squares solves with diffrax.RecursiveCheckpointAdjoint, which
    supports reverse-mode autodiff only. A relative eigenvalue near zero marks a direction in theta that
    the data cannot resolve.
    """
    if mode not in ("fwd", "rev"):
        raise ValueError(f"mode must be 'fwd' or 'rev', got {mode!r}")
    theta = _as_vector("theta_hat", theta_hat)
    jacobian = jax.jacfwd if mode == "fwd" else jax.jacrev
    jac = np.asarray(jacobian(residual_fn)(jnp.asarray(theta)), dtype=np.float64)
    jac = jac.reshape(-1, theta.shape[0])
    if not np.all(np.isfinite(jac)):
        raise ValueError("residual_fn has a non-finite Jacobian at theta_hat")
    eigenvalues, eigenvectors = np.linalg.eigh(jac.T @ jac)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    clipped = np.maximum(eigenvalues, 0.0)
    largest = clipped[0]
    relative = clipped / largest if largest > 0 else np.zeros_like(clipped)
    return Spectrum(eigenvalues=eigenvalues.copy(), relative=relative, eigenvectors=eigenvectors.copy())


# --- optimization ---


ValueAndGrad = Callable[[jax.Array], tuple[jax.Array, jax.Array]]


def _lbfgs(
    value_and_grad: ValueAndGrad,
    x0: np.ndarray,
    maxiter: int,
    gtol: float,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
) -> tuple[np.ndarray, float, bool]:
    """scipy L-BFGS-B on a JAX value_and_grad(x), float64 numpy in and out.

    bounds[i] = (v, v) holds entry i fixed at v: L-BFGS-B projects its gradient to zero.
    """

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(x))
        return float(value), np.asarray(grad, dtype=np.float64)

    res = scipy.optimize.minimize(
        fun, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "gtol": gtol}
    )
    return np.asarray(res.x, dtype=np.float64), float(res.fun), bool(res.success)


def _check_options(maxiter: int, gtol: float) -> tuple[int, float]:
    maxiter = _check_positive_int("maxiter", maxiter)
    if not (np.isfinite(gtol) and gtol > 0):
        raise ValueError(f"gtol must be finite and > 0, got {gtol!r}")
    return maxiter, float(gtol)


def _value_and_grad(loss_fn: LossFn, value_and_grad: ValueAndGrad | None) -> ValueAndGrad:
    """The caller's jitted value_and_grad (reused across calls, compiled once), or a new one."""
    if value_and_grad is None:
        return jax.jit(jax.value_and_grad(loss_fn))
    if not callable(value_and_grad):
        raise ValueError(f"value_and_grad must be callable or None, got {type(value_and_grad).__name__}")
    return value_and_grad


def minimize(
    loss_fn: LossFn,
    theta0: ArrayLike,
    *,
    maxiter: int = 500,
    gtol: float = 1e-8,
    value_and_grad: ValueAndGrad | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Minimize a scalar JAX loss with L-BFGS-B on a jitted value_and_grad.

    value_and_grad, if given, must be jax.jit(jax.value_and_grad(loss_fn)); passing the same object to
    several calls reuses one compilation. Returns (theta, loss, success) with theta as a float64 vector.
    """
    theta0 = _as_vector("theta0", theta0)
    maxiter, gtol = _check_options(maxiter, gtol)
    return _lbfgs(_value_and_grad(loss_fn, value_and_grad), theta0, maxiter, gtol)


@dataclass(frozen=True, eq=False)
class ProfileResult:
    """Profile of one parameter: loss[i] is the minimum over the other entries at grid[i], at thetas[i]."""

    grid: np.ndarray  # (G,)
    loss: np.ndarray  # (G,)
    thetas: np.ndarray  # (G, n) full vectors, thetas[:, index] == grid
    success: np.ndarray  # (G,) bool, optimizer success per grid point


def profile_likelihood(
    loss_fn: LossFn,
    theta_hat: ArrayLike,
    index: int,
    grid: ArrayLike,
    *,
    maxiter: int = 500,
    gtol: float = 1e-8,
    value_and_grad: ValueAndGrad | None = None,
) -> ProfileResult:
    """Profile loss of theta[index]: fix it at each grid value and minimize over the other entries.

    Grid points are visited from the one nearest theta_hat[index] outward in both directions, and each
    solve starts from its neighbour's solution, so the path follows one valley of the loss.

    Each solve optimizes the full-length vector with theta[index] held by equal L-BFGS-B bounds
    (grid[i], grid[i]). The loss and its gradient therefore have one signature for every index and grid
    point, so a value_and_grad passed by the caller (jax.jit(jax.value_and_grad(loss_fn))) is compiled
    once and reused for every profiled parameter and for the MLE.
    """
    theta_hat = _as_vector("theta_hat", theta_hat)
    n = theta_hat.shape[0]
    if isinstance(index, bool) or not isinstance(index, int | np.integer) or not 0 <= index < n:
        raise ValueError(f"index must be an integer in [0, {n}), got {index!r}")
    index = int(index)
    grid = _as_grid(grid)
    maxiter, gtol = _check_options(maxiter, gtol)
    vg = _value_and_grad(loss_fn, value_and_grad)
    n_grid = grid.shape[0]
    losses = np.full(n_grid, np.nan)
    thetas = np.zeros((n_grid, n))
    success = np.zeros(n_grid, dtype=bool)
    free_bounds: list[tuple[float | None, float | None]] = [(None, None)] * n

    def solve(i: int, theta0: np.ndarray) -> np.ndarray:
        x0 = theta0.copy()
        x0[index] = grid[i]
        if n == 1:
            theta, loss, ok = x0, float(vg(jnp.asarray(x0))[0]), True
        else:
            bounds = list(free_bounds)
            bounds[index] = (float(grid[i]), float(grid[i]))
            theta, loss, ok = _lbfgs(vg, x0, maxiter, gtol, bounds)
            theta[index] = grid[i]
        losses[i], success[i] = loss, ok
        thetas[i] = theta
        return theta

    start = int(np.argmin(np.abs(grid - theta_hat[index])))
    theta_start = solve(start, theta_hat)
    theta = theta_start
    for i in range(start + 1, n_grid):
        theta = solve(i, theta)
    theta = theta_start
    for i in range(start - 1, -1, -1):
        theta = solve(i, theta)
    return ProfileResult(grid=grid, loss=losses, thetas=thetas, success=success)


# --- deviance, intervals, labels ---


def sse_deviance(sse: ArrayLike, n_obs: int, sse_min: float | None = None) -> np.ndarray:
    """Profile deviance n_obs * log(sse / sse_min).

    This is -2 log of the profile likelihood ratio for Gaussian noise with unknown sigma: the noise
    variance is profiled out as SSE / n_obs, so -2 log L = n_obs * log(SSE) + const. It is chi2(1)
    distributed near the truth. sse_min defaults to min(sse); pass the global MLE SSE when it is known.
    """
    arr = np.asarray(sse, dtype=np.float64)
    if arr.size < 1 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("sse must be non-empty, finite, and > 0")
    n_obs = _check_positive_int("n_obs", n_obs)
    if sse_min is None:
        sse_min = float(np.min(arr))
    elif not (np.isfinite(sse_min) and sse_min > 0):
        raise ValueError(f"sse_min must be finite and > 0, got {sse_min!r}")
    return n_obs * np.log(arr / float(sse_min))


@dataclass(frozen=True)
class Interval:
    """Likelihood-based interval. An unbounded side reports the grid edge and has *_bounded = False."""

    lo: float
    hi: float
    lo_bounded: bool
    hi_bounded: bool
    threshold: float

    @property
    def bounded(self) -> bool:
        return self.lo_bounded and self.hi_bounded


def likelihood_interval(grid: ArrayLike, deviance: ArrayLike, level: float = 0.95) -> Interval:
    """Interval where the deviance stays at or below chi2.ppf(level, 1) (3.841 at 95%).

    The interval is the connected region of grid points around the minimum. Each bounded endpoint is the
    linear interpolation of the threshold crossing between the last point inside and the first outside.
    If even the minimum exceeds the threshold, the interval collapses to the minimum grid point.
    """
    if not (np.isfinite(level) and 0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    grid = _as_grid(grid, min_points=2)
    dev = np.asarray(deviance, dtype=np.float64)
    if dev.shape != grid.shape or not np.all(np.isfinite(dev)):
        raise ValueError(f"deviance must be finite with the grid shape {grid.shape}, got {dev.shape}")
    threshold = float(scipy.stats.chi2.ppf(level, 1))
    i_min = int(np.argmin(dev))
    if dev[i_min] > threshold:
        x = float(grid[i_min])
        return Interval(lo=x, hi=x, lo_bounded=True, hi_bounded=True, threshold=threshold)

    j = i_min
    while j + 1 < grid.shape[0] and dev[j + 1] <= threshold:
        j += 1
    if j == grid.shape[0] - 1:
        hi, hi_bounded = float(grid[-1]), False
    else:
        step = (grid[j + 1] - grid[j]) / (dev[j + 1] - dev[j])
        hi, hi_bounded = float(grid[j] + (threshold - dev[j]) * step), True

    j = i_min
    while j - 1 >= 0 and dev[j - 1] <= threshold:
        j -= 1
    if j == 0:
        lo, lo_bounded = float(grid[0]), False
    else:
        step = (grid[j] - grid[j - 1]) / (dev[j - 1] - dev[j])
        lo, lo_bounded = float(grid[j] - (threshold - dev[j]) * step), True
    return Interval(lo=lo, hi=hi, lo_bounded=lo_bounded, hi_bounded=hi_bounded, threshold=threshold)


def profile_shape_label(interval: Interval, truth: float) -> str:
    """'flat' if unbounded on either side, 'sharp' if inside [0.5, 1.5] x truth, else 'wide'.

    interval and truth must be in the same (natural, positive) units: convert log / logit profile
    endpoints back with ThetaLayout.constrained or exp / sigmoid first.
    """
    if not (np.isfinite(truth) and truth > 0):
        raise ValueError(f"truth must be finite and > 0, got {truth!r}")
    if not interval.bounded:
        return "flat"
    low, high = (1.0 - SHARP_REL_HALF_WIDTH) * truth, (1.0 + SHARP_REL_HALF_WIDTH) * truth
    return "sharp" if low <= interval.lo and interval.hi <= high else "wide"


def identifiability_label(recovered: Sequence[bool] | bool | None, shapes: Sequence[str]) -> str:
    """GLOSSARY section 6 label from S1b recovery and S1c profile shapes.

    'identifiable' if every seed recovered (within the O6 tolerance) and every shape is 'sharp';
    'non_identifiable' if at least half of the shapes are 'flat'; else 'weakly_identifiable'.
    recovered=None (no O6 tolerance for this parameter) labels from the shapes only.
    """
    shapes = list(shapes)
    if not shapes or any(s not in SHAPE_LABELS for s in shapes):
        raise ValueError(f"shapes must be a non-empty sequence of {SHAPE_LABELS}, got {shapes!r}")
    if recovered is not None:
        recovered = [recovered] if isinstance(recovered, bool | np.bool_) else list(recovered)
        if not recovered:
            raise ValueError("recovered must be None or a non-empty sequence of bools")
    if 2 * shapes.count("flat") >= len(shapes):
        return "non_identifiable"
    if all(s == "sharp" for s in shapes) and (recovered is None or all(bool(r) for r in recovered)):
        return "identifiable"
    return "weakly_identifiable"


# --- priors and parameter layout ---


def sample_prior_physics(key: jax.Array) -> PhysicsParams:
    """Initialization draw from PRIOR_RANGES, independent per compound (all 5) and per parameter."""
    keys = jax.random.split(key, len(PRIOR_RANGES))
    values: dict[str, np.ndarray] = {}
    for sub, (name, (low, high, scale)) in zip(keys, PRIOR_RANGES.items(), strict=True):
        shape = () if name == "kappa" else (N_COMPOUNDS,)
        u = np.asarray(jax.random.uniform(sub, shape), dtype=np.float64)
        if scale == "log":
            draw = np.exp(np.log(low) + u * (np.log(high) - np.log(low)))
        else:
            draw = low + u * (high - low)
        values[name] = np.clip(draw, low, high)
    return PhysicsParams.from_constrained(**values)


def _inverse_softplus(x: jax.Array) -> jax.Array:
    return x + jnp.log(-jnp.expm1(-x))


@dataclass(frozen=True)
class ThetaLayout:
    """Vector coordinates of Theta for the compounds in use.

    Order: log k_c, log rho_c, log gamma_c, logit h_opt_c (each over compound_idx), then log kappa.
    Compounds outside compound_idx keep the values of the base PhysicsParams.
    """

    compound_idx: tuple[int, ...]

    def __post_init__(self) -> None:
        idx = tuple(self.compound_idx)
        ok = all(not isinstance(i, bool) and isinstance(i, int | np.integer) for i in idx)
        if not idx or not ok or len(set(idx)) != len(idx) or not all(0 <= i < N_COMPOUNDS for i in idx):
            raise ValueError(f"compound_idx must be distinct integers in [0, {N_COMPOUNDS}), got {idx!r}")
        object.__setattr__(self, "compound_idx", tuple(int(i) for i in idx))

    @classmethod
    def from_compounds(cls, compounds: Sequence[str]) -> ThetaLayout:
        unknown = [c for c in compounds if c not in COMPOUNDS]
        if unknown:
            raise ValueError(f"compounds must be names from {COMPOUNDS}, got unknown {unknown}")
        return cls(tuple(COMPOUNDS.index(c) for c in compounds))

    @property
    def names(self) -> tuple[str, ...]:
        per = tuple(f"{g}_c[{COMPOUNDS[i]}]" for g in _GROUPS for i in self.compound_idx)
        return (*per, "kappa")

    @property
    def size(self) -> int:
        return len(_GROUPS) * len(self.compound_idx) + 1

    def to_vector(self, params: PhysicsParams) -> np.ndarray:
        """Coordinates of params (numpy float64, not differentiable)."""
        idx = list(self.compound_idx)
        k, rho, gamma, h_opt = (np.asarray(getattr(params, g), dtype=np.float64)[idx] for g in _GROUPS)
        kappa = np.asarray(params.kappa, dtype=np.float64).reshape(1)
        logit_h = np.log(h_opt) - np.log1p(-h_opt)
        return np.concatenate([np.log(k), np.log(rho), np.log(gamma), logit_h, np.log(kappa)])

    def _check_v(self, v: ArrayLike) -> jax.Array:
        v = jnp.asarray(v)
        if v.shape != (self.size,):
            raise ValueError(f"v must have shape ({self.size},), got {v.shape}")
        return v

    def to_physics(self, v: ArrayLike, base: PhysicsParams) -> PhysicsParams:
        """PhysicsParams with the used compounds set from v. Differentiable in v (JAX)."""
        if not isinstance(base, PhysicsParams):
            raise ValueError(f"base must be PhysicsParams, got {type(base).__name__}")
        v = self._check_v(v)
        m = len(self.compound_idx)
        idx = jnp.asarray(self.compound_idx)
        log_k, log_rho, log_gamma, logit_h = (v[i * m : (i + 1) * m] for i in range(len(_GROUPS)))
        return PhysicsParams(
            raw_k=base.raw_k.at[idx].set(_inverse_softplus(jnp.exp(log_k) - POSITIVITY_FLOOR)),
            raw_rho=base.raw_rho.at[idx].set(_inverse_softplus(jnp.exp(log_rho) - POSITIVITY_FLOOR)),
            raw_gamma=base.raw_gamma.at[idx].set(_inverse_softplus(jnp.exp(log_gamma))),
            raw_h_opt=base.raw_h_opt.at[idx].set(logit_h),
            raw_kappa=_inverse_softplus(jnp.exp(v[-1])).astype(base.raw_kappa.dtype),
        )

    def constrained(self, v: ArrayLike) -> dict[str, float]:
        """name -> physical value (exp of log entries, sigmoid of logit entries)."""
        v = np.asarray(self._check_v(v), dtype=np.float64)
        m = len(self.compound_idx)
        values = np.exp(v)
        values[3 * m : 4 * m] = 1.0 / (1.0 + np.exp(-v[3 * m : 4 * m]))
        return {name: float(x) for name, x in zip(self.names, values, strict=True)}


# --- exact ODE likelihood (M2 forward model) ---


def ode_least_squares(
    batch: StintBatch, layout: ThetaLayout, base: PhysicsParams, *, rtol: float = 1e-6, atol: float = 1e-6
) -> tuple[LossFn, LossFn, int]:
    """(sse_fn, residual_fn, n_obs) of the M2 forward model on batch, in layout coordinates.

    The shape g is residuals.observation with b = 0 on ODEModel.solve states at batch.tau. Each stint
    offset is concentrated out in closed form: b_s = masked mean of (y - g) over the scored laps.
    residual_fn returns where(mask, y - g - b, 0) flattened to (n * K,); sse_fn is its sum of squares.
    """
    ODEModel.init(rtol=rtol, atol=atol)  # validates the tolerances once, outside any trace
    mask_np = np.asarray(batch.mask, dtype=bool)
    n_obs = int(mask_np.sum())
    if n_obs < 1:
        raise ValueError("batch.mask must contain at least one scored lap")
    mask = jnp.asarray(mask_np)
    y = jnp.where(mask, batch.y, 0.0)
    counts = jnp.maximum(mask.sum(axis=1), 1).astype(y.dtype)  # a stint with no scored lap gets b_s = 0

    def residual_fn(v: jax.Array) -> jax.Array:
        physics = layout.to_physics(v, base)
        model = ODEModel.init(physics=physics, rtol=rtol, atol=atol)
        h, w = jax.vmap(lambda c, q: model.solve(c, q, batch.tau))(batch.compound_idx, batch.q)
        g = observation(physics, batch.compound_idx[:, None], h, w, 0.0)
        r = jnp.where(mask, y - g, 0.0)
        b = r.sum(axis=1) / counts
        return jnp.where(mask, r - b[:, None], 0.0).reshape(-1)

    def sse_fn(v: jax.Array) -> jax.Array:
        return jnp.sum(residual_fn(v) ** 2)

    return sse_fn, residual_fn, n_obs


@dataclass(frozen=True, eq=False)
class MLEResult:
    """Best of several L-BFGS-B starts. start_sse[i] and start_theta[i] are the final values of start i."""

    theta: np.ndarray
    sse: float
    start_sse: list[float]
    start_theta: list[np.ndarray]
    best_index: int
    success: bool


def ode_mle(
    sse_fn: LossFn,
    starts: Sequence[ArrayLike],
    *,
    maxiter: int = 500,
    gtol: float = 1e-8,
    value_and_grad: ValueAndGrad | None = None,
) -> MLEResult:
    """Minimize sse_fn from each start and keep the lowest finite SSE.

    All starts share one jitted value_and_grad: the caller's (jax.jit(jax.value_and_grad(sse_fn)), so the
    profiles of the same dataset reuse its compilation) or a new one.
    """
    starts = [_as_vector(f"starts[{i}]", s) for i, s in enumerate(starts)]
    if not starts:
        raise ValueError("starts must contain at least one start vector")
    if len({s.shape for s in starts}) != 1:
        raise ValueError("starts must all have the same length")
    maxiter, gtol = _check_options(maxiter, gtol)
    vg = _value_and_grad(sse_fn, value_and_grad)
    results = [_lbfgs(vg, s, maxiter, gtol) for s in starts]
    start_sse = [r[1] for r in results]
    finite = [s if np.isfinite(s) else np.inf for s in start_sse]
    best = int(np.argmin(finite))
    theta, sse, success = results[best]
    return MLEResult(
        theta=theta,
        sse=sse,
        start_sse=start_sse,
        start_theta=[r[0] for r in results],
        best_index=best,
        success=success,
    )
