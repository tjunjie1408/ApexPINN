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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
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
# Optimizer box for the S1c ODE likelihood, natural units (low, high). PRIOR_RANGES widened 10x on each
# side in log; h_opt matches s1_study.H_OPT_GRID_CLIP. The box keeps L-BFGS-B line searches out of the
# stiff region of very large k_c, where explicit Tsit5 runs out of max_steps.
S1C_BOUNDS: dict[str, tuple[float, float]] = {
    "k": (0.01, 10.0),
    "rho": (0.002, 2.0),
    "gamma": (0.1, 60.0),
    "h_opt": (0.02, 0.98),
    "kappa": (0.03, 30.0),
}
# A non-finite evaluation inside an L-BFGS-B solve is answered with f_ref + FAIL_PENALTY * (1 + |f_ref|)
# and a zero gradient, where f_ref is the first finite value of the solve, so the line search backtracks.
FAIL_PENALTY = 1e6
_GROUPS = ("k", "rho", "gamma", "h_opt")
Bounds = Sequence[tuple[float | None, float | None]]


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


@dataclass(frozen=True, eq=False)
class _Solve:
    """One L-BFGS-B solve. failed: the first evaluation was not finite, or a later one was not finite and
    no evaluation after the first one was (loss NaN, success False). This includes a final iterate equal
    to the start, restored by scipy after its first line search found only non-finite trials.

    A solve whose final iterate was not finite but which saw a finite evaluation after the first returns
    the best finite evaluation with success False and failed False (see _lbfgs).
    """

    theta: np.ndarray
    loss: float
    success: bool
    failed: bool
    n_failed_evals: int


class _StartFailed(Exception):
    """The first evaluation of a solve was not finite, so there is no f_ref to build a penalty from."""


def _check_bounds(bounds: Bounds | None, n: int) -> list[tuple[float | None, float | None]]:
    """bounds as a list of n (lo, hi) pairs (None = free side); None means all coordinates free."""
    if bounds is None:
        return [(None, None)] * n
    try:
        pairs = [tuple(b) for b in bounds]
    except TypeError:
        raise ValueError("bounds must be a sequence of (lo, hi) pairs") from None
    if len(pairs) != n or any(len(p) != 2 for p in pairs):
        raise ValueError(f"bounds must have {n} (lo, hi) pairs, got {len(pairs)}")
    out: list[tuple[float | None, float | None]] = []
    for i, (lo, hi) in enumerate(pairs):
        side = [None if v is None else float(v) for v in (lo, hi)]
        if any(v is not None and not np.isfinite(v) for v in side):
            raise ValueError(f"bounds[{i}] must be finite or None, got {(lo, hi)!r}")
        if side[0] is not None and side[1] is not None and side[0] > side[1]:
            raise ValueError(f"bounds[{i}] must have lo <= hi, got {(lo, hi)!r}")
        out.append((side[0], side[1]))
    return out


def _clip_into(x: np.ndarray, bounds: Bounds | None) -> np.ndarray:
    if bounds is None:
        return x.copy()
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], dtype=np.float64)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], dtype=np.float64)
    return np.clip(x, lo, hi)


def _lbfgs(
    value_and_grad: ValueAndGrad,
    x0: np.ndarray,
    maxiter: int,
    gtol: float,
    bounds: Bounds | None = None,
) -> _Solve:
    """scipy L-BFGS-B on a JAX value_and_grad(x), float64 numpy in and out.

    bounds[i] = (v, v) holds entry i fixed at v: L-BFGS-B projects its gradient to zero. x0 is clipped
    into the bounds first.

    An evaluation fails when its value or any gradient entry is non-finite (for example an ODE solve that
    hit max_steps). It is answered with f_ref + FAIL_PENALTY * (1 + |f_ref|) and a zero gradient, where
    f_ref is the first finite value of this solve, so the line search backtracks. If the first evaluation
    fails the solve stops at x0 with loss NaN.

    The final iterate can still be a penalty point (the line search may end on a warning at its last
    trial, and the zero gradient then passes the gtol test), or scipy can restore an earlier iterate
    while res.fun keeps the last evaluated value. The result therefore uses the values recorded here, not
    res.fun. A solve that saw a non-finite evaluation and no finite evaluation after the first one is
    failed (loss NaN, success False), whether its final iterate is a penalty point or x0 restored by
    scipy after every trial of the first line search failed. Otherwise, if the final iterate has no
    finite recorded value, the solve falls back to its best finite evaluation with success False.
    Evaluations are counted, not positions: a later finite evaluation at the same x as the start counts.
    """
    x0 = _clip_into(x0, bounds)
    f_ref: list[float] = []  # first finite value of this solve
    finite_values: dict[bytes, float] = {}
    best: list[tuple[float, np.ndarray]] = []  # (value, x) of the lowest finite evaluation so far
    n_finite = 0
    n_failed = 0

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal n_failed, n_finite
        x = np.array(x, dtype=np.float64)  # scipy updates its x in place
        value, grad = value_and_grad(jnp.asarray(x))
        value, grad = float(value), np.asarray(grad, dtype=np.float64)
        if np.isfinite(value) and np.all(np.isfinite(grad)):
            n_finite += 1
            finite_values[x.tobytes()] = value
            if not f_ref:
                f_ref.append(value)
            if not best or value < best[0][0]:
                best[:] = [(value, x)]
            return value, grad
        n_failed += 1
        if not f_ref:
            raise _StartFailed
        return f_ref[0] + FAIL_PENALTY * (1.0 + abs(f_ref[0])), np.zeros(x.shape, dtype=np.float64)

    try:
        res = scipy.optimize.minimize(
            fun, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "gtol": gtol}
        )
    except _StartFailed:
        return _Solve(theta=x0, loss=float("nan"), success=False, failed=True, n_failed_evals=n_failed)
    theta = np.asarray(res.x, dtype=np.float64)
    key = theta.tobytes()
    if n_finite < 2 and (n_failed > 0 or key not in finite_values):
        # Only the start was finite: the final iterate is a penalty point, or scipy restored x0 after
        # every trial of its first line search failed. A clean stop at x0 (n_failed == 0) is not failed.
        return _Solve(theta=theta, loss=float("nan"), success=False, failed=True, n_failed_evals=n_failed)
    if key in finite_values:
        loss = finite_values[key]
        success = bool(res.success) and loss == float(res.fun)
        return _Solve(theta=theta, loss=loss, success=success, failed=False, n_failed_evals=n_failed)
    value, x_best = best[0]
    return _Solve(theta=x_best.copy(), loss=value, success=False, failed=False, n_failed_evals=n_failed)


def _check_options(maxiter: int, gtol: float) -> tuple[int, float]:
    maxiter = _check_positive_int("maxiter", maxiter)
    if not (np.isfinite(gtol) and gtol > 0):
        raise ValueError(f"gtol must be finite and > 0, got {gtol!r}")
    return maxiter, float(gtol)


def _value_and_grad(loss_fn: LossFn, value_and_grad: ValueAndGrad | None) -> ValueAndGrad:
    """The caller's jitted value_and_grad (reused across calls, compiled once), or a new one.

    The new one uses eqx.filter_jit, so equinox runtime errors raised inside carry readable messages.
    """
    if value_and_grad is None:
        return eqx.filter_jit(jax.value_and_grad(loss_fn))
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

    value_and_grad, if given, must be a jitted jax.value_and_grad(loss_fn) (eqx.filter_jit or jax.jit);
    passing the same object to several calls reuses one compilation. Returns (theta, loss, success) with
    theta as a float64 vector. Non-finite evaluations are handled as in _lbfgs: a solve that ends on a
    non-finite iterate returns its best finite evaluation with success False, or loss NaN and success
    False when it has none after the start, instead of raising.
    """
    theta0 = _as_vector("theta0", theta0)
    maxiter, gtol = _check_options(maxiter, gtol)
    result = _lbfgs(_value_and_grad(loss_fn, value_and_grad), theta0, maxiter, gtol)
    return result.theta, result.loss, result.success


@dataclass(frozen=True, eq=False)
class ProfileResult:
    """Profile of one parameter: loss[i] is the minimum over the other entries at grid[i], at thetas[i].

    failed[i]: the solve at grid[i] found no finite evaluation after its start (or its start was not
    finite), so loss[i] is NaN and success[i] is False. A solve that ended on a non-finite iterate after
    other finite evaluations keeps its best finite one, with success[i] False and failed[i] False.
    n_failed_evals[i] counts the non-finite evaluations seen during that solve (including any the line
    search backtracked from).
    """

    grid: np.ndarray  # (G,)
    loss: np.ndarray  # (G,)
    thetas: np.ndarray  # (G, n) full vectors, thetas[:, index] == grid
    success: np.ndarray  # (G,) bool, optimizer success per grid point
    failed: np.ndarray  # (G,) bool, no finite evaluation after the start per grid point
    n_failed_evals: np.ndarray  # (G,) int


class ProfilePointError(RuntimeError):
    """An exception raised while solving one profile grid point, with the points solved before it.

    partial is a ProfileResult: unsolved entries have loss NaN, thetas NaN, success False, failed False.
    The original exception is __cause__.
    """

    def __init__(self, message: str, partial: ProfileResult) -> None:
        super().__init__(message)
        self.partial = partial


def profile_likelihood(
    loss_fn: LossFn,
    theta_hat: ArrayLike,
    index: int,
    grid: ArrayLike,
    *,
    maxiter: int = 500,
    gtol: float = 1e-8,
    value_and_grad: ValueAndGrad | None = None,
    bounds: Bounds | None = None,
) -> ProfileResult:
    """Profile loss of theta[index]: fix it at each grid value and minimize over the other entries.

    Grid points are visited from the one nearest theta_hat[index] outward in both directions, and each
    solve starts from its neighbour's solution, so the path follows one valley of the loss. A failed
    point (no finite evaluation after its start, see ProfileResult) gets loss NaN, success False and
    failed True, and the next point of the walk starts from the last non-failed solution (theta_hat for
    the first point).

    Each solve optimizes the full-length vector with theta[index] held by equal L-BFGS-B bounds
    (grid[i], grid[i]). The loss and its gradient therefore have one signature for every index and grid
    point, so a value_and_grad passed by the caller (a jitted jax.value_and_grad(loss_fn)) is compiled
    once and reused for every profiled parameter and for the MLE.

    bounds holds one (lo, hi) pair per coordinate (None = free side; default all free). The free
    coordinates stay inside them; the profiled index is pinned to grid[i], which may lie outside its box.

    An exception raised while solving grid point i is re-raised as ProfilePointError naming index, i and
    grid[i], with the points solved so far in its partial attribute.
    """
    theta_hat = _as_vector("theta_hat", theta_hat)
    n = theta_hat.shape[0]
    if isinstance(index, bool) or not isinstance(index, int | np.integer) or not 0 <= index < n:
        raise ValueError(f"index must be an integer in [0, {n}), got {index!r}")
    index = int(index)
    grid = _as_grid(grid)
    maxiter, gtol = _check_options(maxiter, gtol)
    free_bounds = _check_bounds(bounds, n)
    vg = _value_and_grad(loss_fn, value_and_grad)
    n_grid = grid.shape[0]
    losses = np.full(n_grid, np.nan)
    thetas = np.full((n_grid, n), np.nan)
    success = np.zeros(n_grid, dtype=bool)
    failed = np.zeros(n_grid, dtype=bool)
    n_failed_evals = np.zeros(n_grid, dtype=int)

    def result() -> ProfileResult:
        return ProfileResult(
            grid=grid.copy(),
            loss=losses.copy(),
            thetas=thetas.copy(),
            success=success.copy(),
            failed=failed.copy(),
            n_failed_evals=n_failed_evals.copy(),
        )

    def solve_point(i: int, theta0: np.ndarray) -> _Solve:
        bounds_i = list(free_bounds)
        bounds_i[index] = (float(grid[i]), float(grid[i]))
        x0 = _clip_into(theta0, bounds_i)
        x0[index] = grid[i]
        if n == 1:
            value, grad = vg(jnp.asarray(x0))
            ok = bool(np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad))))
            loss = float(value) if ok else float("nan")
            return _Solve(theta=x0, loss=loss, success=ok, failed=not ok, n_failed_evals=int(not ok))
        out = _lbfgs(vg, x0, maxiter, gtol, bounds_i)
        out.theta[index] = grid[i]
        return out

    def solve(i: int, theta0: np.ndarray) -> np.ndarray:
        """Solve grid point i from theta0; returns the start for the next point of the walk."""
        try:
            out = solve_point(i, theta0)
        except Exception as exc:
            message = (
                f"profile of index {index} failed at grid point {i} (value {grid[i]:.6g}): "
                f"{type(exc).__name__}: {exc}"
            )
            raise ProfilePointError(message, result()) from exc
        losses[i], success[i], failed[i] = out.loss, out.success, out.failed
        n_failed_evals[i] = out.n_failed_evals
        thetas[i] = out.theta
        return theta0 if out.failed else out.theta

    start = int(np.argmin(np.abs(grid - theta_hat[index])))
    theta_start = solve(start, theta_hat)
    theta = theta_start
    for i in range(start + 1, n_grid):
        theta = solve(i, theta)
    theta = theta_start
    for i in range(start - 1, -1, -1):
        theta = solve(i, theta)
    return result()


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
    """Likelihood-based interval. An unbounded side reports the grid edge and has *_bounded = False.

    *_failed = True: that side's outward walk met a non-finite deviance (a failed profile point) before
    the threshold crossing. The side then reports the last finite grid point inside and is unbounded.
    """

    lo: float
    hi: float
    lo_bounded: bool
    hi_bounded: bool
    threshold: float
    lo_failed: bool = False
    hi_failed: bool = False

    @property
    def bounded(self) -> bool:
        return self.lo_bounded and self.hi_bounded


def likelihood_interval(grid: ArrayLike, deviance: ArrayLike, level: float = 0.95) -> Interval:
    """Interval where the deviance stays at or below chi2.ppf(level, 1) (3.841 at 95%).

    The interval is the connected region of grid points around the minimum. Each bounded endpoint is the
    linear interpolation of the threshold crossing between the last point inside and the first outside.
    If even the minimum exceeds the threshold, the interval collapses to the minimum grid point.

    deviance may hold non-finite entries (failed profile points); at least one must be finite. The
    minimum is taken over the finite entries. A non-finite entry met before the crossing ends that side:
    the endpoint is the last finite grid point inside, *_bounded is False and *_failed is True. Entries
    beyond a crossing do not affect the interval.
    """
    if not (np.isfinite(level) and 0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    grid = _as_grid(grid, min_points=2)
    dev = np.asarray(deviance, dtype=np.float64)
    if dev.shape != grid.shape:
        raise ValueError(f"deviance must have the grid shape {grid.shape}, got {dev.shape}")
    finite = np.isfinite(dev)
    if not np.any(finite):
        raise ValueError("deviance must have at least one finite entry")
    threshold = float(scipy.stats.chi2.ppf(level, 1))
    i_min = int(np.argmin(np.where(finite, dev, np.inf)))
    if dev[i_min] > threshold:
        x = float(grid[i_min])
        return Interval(lo=x, hi=x, lo_bounded=True, hi_bounded=True, threshold=threshold)

    def inside(j: int) -> bool:
        return bool(finite[j] and dev[j] <= threshold)

    j = i_min
    while j + 1 < grid.shape[0] and inside(j + 1):
        j += 1
    hi_failed = False
    if j == grid.shape[0] - 1:
        hi, hi_bounded = float(grid[-1]), False
    elif not finite[j + 1]:
        hi, hi_bounded, hi_failed = float(grid[j]), False, True
    else:
        step = (grid[j + 1] - grid[j]) / (dev[j + 1] - dev[j])
        hi, hi_bounded = float(grid[j] + (threshold - dev[j]) * step), True

    j = i_min
    while j - 1 >= 0 and inside(j - 1):
        j -= 1
    lo_failed = False
    if j == 0:
        lo, lo_bounded = float(grid[0]), False
    elif not finite[j - 1]:
        lo, lo_bounded, lo_failed = float(grid[j]), False, True
    else:
        step = (grid[j] - grid[j - 1]) / (dev[j - 1] - dev[j])
        lo, lo_bounded = float(grid[j] - (threshold - dev[j]) * step), True
    return Interval(
        lo=lo,
        hi=hi,
        lo_bounded=lo_bounded,
        hi_bounded=hi_bounded,
        threshold=threshold,
        lo_failed=lo_failed,
        hi_failed=hi_failed,
    )


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

    def coordinate_bounds(
        self, box: Mapping[str, tuple[float, float]] | None = None
    ) -> list[tuple[float, float]]:
        """(lo, hi) per coordinate in layout order, from a natural-units box (default S1C_BOUNDS).

        Log of the box for k, rho, gamma and kappa, logit for h_opt. box maps each group name and
        'kappa' to (low, high) with 0 < low < high (and high < 1 for h_opt).
        """
        box = S1C_BOUNDS if box is None else box
        coords: dict[str, tuple[float, float]] = {}
        for name in (*_GROUPS, "kappa"):
            if name not in box:
                raise ValueError(f"box must have an entry for {name!r}")
            low, high = (float(x) for x in box[name])
            upper = 1.0 if name == "h_opt" else np.inf
            if not (np.isfinite(low) and 0.0 < low < high < upper):
                raise ValueError(
                    f"box[{name!r}] must satisfy 0 < low < high (< 1 for h_opt), got {box[name]!r}"
                )
            if name == "h_opt":
                coords[name] = (float(np.log(low / (1.0 - low))), float(np.log(high / (1.0 - high))))
            else:
                coords[name] = (float(np.log(low)), float(np.log(high)))
        m = len(self.compound_idx)
        return [coords[g] for g in _GROUPS for _ in range(m)] + [coords["kappa"]]

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

    The shape g is residuals.observation with b = 0 on ODEModel.solve_with_status states at batch.tau.
    Each stint offset is concentrated out in closed form: b_s = masked mean of (y - g) over the scored
    laps. residual_fn returns where(mask, y - g - b, 0) flattened to (n * K,); sse_fn is its sum of
    squares.

    The solves do not raise on failure: if any stint's ODE solve fails (for example max_steps reached
    in the stiff region of very large k_c), residual_fn returns all NaN and sse_fn returns NaN. The
    optimizers in this module treat that as a failed evaluation.
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
        h, w, ok = jax.vmap(lambda c, q: model.solve_with_status(c, q, batch.tau))(
            batch.compound_idx, batch.q
        )
        g = observation(physics, batch.compound_idx[:, None], h, w, 0.0)
        r = jnp.where(mask, y - g, 0.0)
        b = r.sum(axis=1) / counts
        out = jnp.where(mask, r - b[:, None], 0.0).reshape(-1)
        return jnp.where(jnp.all(ok), out, jnp.nan)

    def sse_fn(v: jax.Array) -> jax.Array:
        return jnp.sum(residual_fn(v) ** 2)

    return sse_fn, residual_fn, n_obs


@dataclass(frozen=True, eq=False)
class MLEResult:
    """Best of several L-BFGS-B starts. start_sse[i] and start_theta[i] are the final values of start i.

    start_failed[i]: start i found no finite evaluation after its first one (start_sse[i] is NaN); such a
    start is never chosen while a finite one exists. n_failed_evals[i] counts the non-finite evaluations
    of start i.
    """

    theta: np.ndarray
    sse: float
    start_sse: list[float]
    start_theta: list[np.ndarray]
    best_index: int
    success: bool
    start_failed: list[bool]
    n_failed_evals: list[int]


def ode_mle(
    sse_fn: LossFn,
    starts: Sequence[ArrayLike],
    *,
    maxiter: int = 500,
    gtol: float = 1e-8,
    value_and_grad: ValueAndGrad | None = None,
    bounds: Bounds | None = None,
) -> MLEResult:
    """Minimize sse_fn from each start and keep the lowest finite SSE.

    All starts share one jitted value_and_grad: the caller's (a jitted jax.value_and_grad(sse_fn), so the
    profiles of the same dataset reuse its compilation) or a new one. bounds (one (lo, hi) pair per
    coordinate, None = free side; default all free) applies to every start; starts outside it are
    clipped into it.
    """
    starts = [_as_vector(f"starts[{i}]", s) for i, s in enumerate(starts)]
    if not starts:
        raise ValueError("starts must contain at least one start vector")
    if len({s.shape for s in starts}) != 1:
        raise ValueError("starts must all have the same length")
    maxiter, gtol = _check_options(maxiter, gtol)
    box = None if bounds is None else _check_bounds(bounds, starts[0].shape[0])
    vg = _value_and_grad(sse_fn, value_and_grad)
    results = [_lbfgs(vg, s, maxiter, gtol, box) for s in starts]
    start_sse = [r.loss for r in results]
    finite = [s if np.isfinite(s) else np.inf for s in start_sse]
    best = int(np.argmin(finite))
    return MLEResult(
        theta=results[best].theta,
        sse=results[best].loss,
        start_sse=start_sse,
        start_theta=[r.theta for r in results],
        best_index=best,
        success=results[best].success,
        start_failed=[r.failed for r in results],
        n_failed_evals=[r.n_failed_evals for r in results],
    )
