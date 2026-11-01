import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
from scipy.stats import chi2

from apexpinn.models.ode_model import ODEModel
from apexpinn.models.pinn import PINN
from apexpinn.models.stints import batch_from_synthetic
from apexpinn.physics.identifiability import (
    PRIOR_RANGES,
    S1C_BOUNDS,
    Interval,
    ProfilePointError,
    ProfileResult,
    ThetaLayout,
    gauss_newton_spectrum,
    identifiability_label,
    likelihood_interval,
    minimize,
    ode_least_squares,
    ode_mle,
    profile_likelihood,
    profile_shape_label,
    refine_interval,
    sample_prior_physics,
    sse_deviance,
)
from apexpinn.physics.residuals import PhysicsParams
from apexpinn.physics.synthetic import default_theta_true, make_synthetic_dataset

THRESHOLD_95 = float(chi2.ppf(0.95, 1))
USED = (1, 2, 3)  # C2, C3, C4
UNUSED = (0, 4)  # C1, C5
GROUPS = ("k", "rho", "gamma", "h_opt")


@pytest.fixture
def x64():
    """Enable float64 for one test only, then restore the previous setting."""
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _neutral_base() -> PhysicsParams:
    return PhysicsParams.from_constrained(k=0.3, rho=0.05, gamma=1.0, h_opt=0.5, kappa=0.5)


# --- PLAN T5 tooling tests ---


def test_at1_product_model_has_one_null_direction_and_flat_profile(x64):
    """Tooling test 1: y = a*b*x is identifiable only through a*b."""
    x = np.linspace(0.5, 3.0, 20)
    y = 2.0 * 1.5 * x + 0.01 * np.random.default_rng(0).standard_normal(x.shape[0])

    def residual_fn(theta):
        return y - theta[0] * theta[1] * x

    def loss_fn(theta):
        return jnp.sum(residual_fn(theta) ** 2)

    spectrum = gauss_newton_spectrum(residual_fn, [2.0, 1.5])
    assert spectrum.eigenvalues.shape == (2,)
    assert spectrum.relative[0] == pytest.approx(1.0)
    assert int(np.sum(spectrum.relative < 1e-6)) == 1

    theta_hat, sse_hat, success = minimize(loss_fn, [2.0, 1.5])
    assert success and sse_hat > 0
    grid = [1.0, 1.5, 2.0, 2.5, 3.0]
    profile = profile_likelihood(loss_fn, theta_hat, index=0, grid=grid)
    np.testing.assert_allclose(profile.thetas[:, 0], grid)
    # The free parameter b moved so that a*b stays at the fitted product.
    product = profile.thetas[:, 0] * profile.thetas[:, 1]
    np.testing.assert_allclose(product, theta_hat[0] * theta_hat[1], rtol=1e-5)
    deviance = sse_deviance(profile.loss, n_obs=x.shape[0], sse_min=sse_hat)
    assert np.all(deviance < 1e-3)
    interval = likelihood_interval(profile.grid, deviance)
    assert not interval.lo_bounded and not interval.hi_bounded
    assert profile_shape_label(interval, truth=2.0) == "flat"


def test_at2_linear_model_has_sharp_profiles_at_the_truth(x64):
    """Tooling test 2: y = a*x + b with spread x gives bounded intervals around the truth."""
    x = np.linspace(-3.0, 3.0, 40)
    truth = np.array([1.3, -0.4])
    y = truth[0] * x + truth[1] + 0.05 * np.random.default_rng(0).standard_normal(x.shape[0])

    def residual_fn(theta):
        return y - theta[0] * x - theta[1]

    def loss_fn(theta):
        return jnp.sum(residual_fn(theta) ** 2)

    theta_hat = np.polyfit(x, y, 1)
    sse_hat = float(np.sum((y - np.polyval(theta_hat, x)) ** 2))
    spectrum = gauss_newton_spectrum(residual_fn, theta_hat)
    assert np.all(spectrum.relative >= 1e-3)
    rev = gauss_newton_spectrum(residual_fn, theta_hat, mode="rev")
    np.testing.assert_allclose(rev.eigenvalues, spectrum.eigenvalues, rtol=1e-10)
    with pytest.raises(ValueError, match="mode"):
        gauss_newton_spectrum(residual_fn, theta_hat, mode="central")

    for index in (0, 1):
        grid = truth[index] + np.linspace(-0.2, 0.2, 21)
        profile = profile_likelihood(loss_fn, theta_hat, index, grid)
        assert np.all(profile.success)
        nearest = int(np.argmin(np.abs(grid - truth[index])))
        assert abs(int(np.argmin(profile.loss)) - nearest) <= 1
        interval = likelihood_interval(grid, sse_deviance(profile.loss, x.shape[0], sse_min=sse_hat))
        assert interval.lo_bounded and interval.hi_bounded
        assert interval.lo <= truth[index] <= interval.hi


def test_profile_reoptimizes_free_parameters_on_correlated_design(x64):
    """With x > 0, a and b are correlated, so the profiled b must follow the conditional fit."""
    x = np.linspace(0.5, 3.0, 20)
    y = 0.7 * x + 0.2 + 0.01 * np.random.default_rng(1).standard_normal(x.shape[0])

    def loss_fn(theta):
        return jnp.sum((y - theta[0] * x - theta[1]) ** 2)

    theta_hat = np.polyfit(x, y, 1)
    grid = np.linspace(0.5, 0.9, 5)
    profile = profile_likelihood(loss_fn, theta_hat, 0, grid)
    np.testing.assert_allclose(profile.thetas[:, 0], grid)
    np.testing.assert_allclose(profile.thetas[:, 1], [np.mean(y - a * x) for a in grid], atol=1e-6)


# --- interval method v2 (PLAN T5 'Interval method v2') ---


def _slope_profile(factors):
    """Coarse profile of the slope a of y = a x + b on grid a_hat + half * factors.

    With b concentrated out, SSE(a) = SSE_min + Sxx (a - a_hat)^2, so the deviance is
    n log(1 + f^2 q0) at a = a_hat + f * half, with q0 = expm1(3.841 / n), and the exact 95% endpoints are
    a_hat -+ half, half = sqrt(SSE_min q0 / Sxx). For n = 20 the deviance is 1.03 at f = 0.5, 3.16 at
    f = 0.9 and 12.3 at f = 2, whatever the noise draw.
    """
    x = np.linspace(-1.0, 1.0, 20)
    y = 0.8 * x + 0.3 + 0.5 * np.random.default_rng(3).standard_normal(x.shape[0])

    def loss_fn(theta):
        return jnp.sum((y - theta[0] * x - theta[1]) ** 2)

    theta_hat = np.polyfit(x, y, 1)
    sse_min = float(np.sum((y - np.polyval(theta_hat, x)) ** 2))
    n = x.shape[0]
    half = float(np.sqrt(sse_min * np.expm1(THRESHOLD_95 / n) / np.sum((x - x.mean()) ** 2)))
    grid = theta_hat[0] + half * np.asarray(factors, dtype=np.float64)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    profile = profile_likelihood(loss_fn, theta_hat, 0, grid, value_and_grad=vg)
    deviance = sse_deviance(profile.loss, n, sse_min=sse_min)
    return {
        "loss_fn": loss_fn,
        "vg": vg,
        "profile": profile,
        "deviance": deviance,
        "coarse": likelihood_interval(grid, deviance),
        "exact": (float(theta_hat[0] - half), float(theta_hat[0] + half)),
        "half": half,
        "kwargs": {"sse_min": sse_min, "n_obs": n, "bounds": None, "maxiter": 500},
    }


@pytest.mark.parametrize("factors", [(-2.0, -0.5, 0.0, 0.5, 2.0), (-2.0, 0.0, 2.0)])
def test_refine_interval_finds_the_analytic_crossing_that_linear_interpolation_misses(x64, factors):
    """The second grid has the coarse minimum as the inside bracket point on both sides."""
    p = _slope_profile(factors)
    coarse = p["coarse"]
    assert coarse.bounded
    interval, lo, hi = refine_interval(p["vg"], p["profile"], 0, coarse, p["deviance"], **p["kwargs"])
    assert interval.bounded and interval.threshold == coarse.threshold
    sides = ((lo, coarse.lo, interval.lo, p["exact"][0]), (hi, coarse.hi, interval.hi, p["exact"][1]))
    for side, linear, refined, exact in sides:
        assert side.refined and side.reason is None
        assert side.n_evals > 0 and side.coords.shape == side.sse.shape == (side.n_evals,)
        assert np.all(np.isfinite(side.sse)) and np.all(np.isfinite(side.deviance))
        assert side.coord == refined
        assert abs(refined - exact) < 1e-3
        assert abs(linear - exact) > 0.1 * p["half"]  # linear interpolation is far off on this grid


def test_refine_interval_keeps_the_linear_endpoint_when_a_refinement_solve_fails(x64):
    p = _slope_profile((-2.0, -0.5, 0.0, 0.5, 2.0))
    coarse = p["coarse"]
    cut = p["exact"][1] - 0.2 * p["half"]

    def nan_beyond(theta):
        return jnp.where(theta[0] > cut, jnp.nan, p["loss_fn"](theta))

    vg_nan = jax.jit(jax.value_and_grad(nan_beyond))
    interval, lo, hi = refine_interval(vg_nan, p["profile"], 0, coarse, p["deviance"], **p["kwargs"])
    assert hi.reason == "failed_solve" and not hi.refined
    assert interval.hi == hi.coord == coarse.hi and interval.hi_bounded
    assert hi.n_evals >= 1 and np.isnan(hi.sse[-1]) and not hi.success[-1]
    assert lo.refined and abs(interval.lo - p["exact"][0]) < 1e-3


def test_refine_interval_leaves_open_sides_and_reports_each_fallback(x64):
    p = _slope_profile((-2.0, -0.5, 0.0, 0.5, 0.9))  # deviance 3.16 at the upper grid edge: open
    coarse = p["coarse"]
    assert coarse.lo_bounded and not coarse.hi_bounded
    interval, lo, hi = refine_interval(p["vg"], p["profile"], 0, coarse, p["deviance"], **p["kwargs"])
    assert hi.reason == "unbounded" and not hi.refined and hi.n_evals == 0 and hi.coords.shape == (0,)
    assert interval.hi == coarse.hi and not interval.hi_bounded
    assert lo.refined and abs(interval.lo - p["exact"][0]) < 1e-3

    def no_solve(theta):
        raise AssertionError("no refinement solve expected")

    high = np.array([10.0, 9.0, 5.0, 9.0, 10.0])  # even the minimum is above 3.841
    collapsed = likelihood_interval(p["profile"].grid, high)
    out, lo, hi = refine_interval(no_solve, p["profile"], 0, collapsed, high, **p["kwargs"])
    assert out is collapsed and lo.reason == hi.reason == "no_interval" and lo.n_evals == hi.n_evals == 0

    q = _slope_profile((-2.0, -0.5, 0.0, 0.5, 2.0))
    coarse = q["coarse"]
    doubled = jax.jit(jax.value_and_grad(lambda theta: 2.0 * q["loss_fn"](theta)))
    out, lo, hi = refine_interval(doubled, q["profile"], 0, coarse, q["deviance"], **q["kwargs"])
    assert (lo.reason, hi.reason) == ("bracket_lost", "bracket_lost")  # both ends re-solve above 3.841
    assert (out.lo, out.hi) == (coarse.lo, coarse.hi) and lo.n_evals == hi.n_evals == 2

    args = (q["vg"], q["profile"], 0, coarse, q["deviance"])
    out, lo, hi = refine_interval(*args, **q["kwargs"], xtol=1e-12, max_iter=1)
    assert (lo.reason, hi.reason) == ("not_converged", "not_converged")
    assert (out.lo, out.hi) == (coarse.lo, coarse.hi) and not lo.refined and lo.n_evals >= 3

    with pytest.raises(ValueError, match="deviance"):
        refine_interval(q["vg"], q["profile"], 0, coarse, q["deviance"][:3], **q["kwargs"])


def test_passed_value_and_grad_is_reused_and_matches_the_default_path(x64):
    """Compile once: minimize, profile_likelihood and ode_mle call the caller's value_and_grad."""
    x = np.linspace(0.5, 3.0, 20)
    y = 0.7 * x + 0.2 + 0.01 * np.random.default_rng(2).standard_normal(x.shape[0])

    def loss_fn(theta):
        return jnp.sum((y - theta[0] * x - theta[1]) ** 2)

    jitted = jax.jit(jax.value_and_grad(loss_fn))
    calls = {"n": 0}

    def counted(theta):
        calls["n"] += 1
        return jitted(theta)

    theta_hat, sse_hat, ok = minimize(loss_fn, [1.0, 0.0], value_and_grad=counted)
    after_minimize = calls["n"]
    assert ok and after_minimize > 0
    default_theta, default_sse, _ = minimize(loss_fn, [1.0, 0.0])
    assert calls["n"] == after_minimize  # the default path builds its own function
    np.testing.assert_allclose(theta_hat, default_theta, rtol=1e-12)
    assert sse_hat == pytest.approx(default_sse, rel=1e-12)

    grid = np.linspace(0.5, 0.9, 5)
    profile = profile_likelihood(loss_fn, theta_hat, 0, grid, value_and_grad=counted)
    after_profile = calls["n"]
    assert after_profile > after_minimize
    default_profile = profile_likelihood(loss_fn, theta_hat, 0, grid)
    assert calls["n"] == after_profile
    np.testing.assert_allclose(profile.loss, default_profile.loss, rtol=1e-12)
    np.testing.assert_allclose(profile.thetas, default_profile.thetas, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(profile.thetas[:, 0], grid)

    starts = [[1.0, 0.0], [0.3, 0.6]]
    fit = ode_mle(loss_fn, starts, value_and_grad=counted)
    assert calls["n"] > after_profile
    default_fit = ode_mle(loss_fn, starts)
    np.testing.assert_allclose(fit.theta, default_fit.theta, rtol=1e-12)
    np.testing.assert_allclose(fit.start_sse, default_fit.start_sse, rtol=1e-12)
    with pytest.raises(ValueError, match="value_and_grad"):
        minimize(loss_fn, [1.0, 0.0], value_and_grad="not a function")


def test_profile_and_deviance_validation():
    def loss_fn(theta):
        return jnp.sum(theta**2)

    with pytest.raises(ValueError, match="grid"):
        profile_likelihood(loss_fn, [0.0, 0.0], 0, [1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="grid"):
        profile_likelihood(loss_fn, [0.0, 0.0], 0, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="index"):
        profile_likelihood(loss_fn, [0.0, 0.0], 2, [1.0, 2.0])
    with pytest.raises(ValueError, match="sse"):
        sse_deviance([1.0, 0.0], n_obs=10)
    with pytest.raises(ValueError, match="sse_min"):
        sse_deviance([1.0, 2.0], n_obs=10, sse_min=-1.0)
    with pytest.raises(ValueError, match="n_obs"):
        sse_deviance([1.0, 2.0], n_obs=0)


def test_sse_deviance_is_the_gaussian_profile_likelihood_ratio():
    sse = np.array([2.0, 1.0, 4.0])
    np.testing.assert_allclose(sse_deviance(sse, n_obs=10), 10 * np.log(sse / 1.0))
    np.testing.assert_allclose(sse_deviance(sse, n_obs=10, sse_min=0.5), 10 * np.log(sse / 0.5))


# --- optimizer failures and bounds ---


def _nan_beyond(theta):
    """Quadratic with its minimum at theta0 = 2, but NaN wherever theta0 > 1.5 (a failed ODE solve)."""
    loss = (theta[0] - 2.0) ** 2 + theta[1] ** 2
    return jnp.where(theta[0] > 1.5, jnp.nan, loss)


def test_minimize_backtracks_from_non_finite_evaluations(x64):
    start = [0.0, 0.3]
    theta, loss, _ok = minimize(_nan_beyond, start)
    assert np.all(np.isfinite(theta)) and np.isfinite(loss)
    assert theta[0] <= 1.5
    assert loss == pytest.approx(float(_nan_beyond(jnp.asarray(theta))), rel=1e-12)
    assert loss < float(_nan_beyond(jnp.asarray(start)))

    fit = ode_mle(_nan_beyond, [start])
    assert fit.start_failed == [False] and fit.n_failed_evals[0] > 0
    assert np.isfinite(fit.sse) and fit.theta[0] <= 1.5

    profile = profile_likelihood(_nan_beyond, start, index=1, grid=[-0.1, 0.0, 0.1])
    assert profile.failed.dtype == bool and not np.any(profile.failed)
    assert np.all(np.isfinite(profile.loss)) and np.all(profile.thetas[:, 0] <= 1.5)
    assert np.all(profile.n_failed_evals > 0)


def _scripted_lbfgs(monkeypatch, trials, final, stale_fun=False, success=True):
    """Replace scipy's L-BFGS-B by a script: evaluate x0 and trials, then return the iterate final.

    Trials and final are clipped into the bounds first. The returned fun is the value of the last
    evaluation when stale_fun (as scipy does after restoring an iterate), else the value at final.
    """

    def fake(fun, x0, bounds=None, **kwargs):
        def clip(x):
            x = np.asarray(x, dtype=np.float64)
            if bounds is None:
                return x
            lo = [-np.inf if b[0] is None else b[0] for b in bounds]
            hi = [np.inf if b[1] is None else b[1] for b in bounds]
            return np.clip(x, lo, hi)

        values = {}
        last = None
        for x in [x0, *trials]:
            x = clip(x)
            last, _ = fun(x)
            values[x.tobytes()] = last
        x_final = clip(final)
        f_final = last if stale_fun else values[x_final.tobytes()]
        return scipy.optimize.OptimizeResult(x=x_final, fun=f_final, success=success)

    monkeypatch.setattr(scipy.optimize, "minimize", fake)


def test_minimize_falls_back_to_the_best_finite_iterate(x64, monkeypatch):
    """A penalty point accepted as the final iterate, or a stale penalty in res.fun, is not a failure."""
    start, good, bad = [0.0, 0.3], [1.0, 0.1], [1.8, 0.0]  # _nan_beyond: 4.09, 1.01, NaN
    _scripted_lbfgs(monkeypatch, trials=[good, bad], final=bad)  # the line search accepted the penalty
    theta, loss, ok = minimize(_nan_beyond, start)
    np.testing.assert_array_equal(theta, good)
    assert loss == pytest.approx(1.01, rel=1e-12) and ok is False
    fit = ode_mle(_nan_beyond, [start])
    assert fit.start_failed == [False] and fit.n_failed_evals == [1] and fit.sse == pytest.approx(1.01)

    _scripted_lbfgs(monkeypatch, trials=[good, bad], final=good, stale_fun=True, success=False)
    theta, loss, ok = minimize(_nan_beyond, start)  # scipy restored good but kept the penalty in fun
    np.testing.assert_array_equal(theta, good)
    assert loss == pytest.approx(1.01, rel=1e-12) and ok is False

    _scripted_lbfgs(monkeypatch, trials=[good], final=good)
    theta, loss, ok = minimize(_nan_beyond, start)  # a clean finish is unchanged
    np.testing.assert_array_equal(theta, good)
    assert loss == pytest.approx(1.01, rel=1e-12) and ok is True

    _scripted_lbfgs(monkeypatch, trials=[bad], final=bad)  # no finite iterate beyond the start
    theta, loss, ok = minimize(_nan_beyond, start)
    assert np.isnan(loss) and ok is False
    fit = ode_mle(_nan_beyond, [start])
    assert fit.start_failed == [True] and np.isnan(fit.sse)

    grid = np.array([-0.1, 0.0, 0.1])  # profile of theta[1]; each point accepts the penalty
    _scripted_lbfgs(monkeypatch, trials=[good, bad], final=bad)
    profile = profile_likelihood(_nan_beyond, start, index=1, grid=grid)
    assert not np.any(profile.failed) and not np.any(profile.success)
    np.testing.assert_allclose(profile.thetas, np.stack([np.ones(3), grid], axis=1))
    np.testing.assert_allclose(profile.loss, 1.0 + grid**2, rtol=1e-12)
    assert np.all(profile.n_failed_evals == 1)


def test_minimize_marks_a_restored_start_as_failed(x64, monkeypatch):
    """scipy restores x0 when every trial of its first line search fails: that solve is failed."""
    start, good, bad = [0.0, 0.3], [1.0, 0.1], [1.8, 0.0]  # _nan_beyond: 4.09, 1.01, NaN
    _scripted_lbfgs(monkeypatch, trials=[bad], final=start, stale_fun=True, success=False)
    theta, loss, ok = minimize(_nan_beyond, start)
    np.testing.assert_array_equal(theta, start)
    assert np.isnan(loss) and ok is False
    fit = ode_mle(_nan_beyond, [start])
    assert fit.start_failed == [True] and fit.n_failed_evals == [1] and np.isnan(fit.sse)

    grid = np.array([-0.1, 0.0, 0.1])  # profile of theta[1]; every point restores its start
    profile = profile_likelihood(_nan_beyond, start, index=1, grid=grid)
    assert np.all(profile.failed) and not np.any(profile.success)
    assert np.all(np.isnan(profile.loss)) and np.all(profile.n_failed_evals == 1)

    _scripted_lbfgs(monkeypatch, trials=[], final=start)  # a clean stop at the start is not failed
    theta, loss, ok = minimize(_nan_beyond, start)
    assert loss == pytest.approx(4.09, rel=1e-12) and ok is True

    for trials in ([bad, start], [good, bad]):  # a finite evaluation after the start, then x0 restored
        _scripted_lbfgs(monkeypatch, trials=trials, final=start, stale_fun=True, success=False)
        theta, loss, ok = minimize(_nan_beyond, start)
        np.testing.assert_array_equal(theta, start)
        assert loss == pytest.approx(4.09, rel=1e-12) and ok is False


def test_minimize_reports_a_non_finite_start_as_failed(x64):
    def loss_fn(theta):
        return jnp.where(theta[0] > 0.0, jnp.nan, jnp.sum(theta**2))

    theta, loss, ok = minimize(loss_fn, [1.0, 2.0])
    assert np.isnan(loss) and ok is False
    np.testing.assert_array_equal(theta, [1.0, 2.0])

    fit = ode_mle(loss_fn, [[1.0, 2.0], [-1.0, 2.0]])
    assert fit.start_failed == [True, False] and fit.best_index == 1
    assert np.isnan(fit.start_sse[0]) and np.isfinite(fit.sse)
    assert fit.n_failed_evals[0] == 1


def test_bounds_keep_free_coordinates_inside_the_box(x64):
    def loss_fn(theta):
        return (theta[0] - 3.0) ** 2 + (theta[1] + 2.0) ** 2

    box = [(-1.0, 1.0), (-1.0, 1.0)]
    fit = ode_mle(loss_fn, [[0.0, 0.0], [5.0, -5.0]], bounds=box)
    np.testing.assert_allclose(fit.theta, [1.0, -1.0], atol=1e-8)
    for theta in fit.start_theta:
        np.testing.assert_allclose(theta, [1.0, -1.0], atol=1e-8)

    grid = [-2.0, 0.0, 2.0]  # the outer grid values lie outside the box of theta[0]
    profile = profile_likelihood(loss_fn, fit.theta, 0, grid, bounds=box)
    np.testing.assert_array_equal(profile.thetas[:, 0], grid)
    np.testing.assert_allclose(profile.thetas[:, 1], -1.0, atol=1e-8)
    np.testing.assert_allclose(profile.loss, [(g - 3.0) ** 2 + 1.0 for g in grid], atol=1e-8)
    with pytest.raises(ValueError, match="bounds"):
        ode_mle(loss_fn, [[0.0, 0.0]], bounds=[(-1.0, 1.0)])
    with pytest.raises(ValueError, match="bounds"):
        profile_likelihood(loss_fn, [0.0, 0.0], 0, grid, bounds=[(1.0, -1.0), (None, None)])


def _valley(theta):
    return (theta[0] - 1.0) ** 2 + (theta[1] - 0.5 * theta[0]) ** 2


def test_profile_restarts_after_a_failed_point_from_the_last_finite_solution(x64):
    def holed(theta):
        return jnp.where(theta[0] == 1.5, jnp.nan, _valley(theta))

    grid = np.linspace(0.0, 2.0, 5)  # the walk starts at 1.0 and meets the hole at grid[3] = 1.5
    theta_hat = [1.0, 0.5]
    reference = profile_likelihood(_valley, theta_hat, 0, grid)
    profile = profile_likelihood(holed, theta_hat, 0, grid)
    assert np.isnan(profile.loss[3]) and not profile.success[3] and profile.failed[3]
    assert profile.n_failed_evals[3] == 1
    keep = [0, 1, 2, 4]
    assert not np.any(profile.failed[keep])
    np.testing.assert_allclose(profile.loss[keep], reference.loss[keep], atol=1e-10)
    np.testing.assert_allclose(profile.thetas[keep], reference.thetas[keep], atol=1e-6)
    np.testing.assert_allclose(profile.loss[keep], (grid[keep] - 1.0) ** 2, atol=1e-10)


def test_profile_point_error_names_the_point_and_keeps_the_partial_result(x64):
    jitted = jax.jit(jax.value_and_grad(_valley))

    def raising(theta):
        if float(theta[0]) == 0.5:
            raise RuntimeError("solver exploded")
        return jitted(theta)

    grid = np.linspace(0.0, 2.0, 5)  # walk order 2, 3, 4, then 1 (value 0.5) raises
    with pytest.raises(ProfilePointError) as info:
        profile_likelihood(_valley, [1.0, 0.5], 0, grid, value_and_grad=raising)
    err = info.value
    assert isinstance(err, RuntimeError)
    message = str(err)
    assert "index 0" in message and "grid point 1" in message and "value 0.5" in message
    assert "RuntimeError: solver exploded" in message
    assert isinstance(err.__cause__, RuntimeError) and str(err.__cause__) == "solver exploded"
    partial = err.partial
    assert isinstance(partial, ProfileResult)
    np.testing.assert_array_equal(partial.grid, grid)
    solved, unsolved = [2, 3, 4], [0, 1]
    assert np.all(np.isfinite(partial.loss[solved])) and np.all(partial.success[solved])
    np.testing.assert_allclose(partial.thetas[solved, 0], grid[solved])
    assert np.all(np.isnan(partial.loss[unsolved])) and not np.any(partial.success[unsolved])
    assert np.all(np.isnan(partial.thetas[unsolved]))
    assert not np.any(partial.failed)


# --- likelihood_interval ---


def test_likelihood_interval_bounded_with_linear_interpolation():
    grid = np.arange(5.0)
    interval = likelihood_interval(grid, [8.0, 2.0, 0.0, 2.0, 8.0])
    assert interval.threshold == pytest.approx(3.841, abs=1e-3)
    assert interval.lo_bounded and interval.hi_bounded
    assert interval.lo == pytest.approx(1.0 - (THRESHOLD_95 - 2.0) / 6.0)
    assert interval.hi == pytest.approx(3.0 + (THRESHOLD_95 - 2.0) / 6.0)


def test_likelihood_interval_unbounded_sides_report_grid_edges():
    left = likelihood_interval([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 8.0])
    assert not left.lo_bounded and left.hi_bounded
    assert left.lo == 0.0
    assert left.hi == pytest.approx(2.0 + (THRESHOLD_95 - 2.0) / 6.0)

    right = likelihood_interval([0.0, 1.0, 2.0, 3.0], [8.0, 2.0, 1.0, 0.0])
    assert right.lo_bounded and not right.hi_bounded
    assert right.hi == 3.0
    assert right.lo == pytest.approx(1.0 - (THRESHOLD_95 - 2.0) / 6.0)


def test_likelihood_interval_is_the_connected_region_around_the_minimum():
    grid = np.arange(7.0)
    interval = likelihood_interval(grid, [0.5, 6.0, 1.0, 0.0, 2.0, 5.0, 1.0])
    assert interval.lo_bounded and interval.hi_bounded
    assert interval.lo == pytest.approx(2.0 - (THRESHOLD_95 - 1.0) / 5.0)
    assert interval.hi == pytest.approx(4.0 + (THRESHOLD_95 - 2.0) / 3.0)


def test_likelihood_interval_level_and_input_validation():
    grid, deviance = [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]
    for level in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="level"):
            likelihood_interval(grid, deviance, level=level)
    with pytest.raises(ValueError, match="grid"):
        likelihood_interval([0.0, 2.0, 1.0], deviance)
    with pytest.raises(ValueError, match="deviance"):
        likelihood_interval(grid, [1.0, 0.0])
    assert likelihood_interval(grid, deviance, level=0.5).threshold == pytest.approx(chi2.ppf(0.5, 1))


def test_likelihood_interval_with_non_finite_entries():
    nan = np.nan
    grid = np.arange(7.0)
    clean = likelihood_interval(grid, [8.0, 8.0, 2.0, 0.0, 2.0, 8.0, 8.0])
    beyond = likelihood_interval(grid, [nan, 8.0, 2.0, 0.0, 2.0, 8.0, nan])
    assert beyond == clean and not clean.lo_failed and not clean.hi_failed

    hi = likelihood_interval(grid, [8.0, 8.0, 2.0, 0.0, 1.0, nan, 0.5])
    assert not hi.hi_bounded and hi.hi_failed and hi.hi == 4.0
    assert hi.lo_bounded and not hi.lo_failed
    assert hi.lo == pytest.approx(clean.lo)
    assert profile_shape_label(hi, truth=3.0) == "flat"

    lo = likelihood_interval(grid, [0.5, nan, 1.0, 0.0, 2.0, 8.0, 8.0])
    assert not lo.lo_bounded and lo.lo_failed and lo.lo == 2.0
    assert lo.hi_bounded and not lo.hi_failed
    assert lo.hi == pytest.approx(clean.hi)
    assert profile_shape_label(lo, truth=3.0) == "flat"

    with pytest.raises(ValueError, match="deviance"):
        likelihood_interval(grid, [nan] * 7)

    # The minimum is taken over the finite entries only (np.argmin alone would pick the NaN).
    first_nan = likelihood_interval(np.arange(5.0), [nan, 8.0, 0.0, 8.0, 8.0])
    assert first_nan.lo_bounded and first_nan.hi_bounded
    assert first_nan.lo == pytest.approx(2.0 - THRESHOLD_95 / 8.0)
    assert first_nan.hi == pytest.approx(2.0 + THRESHOLD_95 / 8.0)


# --- labels ---


def _interval(lo, hi, lo_bounded=True, hi_bounded=True):
    return Interval(lo=lo, hi=hi, lo_bounded=lo_bounded, hi_bounded=hi_bounded, threshold=THRESHOLD_95)


def test_profile_shape_label_rules():
    assert profile_shape_label(_interval(0.06, 0.14), truth=0.1) == "sharp"
    assert profile_shape_label(_interval(0.05, 0.15), truth=0.1) == "sharp"
    assert profile_shape_label(_interval(0.04, 0.12), truth=0.1) == "wide"
    assert profile_shape_label(_interval(0.08, 0.16), truth=0.1) == "wide"
    assert profile_shape_label(_interval(0.08, 0.12, lo_bounded=False), truth=0.1) == "flat"
    assert profile_shape_label(_interval(0.08, 0.12, hi_bounded=False), truth=0.1) == "flat"
    with pytest.raises(ValueError, match="truth"):
        profile_shape_label(_interval(0.08, 0.12), truth=0.0)


def test_identifiability_label_rules():
    sharp3 = ["sharp", "sharp", "sharp"]
    assert identifiability_label([True] * 5, sharp3) == "identifiable"
    assert identifiability_label([True, True, False, True, True], sharp3) == "weakly_identifiable"
    assert identifiability_label([True] * 5, ["sharp", "wide", "sharp"]) == "weakly_identifiable"
    assert identifiability_label([True] * 5, ["sharp", "flat", "sharp", "sharp"]) == "weakly_identifiable"
    assert identifiability_label([True] * 5, ["sharp", "flat", "flat", "sharp"]) == "non_identifiable"
    assert identifiability_label([False] * 5, ["flat", "flat", "wide"]) == "non_identifiable"
    assert identifiability_label(None, sharp3) == "identifiable"
    assert identifiability_label(None, ["sharp", "wide"]) == "weakly_identifiable"
    assert identifiability_label(None, ["flat", "wide"]) == "non_identifiable"
    with pytest.raises(ValueError, match="shapes"):
        identifiability_label(None, [])
    with pytest.raises(ValueError, match="shapes"):
        identifiability_label(None, ["sharp", "round"])
    with pytest.raises(ValueError, match="recovered"):
        identifiability_label([], sharp3)


# --- ThetaLayout and priors ---


def test_theta_layout_names_and_coordinates(x64):
    layout = ThetaLayout.from_compounds(("C2", "C3", "C4"))
    assert layout.compound_idx == USED
    assert layout.size == 13
    assert layout.names == (
        "k_c[C2]", "k_c[C3]", "k_c[C4]",
        "rho_c[C2]", "rho_c[C3]", "rho_c[C4]",
        "gamma_c[C2]", "gamma_c[C3]", "gamma_c[C4]",
        "h_opt_c[C2]", "h_opt_c[C3]", "h_opt_c[C4]",
        "kappa",
    )  # fmt: skip
    truth = default_theta_true()
    v = layout.to_vector(truth)
    assert isinstance(v, np.ndarray) and v.shape == (13,) and v.dtype == np.float64
    used = list(USED)
    np.testing.assert_allclose(v[0:3], np.log(np.asarray(truth.k)[used]), atol=1e-12)
    np.testing.assert_allclose(v[3:6], np.log(np.asarray(truth.rho)[used]), atol=1e-12)
    np.testing.assert_allclose(v[6:9], np.log(np.asarray(truth.gamma)[used]), atol=1e-12)
    h = np.asarray(truth.h_opt)[used]
    np.testing.assert_allclose(v[9:12], np.log(h / (1.0 - h)), atol=1e-12)
    assert v[12] == pytest.approx(np.log(float(truth.kappa)), abs=1e-12)
    constrained = layout.constrained(v)
    assert list(constrained) == list(layout.names)
    assert constrained["rho_c[C3]"] == pytest.approx(0.06, abs=1e-10)
    assert constrained["h_opt_c[C4]"] == pytest.approx(0.57, abs=1e-10)
    with pytest.raises(ValueError, match="compound_idx"):
        ThetaLayout((1, 1))
    with pytest.raises(ValueError, match="compound_idx"):
        ThetaLayout((5,))


def test_theta_layout_round_trip_keeps_unused_compounds(x64):
    layout = ThetaLayout.from_compounds(("C2", "C3", "C4"))
    truth = default_theta_true()
    base = _neutral_base()
    out = layout.to_physics(layout.to_vector(truth), base)
    used, unused = list(USED), list(UNUSED)
    for name in GROUPS:
        got = np.asarray(getattr(out, name))
        np.testing.assert_allclose(got[used], np.asarray(getattr(truth, name))[used], atol=1e-10)
        np.testing.assert_allclose(got[unused], np.asarray(getattr(base, name))[unused], atol=1e-12)
    assert float(out.kappa) == pytest.approx(float(truth.kappa), abs=1e-10)
    with pytest.raises(ValueError, match="v"):
        layout.to_physics(np.zeros(12), base)


def test_theta_layout_to_physics_is_differentiable(x64):
    layout = ThetaLayout.from_compounds(("C2", "C3", "C4"))
    truth = default_theta_true()
    v = jnp.asarray(layout.to_vector(truth))
    grad = np.asarray(jax.grad(lambda u: jnp.sum(layout.to_physics(u, _neutral_base()).rho))(v))
    assert np.all(np.isfinite(grad))
    # d rho / d log(rho) = rho on the rho entries, zero elsewhere.
    np.testing.assert_allclose(grad[3:6], np.asarray(truth.rho)[list(USED)], rtol=1e-8)
    np.testing.assert_allclose(np.delete(grad, [3, 4, 5]), 0.0, atol=1e-12)


def test_sample_prior_physics_is_deterministic_and_inside_ranges(x64):
    a = sample_prior_physics(jax.random.key(11))
    b = sample_prior_physics(jax.random.key(11))
    c = sample_prior_physics(jax.random.key(12))
    for leaf_a, leaf_b in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(leaf_a), np.asarray(leaf_b))
    assert not np.allclose(np.asarray(a.rho), np.asarray(c.rho))
    assert set(PRIOR_RANGES) == {"k", "rho", "gamma", "h_opt", "kappa"}
    for name, (low, high, _scale) in PRIOR_RANGES.items():
        values = np.atleast_1d(np.asarray(getattr(a, name)))
        assert np.all(values >= low * (1 - 1e-9)) and np.all(values <= high * (1 + 1e-9)), name
    assert np.asarray(a.k).shape == (5,) and np.asarray(a.kappa).shape == ()
    pinn = PINN.init(jax.random.key(0), physics=a)
    ode = ODEModel.init(physics=a)
    np.testing.assert_array_equal(np.asarray(pinn.physics.rho), np.asarray(a.rho))
    np.testing.assert_array_equal(np.asarray(ode.physics.k), np.asarray(a.k))


def test_prior_ranges_cover_the_synthetic_truth():
    truth = default_theta_true()
    for name, (low, high, _scale) in PRIOR_RANGES.items():
        values = np.atleast_1d(np.asarray(getattr(truth, name)))
        if name != "kappa":
            values = values[list(USED)]
        assert np.all(values > low) and np.all(values < high), name


def test_coordinate_bounds_follow_the_layout_order(x64):
    assert S1C_BOUNDS == {
        "k": (0.01, 10.0),
        "rho": (0.002, 2.0),
        "gamma": (0.1, 60.0),
        "h_opt": (0.02, 0.98),
        "kappa": (0.03, 30.0),
    }
    layout = ThetaLayout.from_compounds(("C2", "C3", "C4"))
    bounds = layout.coordinate_bounds(S1C_BOUNDS)
    assert len(bounds) == 13 and all(lo < hi for lo, hi in bounds)
    for i, group in enumerate(GROUPS):
        low, high = S1C_BOUNDS[group]
        if group == "h_opt":
            expected = (np.log(low / (1.0 - low)), np.log(high / (1.0 - high)))
        else:
            expected = (np.log(low), np.log(high))
        for j in range(3):
            assert bounds[3 * i + j] == pytest.approx(expected, rel=1e-12), layout.names[3 * i + j]
    assert bounds[12] == pytest.approx((np.log(0.03), np.log(30.0)), rel=1e-12)
    assert layout.coordinate_bounds() == bounds
    v = layout.to_vector(default_theta_true())
    assert all(lo < x < hi for x, (lo, hi) in zip(v, bounds, strict=True))
    with pytest.raises(ValueError, match="box"):
        layout.coordinate_bounds({**S1C_BOUNDS, "k": (1.0, 0.5)})
    with pytest.raises(ValueError, match="box"):
        layout.coordinate_bounds({**S1C_BOUNDS, "h_opt": (0.1, 1.0)})
    with pytest.raises(ValueError, match="box"):
        layout.coordinate_bounds({"k": (0.01, 10.0)})


# --- ODE likelihood adapter ---


def test_ode_least_squares_concentrates_offsets_on_noise_free_data(x64):
    ds = make_synthetic_dataset(jax.random.key(2), n_stints=3, noise_s=0.0)
    batch = batch_from_synthetic(ds)
    layout = ThetaLayout.from_compounds(("C2", "C3", "C4"))
    sse_fn, residual_fn, n_obs = ode_least_squares(batch, layout, ds.theta_true)
    mask = np.asarray(batch.mask)
    assert n_obs == int(mask.sum())

    v_true = layout.to_vector(ds.theta_true)
    assert float(sse_fn(v_true)) < 1e-8

    v_off = v_true.copy()
    v_off[3:6] += 0.3  # wrong wear rates leave a real misfit
    resid = np.asarray(residual_fn(v_off)).reshape(mask.shape)
    assert np.all(resid[~mask] == 0.0)
    np.testing.assert_allclose((resid * mask).sum(axis=1) / mask.sum(axis=1), 0.0, atol=1e-9)
    sse_off = float(sse_fn(v_off))
    assert sse_off > 1e-4
    assert sse_off == pytest.approx(float(np.sum(resid**2)), rel=1e-12)

    shifted = eqx.tree_at(lambda b: b.y, batch, batch.y.at[0].add(5.0))
    sse_shifted, _, _ = ode_least_squares(shifted, layout, ds.theta_true)
    assert float(sse_shifted(v_off)) == pytest.approx(sse_off, rel=1e-9)


def test_ode_least_squares_returns_nan_when_the_solver_hits_max_steps(x64):
    """A huge k_c makes the thermal ODE stiff: Tsit5 runs out of max_steps, the SSE is NaN, no raise."""
    ds = make_synthetic_dataset(jax.random.key(0), n_stints=2)
    batch = batch_from_synthetic(ds)
    layout = ThetaLayout.from_compounds(("C2", "C3", "C4"))
    sse_fn, _, _ = ode_least_squares(batch, layout, ds.theta_true)
    v_true = layout.to_vector(ds.theta_true)
    assert np.isfinite(float(sse_fn(v_true)))
    c0 = int(batch.compound_idx[0])
    assert c0 in layout.compound_idx
    v_stiff = v_true.copy()
    v_stiff[layout.compound_idx.index(c0)] = np.log(1e6)  # log k_c of the first stint's compound
    assert not np.isfinite(float(sse_fn(v_stiff)))

    good = ODEModel.init(physics=layout.to_physics(v_true, ds.theta_true))
    stiff = ODEModel.init(physics=layout.to_physics(v_stiff, ds.theta_true))
    h, w, ok = good.solve_with_status(c0, batch.q[0], batch.tau)
    assert bool(ok) and np.all(np.isfinite(np.asarray(h))) and np.all(np.isfinite(np.asarray(w)))
    _, _, ok = stiff.solve_with_status(c0, batch.q[0], batch.tau)
    assert not bool(ok)
    with pytest.raises(Exception, match="maximum number of solver steps"):
        jax.block_until_ready(stiff.solve(c0, batch.q[0], batch.tau))  # solve keeps throw=True


# --- slow: optimization through Diffrax ---


@pytest.fixture(scope="module")
def ode_problem():
    """One MLE on a 12-stint dataset from a prior draw, shared by the slow tests. Built in float64."""
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        ds = make_synthetic_dataset(jax.random.key(5), n_stints=12, noise_s=0.05)
        batch = batch_from_synthetic(ds)
        layout = ThetaLayout.from_compounds(("C2", "C3", "C4"))
        prior = sample_prior_physics(jax.random.key(7))
        sse_fn, residual_fn, n_obs = ode_least_squares(batch, layout, prior)
        fit = ode_mle(sse_fn, [layout.to_vector(prior)], maxiter=2000)
        sse_truth = float(sse_fn(layout.to_vector(ds.theta_true)))
    finally:
        jax.config.update("jax_enable_x64", previous)
    return {
        "layout": layout,
        "sse_fn": sse_fn,
        "residual_fn": residual_fn,
        "n_obs": n_obs,
        "fit": fit,
        "sse_truth": sse_truth,
    }


@pytest.mark.slow
def test_ode_mle_from_prior_draw_reaches_truth_sse(x64, ode_problem):
    fit = ode_problem["fit"]
    assert np.all(np.isfinite(fit.theta)) and fit.theta.shape == (13,)
    assert fit.best_index == 0 and len(fit.start_sse) == 1 and len(fit.start_theta) == 1
    assert fit.sse <= ode_problem["sse_truth"] * 1.0001


@pytest.mark.slow
def test_ode_profile_of_rho_c3_is_bounded_around_the_mle(x64, ode_problem):
    layout, fit = ode_problem["layout"], ode_problem["fit"]
    index = layout.names.index("rho_c[C3]")
    grid = fit.theta[index] + np.linspace(-0.3, 0.3, 7)
    profile = profile_likelihood(ode_problem["sse_fn"], fit.theta, index, grid)
    assert abs(int(np.argmin(profile.loss)) - 3) <= 1
    interval = likelihood_interval(grid, sse_deviance(profile.loss, ode_problem["n_obs"], sse_min=fit.sse))
    assert interval.lo_bounded and interval.hi_bounded
    assert interval.lo <= fit.theta[index] <= interval.hi


@pytest.mark.slow
def test_ode_gauss_newton_spectrum_at_the_mle(x64, ode_problem):
    spectrum = gauss_newton_spectrum(ode_problem["residual_fn"], ode_problem["fit"].theta, mode="rev")
    eig = spectrum.eigenvalues
    assert eig.shape == (13,) and spectrum.eigenvectors.shape == (13, 13)
    assert np.all(np.diff(eig) <= 0)
    assert np.all(eig >= -1e-9 * eig[0])
