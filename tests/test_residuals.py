import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexpinn.physics.diagnostics import detect_collapse, residual_rms, wear_violation_rate
from apexpinn.physics.residuals import (
    POSITIVITY_FLOOR,
    PhysicsParams,
    observation,
    states_and_rates,
    thermal_residual,
    thermal_rhs,
    wear_residual,
    wear_rhs,
)

NON_INTEGER_TAU = (0.5, 1.25, 2.75, 4.1)


@pytest.fixture
def x64():
    """Enable float64 for one test only, then restore the previous setting."""
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _params(**overrides) -> PhysicsParams:
    values = {"k": 0.8, "rho": 0.05, "gamma": 1.2, "h_opt": 0.4, "kappa": 1.5}
    values.update(overrides)
    return PhysicsParams.from_constrained(**values)


def test_from_constrained_round_trip():
    k = np.array([0.3, 0.5, 0.8, 1.1, 1.6])
    params = PhysicsParams.from_constrained(k=k, rho=0.05, gamma=1.2, h_opt=0.4, kappa=1.5)
    np.testing.assert_allclose(np.asarray(params.k), k, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(params.rho), 0.05, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(params.gamma), 1.2, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(params.h_opt), 0.4, rtol=1e-5)
    np.testing.assert_allclose(float(params.kappa), 1.5, rtol=1e-5)


def test_from_constrained_names_bad_compound():
    with pytest.raises(ValueError, match=r"rho.*C3"):
        PhysicsParams.from_constrained(
            k=0.8, rho=[0.05, 0.05, POSITIVITY_FLOOR, 0.05, 0.05], gamma=1.2, h_opt=0.4, kappa=1.5
        )
    with pytest.raises(ValueError, match=r"h_opt.*C1"):
        PhysicsParams.from_constrained(
            k=0.8, rho=0.05, gamma=1.2, h_opt=[1.0, 0.4, 0.4, 0.4, 0.4], kappa=1.5
        )
    with pytest.raises(ValueError, match=r"k must be a scalar or have shape \(5,\)"):
        PhysicsParams.from_constrained(k=[0.8, 0.8], rho=0.05, gamma=1.2, h_opt=0.4, kappa=1.5)


# Acceptance test 1
def test_autodiff_rates_match_analytic_derivatives():
    def latent(t):
        return t * jnp.sin(t), t**2 * jnp.exp(-0.1 * t)

    tau = np.array(NON_INTEGER_TAU)
    h, w, dh, dw = states_and_rates(latent, tau)

    np.testing.assert_allclose(np.asarray(h), tau * np.sin(tau), rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(w), tau**2 * np.exp(-0.1 * tau), rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(dh), np.sin(tau) + tau * np.cos(tau), rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(dw), (2 * tau - 0.1 * tau**2) * np.exp(-0.1 * tau), rtol=0, atol=1e-5
    )


# Acceptance test 2
def test_residuals_vanish_on_true_ode_solution(x64):
    compound = 2
    params = _params(k=[0.3, 0.5, 0.8, 1.1, 1.6], h_opt=0.4)

    def q(t):
        return 0.6 + 0.2 * jnp.sin(0.5 * t)

    def vector_field(t, y, args):
        h, _ = y
        return thermal_rhs(params, compound, h, q(t)), wear_rhs(params, compound, h)

    t1 = 20.0
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=0.1,
        y0=(jnp.zeros(()), jnp.zeros(())),
        saveat=diffrax.SaveAt(dense=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=4096,
    )

    def latent(t):
        h, w = sol.evaluate(t)
        return h, w

    tau = jnp.linspace(0.05, t1, 397)  # non-integer collocation points
    h, _, dh, dw = states_and_rates(latent, tau)
    assert h.dtype == jnp.float64
    # The overheating branch is active on part of the grid, so the ReLU term is exercised.
    assert bool(jnp.any(h > params.h_opt[compound])) and bool(jnp.any(h < params.h_opt[compound]))

    r_t = thermal_residual(params, compound, h, dh, q(tau))
    r_w = wear_residual(params, compound, h, dw)
    assert float(residual_rms(r_t)) < 1e-4
    assert float(residual_rms(r_w)) < 1e-4


# Acceptance test 3
def test_wear_violation_rate_extremes():
    tau = jnp.linspace(0.1, 10.0, 50)

    _, _, _, dw_decreasing = states_and_rates(lambda t: (0.0 * t, 5.0 - 0.1 * t), tau)
    _, _, _, dw_monotone = states_and_rates(lambda t: (0.0 * t, 0.02 * t**2), tau)

    assert float(wear_violation_rate(dw_decreasing)) == 1.0
    assert float(wear_violation_rate(dw_monotone)) == 0.0


# Acceptance test 4
def test_collapse_detector_flags_rates_near_floor():
    healthy = _params()
    assert not detect_collapse(healthy).collapsed

    k = [0.8, POSITIVITY_FLOOR * 1.005, 0.8, 0.8, 0.8]
    rho = [0.05, 0.05, 0.05, POSITIVITY_FLOOR * 1.009, POSITIVITY_FLOOR * 1.02]
    report = detect_collapse(_params(k=k, rho=rho))

    assert report.collapsed
    assert report.flagged == ("k_c[C2]", "rho_c[C4]")  # C5 at 2% above the floor is not flagged


def test_observation_combines_thermal_penalty_and_wear():
    params = _params(gamma=2.0, h_opt=0.5)
    y_hat = observation(params, 0, h=jnp.array([0.0, 0.5]), w=jnp.array([0.0, 0.3]), b=90.0)
    np.testing.assert_allclose(np.asarray(y_hat), [90.0 + 2.0 * 0.25, 90.3], rtol=1e-6)
