import inspect
import json
import time
from datetime import datetime

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apexpinn.models.ode_model as ode_module
import apexpinn.models.pinn as pinn_module
from apexpinn.contracts import PredictionOutput
from apexpinn.models.ode_model import ODEModel
from apexpinn.models.pinn import PINN
from apexpinn.models.stints import (
    CTX_DIM,
    SYNTHETIC_K_MIN,
    TRACK_TEMP_REF_C,
    Forecaster,
    Scenario,
    StintContext,
    StintHistory,
    anchor_offset,
    batch_from_synthetic,
    build_prediction,
    ctx_features,
    forecast_q_knots,
    history_from_synthetic,
)
from apexpinn.physics import residuals
from apexpinn.physics.diagnostics import detect_collapse, residual_rms
from apexpinn.physics.residuals import (
    COMPOUNDS,
    N_COMPOUNDS,
    PhysicsParams,
    states_and_rates,
    thermal_residual,
    wear_residual,
)
from apexpinn.physics.synthetic import default_theta_true, make_synthetic_dataset
from apexpinn.train import trainer as trainer_module
from apexpinn.train.run_meta import _git_commit, run_metadata
from apexpinn.train.trainer import (
    PHYSICS_TERMS,
    LossWeights,
    TrainConfig,
    TrainLog,
    curriculum_scale,
    init_normalizers,
    sample_collocation,
    train,
)

NON_INTEGER_TAU = (0.5, 1.25, 2.75, 4.1)
TARGET_NAMES = {"y", "y_true", "target", "y_future", "targets"}


@pytest.fixture
def x64():
    """Enable float64 for one test only, then restore the previous setting."""
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def small_ds():
    return make_synthetic_dataset(jax.random.key(0), n_stints=6)


@pytest.fixture(scope="module")
def small_batch(small_ds):
    return batch_from_synthetic(small_ds)


@pytest.fixture(scope="module")
def ode_run(small_ds):
    """One shared ODEModel train() call: Diffrax forward and backward compiles are the slowest part of the
    fast suite, so AT2, the data-loss check and the fit/forecast check all read this run."""
    batch = batch_from_synthetic(small_ds, observe_states=True)
    model = ODEModel.init()
    weights = LossWeights(data=1.0, state=1.0)
    cfg = TrainConfig(n_steps=20, learning_rate=1e-2, n_warm=0, n_ramp=0, log_every=19)
    fitted, log = train(model, batch, weights, cfg, jax.random.key(3))
    return model, fitted, log, weights


def _tiny_pinn(seed: int = 0) -> PINN:
    return PINN.init(jax.random.key(seed), width=16, depth=2)


# Acceptance test 1
def test_pinn_hard_initial_conditions_exact():
    for seed in range(3):
        k_net, k_ctx = jax.random.split(jax.random.key(seed))
        model = PINN.init(k_net, width=16, depth=2)
        ctx = jax.random.normal(k_ctx, (CTX_DIM,))
        h, w = model.latent(jnp.asarray(0.0), ctx)
        assert float(h) == 0.0
        assert float(w) == 0.0


def test_pinn_latent_and_rates_match_finite_difference(x64):
    k_net, k_ctx = jax.random.split(jax.random.key(7))
    model = PINN.init(k_net, width=16, depth=2)
    ctx = jax.random.normal(k_ctx, (CTX_DIM,))
    tau = jnp.asarray(NON_INTEGER_TAU)
    h, w, dh, dw = model.latent_and_rates(tau, ctx)

    def latent(t):
        return model.latent(t, ctx)

    eps = 1e-3
    h_plus, w_plus = jax.vmap(latent)(tau + eps)
    h_minus, w_minus = jax.vmap(latent)(tau - eps)
    np.testing.assert_allclose(dh, (h_plus - h_minus) / (2 * eps), atol=1e-5)
    np.testing.assert_allclose(dw, (w_plus - w_minus) / (2 * eps), atol=1e-5)

    expected = states_and_rates(latent, tau)
    for got, want in zip((h, w, dh, dw), expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


# Acceptance test 4
def test_m1_and_m2_share_physics_and_observation(x64):
    assert type(PINN.init(jax.random.key(0), width=16, depth=2).physics) is PhysicsParams
    assert type(ODEModel.init().physics) is PhysicsParams
    assert pinn_module.observation is residuals.observation
    assert ode_module.observation is residuals.observation

    theta = default_theta_true()
    ode = ODEModel.init(physics=theta)
    ds = make_synthetic_dataset(jax.random.key(3), n_stints=2, theta_true=theta)
    tau = jnp.linspace(0.3, 30.7, 97)
    for s in range(2):
        c = ds.compound_idx[s]
        q_knots = ds.q[s]
        h, w, dh, dw = states_and_rates(ode.dense_latent(c, q_knots), tau)
        knots = jnp.arange(q_knots.shape[0], dtype=tau.dtype)
        q = jnp.interp(tau, knots, q_knots)
        assert float(residual_rms(thermal_residual(theta, c, h, dh, q))) < 1e-3
        assert float(residual_rms(wear_residual(theta, c, h, dw))) < 1e-3

        n = int(ds.n_laps[s])
        h_int, w_int = ode.solve(c, q_knots, ds.tau[:n])
        np.testing.assert_allclose(h_int, ds.h[s, :n], atol=1e-4)
        np.testing.assert_allclose(w_int, ds.w[s, :n], atol=1e-4)


# Acceptance test 6
def test_forecast_signature_has_no_target_argument():
    for cls in (PINN, ODEModel):
        params = list(inspect.signature(cls.forecast).parameters)
        assert params == ["self", "history", "horizon", "scenario"]
        assert not set(params) & TARGET_NAMES
    assert isinstance(_tiny_pinn(), Forecaster)
    assert isinstance(ODEModel.init(), Forecaster)


def test_ctx_features_layout():
    ctx = ctx_features(2, 10.0, True, TRACK_TEMP_REF_C + 10.0, 0.6)
    assert ctx.shape == (CTX_DIM,)
    assert CTX_DIM == N_COMPOUNDS + 4
    np.testing.assert_allclose(ctx, [0, 0, 1, 0, 0, 1.0, 1.0, 1.0, 0.6], rtol=1e-6)
    batched = ctx_features(jnp.array([0, 4]), 0.0, False, TRACK_TEMP_REF_C, jnp.array([0.5, 0.7]))
    assert batched.shape == (2, CTX_DIM)
    np.testing.assert_allclose(batched[:, N_COMPOUNDS - 1], [0.0, 1.0])
    np.testing.assert_allclose(batched[:, -1], [0.5, 0.7], rtol=1e-6)


def test_stint_context_validates():
    with pytest.raises(ValueError, match="compound_idx"):
        StintContext(compound_idx=5, age_at_start=0.0, fresh=True, track_temp_start=35.0, circuit_q_mean=0.5)
    with pytest.raises(ValueError, match="track_temp_start"):
        StintContext(
            compound_idx=1, age_at_start=0.0, fresh=True, track_temp_start=np.nan, circuit_q_mean=0.5
        )


def test_batch_from_synthetic_shapes(small_ds, small_batch):
    n, k = small_ds.y.shape
    assert small_batch.ctx.shape == (n, CTX_DIM)
    assert small_batch.q.shape == (n, k + 1)
    assert small_batch.h_true is None and small_batch.w_true is None
    onehot = np.asarray(small_batch.ctx[:, :N_COMPOUNDS])
    np.testing.assert_array_equal(onehot.argmax(axis=1), np.asarray(small_ds.compound_idx))
    observed = batch_from_synthetic(small_ds, observe_states=True)
    np.testing.assert_array_equal(observed.h_true, small_ds.h)
    np.testing.assert_array_equal(observed.w_true, small_ds.w)


def test_history_from_synthetic_excludes_future(small_ds):
    history, scenario, y_true = history_from_synthetic(small_ds, stint=0, t0=8, horizon=5)
    tau_all = np.asarray(small_ds.tau)
    expected_tau = tau_all[np.asarray(small_ds.mask[0]) & (tau_all <= 8)]
    assert history.t0 == 8
    assert history.tau.max() == 8
    np.testing.assert_array_equal(history.tau, expected_tau)
    np.testing.assert_allclose(history.y, np.asarray(small_ds.y[0])[expected_tau.astype(int) - 1])
    np.testing.assert_allclose(history.q, np.asarray(small_ds.q[0])[expected_tau.astype(int)])
    np.testing.assert_allclose(y_true, np.asarray(small_ds.y[0, 8:13]))
    np.testing.assert_allclose(scenario.q_bar, np.asarray(small_ds.q[0, 6:9]).mean(), rtol=1e-6)
    assert history.race_lap_t0 == int(small_ds.race_lap[0, 7])

    n = int(small_ds.n_laps[0])
    with pytest.raises(ValueError, match="horizon"):
        history_from_synthetic(small_ds, stint=0, t0=n - 2, horizon=3)
    with pytest.raises(ValueError, match="t0"):
        history_from_synthetic(small_ds, stint=0, t0=0, horizon=3)
    with pytest.raises(ValueError, match="t0"):
        history_from_synthetic(small_ds, stint=0, t0=SYNTHETIC_K_MIN - 1, horizon=3)


def test_synthetic_context_uses_no_post_origin_heat_load(small_ds):
    """PROJECT 4.5: features never use post-t0 information. Changing q after the earliest valid origin
    (t0 = k_min) must leave the forecast context, the history and the training context unchanged."""
    t0 = SYNTHETIC_K_MIN
    q = np.asarray(small_ds.q)
    future = np.arange(q.shape[1])[None, :] > t0
    perturbed_q = jnp.asarray(np.where(future, 1.0 - q, q), dtype=q.dtype)
    perturbed = eqx.tree_at(lambda d: d.q, small_ds, perturbed_q)
    for stint in range(2):
        before, scenario, _ = history_from_synthetic(small_ds, stint=stint, t0=t0, horizon=5)
        after, scenario_after, _ = history_from_synthetic(perturbed, stint=stint, t0=t0, horizon=5)
        assert before.ctx == after.ctx
        np.testing.assert_array_equal(before.q, after.q)
        np.testing.assert_array_equal(before.y, after.y)
        assert scenario == scenario_after
    np.testing.assert_array_equal(batch_from_synthetic(small_ds).ctx, batch_from_synthetic(perturbed).ctx)


def test_stint_history_rejects_laps_after_origin():
    ctx = StintContext(
        compound_idx=1, age_at_start=0.0, fresh=True, track_temp_start=35.0, circuit_q_mean=0.5
    )
    with pytest.raises(ValueError, match="t0"):
        StintHistory(
            stint_id="s",
            t0=3,
            tau=np.array([1.0, 2.0, 4.0]),
            y=np.zeros(3),
            q=np.full(3, 0.5),
            ctx=ctx,
            race_lap_t0=10,
        )
    with pytest.raises(ValueError, match="increasing"):
        StintHistory(
            stint_id="s",
            t0=3,
            tau=np.array([2.0, 1.0]),
            y=np.zeros(2),
            q=np.full(2, 0.5),
            ctx=ctx,
            race_lap_t0=10,
        )
    with pytest.raises(ValueError, match="q_bar"):
        Scenario(q_bar=1.5)


def test_forecast_q_knots(small_ds):
    history, scenario, _ = history_from_synthetic(small_ds, stint=1, t0=8, horizon=5)
    knots = forecast_q_knots(history, 5, scenario)
    assert knots.shape == (8 + 5 + 1,)
    np.testing.assert_allclose(knots[9:], scenario.q_bar)
    np.testing.assert_allclose(knots[history.tau.astype(int)], history.q)
    np.testing.assert_allclose(knots[0], history.q[0])


def test_anchor_offset_and_build_prediction():
    y = np.array([1.0, 2.0, 4.0])
    shape = np.array([0.5, 1.0, 2.5])
    b, resid_var = anchor_offset(y, shape)
    assert b == pytest.approx(1.0)
    assert resid_var == pytest.approx(1.0 / 6.0)

    future = np.array([[3.0, 3.5]])
    out = build_prediction(shape[None, :], future, y, noise_s=0.2)
    assert isinstance(out, PredictionOutput)
    assert out.member_means_s.shape == (1, 2)
    np.testing.assert_allclose(out.mean_s, [4.0, 4.5])
    # Offset-estimate variance max(resid_var, noise^2) / n_hist with n_hist = 3, not the raw residual
    # variance. Here resid_var = 1/6 > noise^2 = 0.04, so the misfit widens the offset term.
    np.testing.assert_allclose(out.sigma_s, np.sqrt(1.0 / 18.0 + 0.04))
    assert out.latent_h is None and out.latent_w_s is None

    two = build_prediction(np.stack([shape, shape + 1.0]), np.repeat(future, 2, axis=0), y, noise_s=0.2)
    np.testing.assert_allclose(two.member_means_s, [[4.0, 4.5], [3.0, 3.5]])
    np.testing.assert_allclose(two.mean_s, [3.5, 4.0])
    np.testing.assert_allclose(two.sigma_s, np.sqrt(1.0 / 18.0 + 0.04 + 0.25))

    # resid_var = 1/6 < noise^2 = 0.25: the offset term is floored at noise^2 / n_hist = 0.25 / 3.
    noisy = build_prediction(shape[None, :], future, y, noise_s=0.5)
    np.testing.assert_allclose(noisy.sigma_s, np.sqrt(0.25 / 3.0 + 0.25))
    # A perfect history fit (resid_var = 0) still keeps the offset term at noise^2 / n_hist.
    exact = build_prediction(shape[None, :], future, shape + 1.0, noise_s=0.2)
    np.testing.assert_allclose(exact.sigma_s, np.sqrt(0.04 / 3.0 + 0.04))


@pytest.mark.parametrize("model_name", ["pinn", "ode"])
def test_forecast_outputs_valid_prediction(small_ds, model_name):
    model = _tiny_pinn(1) if model_name == "pinn" else ODEModel.init()
    history, scenario, _ = history_from_synthetic(small_ds, stint=0, t0=8, horizon=5)
    out = model.forecast(history, horizon=5, scenario=scenario)
    assert isinstance(out, PredictionOutput)
    assert out.mean_s.shape == (5,)
    assert out.member_means_s.shape == (1, 5)
    assert np.all(out.sigma_s > 0)
    assert out.latent_h.shape == (5,)
    assert out.latent_w_s.shape == (5,)
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(history, horizon=0, scenario=scenario)


def test_loss_terms_finite(small_ds, small_batch):
    observed = batch_from_synthetic(small_ds, observe_states=True)
    n = observed.n_stints
    b = jnp.zeros(n)
    colloc = jnp.tile(jnp.linspace(0.0, 45.0, 8)[None, :], (n, 1))
    pinn = _tiny_pinn(2)

    def check(terms, expected):
        assert set(terms) == expected
        for name, value in terms.items():
            assert jnp.shape(value) == (), name
            assert np.isfinite(float(value)), name

    check(pinn.loss_terms(b, observed, colloc), {"data", "thermal", "wear", "mono", "reg", "state"})
    check(pinn.loss_terms(b, small_batch, colloc), {"data", "thermal", "wear", "mono", "reg"})
    ode = ODEModel.init()
    check(ode.loss_terms(b, observed), {"data", "state"})
    check(ode.loss_terms(b, small_batch, None), {"data"})


def test_pinn_loss_ignores_heat_load_knots_past_stint_end(small_batch):
    """StintBatch padding contract: q knots past n_laps never enter the PINN loss, so zero padding
    (as real data might use) gives the same terms as the synthetic q process."""
    knots = np.arange(small_batch.q.shape[1])
    past = knots[None, :] > np.asarray(small_batch.n_laps)[:, None]
    assert past.any()
    zeroed = eqx.tree_at(lambda bt: bt.q, small_batch, jnp.where(jnp.asarray(past), 0.0, small_batch.q))
    pinn = _tiny_pinn(2)
    b = jnp.zeros(small_batch.n_stints)
    colloc = sample_collocation(jax.random.key(6), small_batch.n_laps, 8, 10.0)
    assert np.any(np.asarray(colloc) > np.asarray(small_batch.n_laps)[:, None])
    original = pinn.loss_terms(b, small_batch, colloc)
    padded = pinn.loss_terms(b, zeroed, colloc)
    for term in ("thermal", "wear", "data"):
        np.testing.assert_allclose(float(padded[term]), float(original[term]), rtol=1e-6, err_msg=term)


def test_train_config_and_weights_validate():
    assert hash(TrainConfig()) == hash(TrainConfig())
    assert hash(LossWeights()) == hash(LossWeights())
    with pytest.raises(ValueError, match="data"):
        LossWeights(data=0.0)
    with pytest.raises(ValueError, match="thermal"):
        LossWeights(thermal=-1.0)
    with pytest.raises(ValueError, match="n_steps"):
        TrainConfig(n_steps=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="log_every"):
        TrainConfig(log_every=0)


def _to_float32(tree):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, tree)


def _leaves_differ(a, b) -> bool:
    pairs = zip(
        jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_inexact_array)),
        jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_inexact_array)),
        strict=True,
    )
    return any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in pairs)


def test_curriculum_scale():
    expected = [0.0] * 10 + [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    got = [float(curriculum_scale(step, 10, 4)) for step in range(16)]
    np.testing.assert_allclose(got, expected)
    assert float(curriculum_scale(9, 10, 0)) == 0.0
    assert float(curriculum_scale(10, 10, 0)) == 1.0
    assert float(curriculum_scale(25, 10, 0)) == 1.0
    jitted = jax.jit(lambda step: curriculum_scale(step, 10, 4))
    traced = [float(jitted(jnp.asarray(step, dtype=jnp.int32))) for step in range(16)]
    np.testing.assert_allclose(traced, expected)
    assert set(PHYSICS_TERMS) == {"thermal", "wear", "mono"}


def test_sample_collocation_domain(small_batch):
    n = small_batch.n_stints
    colloc = np.asarray(sample_collocation(jax.random.key(4), small_batch.n_laps, 8, 10.0))
    assert colloc.shape == (n, 8)
    upper = np.asarray(small_batch.n_laps, dtype=np.float64)[:, None] + 10.0
    assert np.all(colloc >= 0.0)
    assert np.all(colloc <= upper)
    assert np.any(colloc != np.round(colloc))
    # Stratified: point j of stint s lies in [j, j + 1) * (T_s + 10) / 8, up to float32 rounding.
    offset = colloc / upper * 8 - np.arange(8)[None, :]
    assert np.all(offset >= -1e-4) and np.all(offset <= 1.0 + 1e-4)
    again = np.asarray(sample_collocation(jax.random.key(4), small_batch.n_laps, 8, 10.0))
    np.testing.assert_array_equal(colloc, again)
    other = np.asarray(sample_collocation(jax.random.key(5), small_batch.n_laps, 8, 10.0))
    assert not np.array_equal(colloc, other)


# Acceptance test 2
@pytest.mark.parametrize("model_name", ["pinn", "ode"])
def test_normalized_terms_start_at_one(request, small_ds, model_name):
    if model_name == "pinn":
        batch = batch_from_synthetic(small_ds, observe_states=True)
        weights = LossWeights(data=1.0, thermal=1.0, wear=1.0, mono=0.1, reg=1e-4, state=1.0)
        model = _tiny_pinn(3)
        cfg = TrainConfig(n_steps=2, log_every=1, n_warm=0, n_ramp=0, n_colloc=8)
        fitted, log = train(model, batch, weights, cfg, jax.random.key(0))
        expected = {"data", "thermal", "wear", "state"}
    else:
        model, fitted, log, weights = request.getfixturevalue("ode_run")  # n_warm = n_ramp = 0 as well
        batch = batch_from_synthetic(small_ds, observe_states=True)
        expected = {"data", "state"}
    assert type(fitted) is type(model)
    assert isinstance(log, TrainLog)
    assert log.steps[0] == 0
    assert expected <= set(log.normalizers)
    assert set(log.normalized) == set(log.normalizers) == set(log.coefficients)
    defaults = TrainConfig()  # both runs keep the default norm_floor and norm_rel_floor
    data0 = log.terms["data"][0]
    for term, normalizer in log.normalizers.items():
        assert log.coefficients[term] == pytest.approx(getattr(weights, term) / normalizer)
        raw0 = log.terms[term][0]
        if term in log.floored_terms:
            # Accepted exception (PLAN T4 AT2): a floored term starts at raw0 / floor, not at 1.
            floor = (
                defaults.norm_floor
                if term == "data"
                else max(defaults.norm_floor, defaults.norm_rel_floor * data0)
            )
            assert raw0 <= floor
            assert normalizer == pytest.approx(floor)
            assert log.normalized[term][0] == pytest.approx(raw0 / floor, rel=1e-6, abs=1e-12), term
            continue
        assert normalizer == pytest.approx(raw0)
        assert abs(log.normalized[term][0] - 1.0) <= 1e-6, term
    # The jitted objective Adam differentiates (not a host recomputation): at step 0 with no warm-up and no
    # ramp, every non-floored term contributes exactly its weight. Dropping the normalizers fails this.
    expected_objective = sum(
        getattr(weights, t) * (1.0 if t not in log.floored_terms else log.terms[t][0] / log.normalizers[t])
        for t in log.normalizers
    )
    assert log.objective[0] == pytest.approx(expected_objective, rel=1e-5)
    assert log.b_train.shape == (batch.n_stints,)
    assert np.all(np.isfinite(log.b_train))


def test_init_normalizers_relative_floor():
    raw0 = {"data": 2.0, "thermal": 5.0, "mono": 1e-8, "state": 0.0, "reg": 3.0}
    active = ("data", "thermal", "mono", "state")
    normalizers, floored = init_normalizers(raw0, active, norm_floor=1e-12, norm_rel_floor=1e-3)
    assert set(normalizers) == set(active)
    assert normalizers["data"] == 2.0 and normalizers["thermal"] == 5.0
    assert normalizers["mono"] == pytest.approx(2e-3)
    assert normalizers["state"] == pytest.approx(2e-3)
    assert floored == ("mono", "state")
    # Continuous in the init value: 0 and 1e-11 get the same normalizer (no 1e10 coefficient).
    zero, _ = init_normalizers({"data": 1.0, "mono": 0.0}, ("data", "mono"), 1e-12, 1e-3)
    tiny, _ = init_normalizers({"data": 1.0, "mono": 1e-11}, ("data", "mono"), 1e-12, 1e-3)
    assert zero["mono"] == tiny["mono"] == pytest.approx(1e-3)
    # norm_rel_floor = 0 falls back to the absolute floor.
    absolute, floored = init_normalizers({"data": 1.0, "mono": 0.0}, ("data", "mono"), 1e-12, 0.0)
    assert absolute["mono"] == 1e-12 and floored == ("mono",)
    with pytest.raises(ValueError, match="data"):
        init_normalizers({"mono": 1.0}, ("mono",), 1e-12, 1e-3)
    with pytest.raises(ValueError, match="norm_rel_floor"):
        TrainConfig(norm_rel_floor=-1.0)


def test_train_rejects_state_weight_without_true_states(small_batch):
    assert small_batch.h_true is None
    cfg = TrainConfig(n_steps=1)
    with pytest.raises(ValueError, match="observe_states"):
        train(_tiny_pinn(), small_batch, LossWeights(state=1.0), cfg, jax.random.key(0))


# Acceptance test 3
def test_trainlog_has_grad_norms_at_three_steps(small_batch):
    cfg = TrainConfig(n_steps=10, log_every=3, n_colloc=8, n_warm=2, n_ramp=4)
    _, log = train(_tiny_pinn(4), small_batch, LossWeights(), cfg, jax.random.key(1))
    np.testing.assert_array_equal(log.steps, [0, 3, 6, 9])
    assert {"data", "thermal", "wear", "mono", "reg"} <= set(log.grad_norms)
    assert set(log.grad_norms) == set(log.normalizers)
    for term, norms in log.grad_norms.items():
        assert len(norms) == len(log.steps), term
        finite_steps = set(log.steps[np.isfinite(norms)].tolist())
        assert len(finite_steps) >= 3, term
    np.testing.assert_allclose(log.curriculum, [0.0, 0.5, 1.0, 1.0])
    assert len(log.total) == len(log.steps) == len(log.objective)
    expected_total = sum(
        getattr(LossWeights(), t) * (log.curriculum if t in PHYSICS_TERMS else 1.0) * log.normalized[t]
        for t in log.normalized
    )
    np.testing.assert_allclose(log.total, expected_total, rtol=1e-6)
    # The jitted objective uses the same normalizers and curriculum (step 0 is data-only, step 3 is half
    # ramped), so dropping either from the objective makes it disagree with the logged raw terms.
    np.testing.assert_allclose(log.objective, log.total, rtol=1e-4)
    assert log.wall_time_s > 0


def test_training_reduces_data_loss(small_batch, ode_run):
    cfg = TrainConfig(n_steps=150, learning_rate=1e-2, n_warm=50, n_ramp=50, n_colloc=16, log_every=50)
    _, log = train(_tiny_pinn(5), small_batch, LossWeights(), cfg, jax.random.key(2))
    np.testing.assert_array_equal(log.steps, [0, 50, 100, 149])
    assert log.terms["data"][-1] < log.terms["data"][0]

    ode, fitted, ode_log, _ = ode_run
    np.testing.assert_array_equal(ode_log.steps, [0, 19])
    assert set(ode_log.terms) == {"data", "state"}
    assert ode_log.objective[-1] < ode_log.objective[0]
    assert ode_log.terms["data"][-1] <= ode_log.terms["data"][0]
    assert _leaves_differ(ode.physics, fitted.physics)


def test_float64_mode_requires_x64(small_batch):
    if jax.config.x64_enabled:
        pytest.skip("jax_enable_x64 is already on")
    cfg = TrainConfig(n_steps=1, float64=True)
    with pytest.raises(ValueError, match="jax_enable_x64"):
        train(_tiny_pinn(), small_batch, LossWeights(), cfg, jax.random.key(0))


def test_float64_mode_casts_to_float64(x64, small_batch):
    model = _to_float32(_tiny_pinn(6))
    batch = _to_float32(small_batch)
    assert model.physics.raw_k.dtype == jnp.float32
    cfg = TrainConfig(n_steps=2, log_every=1, n_colloc=8, float64=True)
    fitted, log = train(model, batch, LossWeights(), cfg, jax.random.key(0))
    assert fitted.physics.raw_k.dtype == jnp.float64
    assert fitted.net.layers[0].weight.dtype == jnp.float64
    assert fitted.noise_s.dtype == jnp.float64
    assert log.b_train.dtype == np.float64


def test_pinn_fit_returns_forecaster_and_warm_up_freezes_physics_only_parameters(small_ds, small_batch):
    # Default n_warm = 200 >= n_steps, so every step is data-only. k_c enters only the thermal residual and
    # rho_c, kappa only the wear residual, so they must stay bit-identical; gamma enters the data term.
    cfg = TrainConfig(n_steps=5, log_every=5, n_colloc=8)
    assert cfg.n_warm >= cfg.n_steps
    model = PINN.init(jax.random.key(8), width=16, depth=2, train_cfg=cfg)
    fitted = model.fit(small_batch, jax.random.key(9))
    assert type(fitted) is PINN
    assert isinstance(fitted, Forecaster)
    assert float(fitted.noise_s) > 0
    for name in ("raw_k", "raw_rho", "raw_kappa"):
        np.testing.assert_array_equal(getattr(fitted.physics, name), getattr(model.physics, name), name)
    assert not np.array_equal(np.asarray(fitted.physics.raw_gamma), np.asarray(model.physics.raw_gamma))
    _check_fitted_forecast(small_ds, fitted)


def test_ode_fit_returns_forecaster(small_ds, ode_run):
    model, fitted, _, _ = ode_run
    assert type(fitted) is ODEModel
    assert isinstance(fitted, Forecaster)
    assert float(fitted.noise_s) > 0
    assert _leaves_differ(model.physics, fitted.physics)
    _check_fitted_forecast(small_ds, fitted)


def test_ode_fit_delegates_to_trainer(monkeypatch):
    """ODEModel.fit passes its own weights and train_cfg to trainer.train and returns the fitted model.
    Checked with a stub so the fast suite compiles Diffrax training only once (ode_run)."""
    calls = []
    sentinel = object()

    def fake_train(model, data, weights, cfg, key):
        calls.append((model, data, weights, cfg))
        return sentinel, None

    monkeypatch.setattr(trainer_module, "train", fake_train)
    cfg, weights = TrainConfig(n_steps=3), LossWeights(state=0.5)
    model = ODEModel.init(train_cfg=cfg, weights=weights)
    assert model.fit("batch", jax.random.key(0)) is sentinel
    assert len(calls) == 1
    got_model, got_data, got_weights, got_cfg = calls[0]
    assert got_model is model and got_data == "batch"
    assert got_weights == weights and got_cfg == cfg


def _check_fitted_forecast(ds, fitted):
    history, scenario, _ = history_from_synthetic(ds, stint=2, t0=8, horizon=5)
    out = fitted.forecast(history, horizon=5, scenario=scenario)
    assert isinstance(out, PredictionOutput)
    assert out.mean_s.shape == (5,)
    assert np.all(out.sigma_s > 0)


def test_run_metadata():
    meta = run_metadata(seeds=[0, 1], data_hash="abc123")
    expected = {
        "python",
        "packages",
        "devices",
        "dtype",
        "seeds",
        "git_commit",
        "data_hash",
        "timestamp_utc",
    }
    assert expected <= set(meta)
    assert isinstance(meta["python"], str)
    for name in ("jax", "jaxlib", "equinox", "diffrax", "optax"):
        assert isinstance(meta["packages"][name], str), name
    assert meta["devices"] and all(isinstance(d, str) for d in meta["devices"])
    assert meta["dtype"] == ("float64" if jax.config.x64_enabled else "float32")
    assert meta["seeds"] == [0, 1]
    assert meta["git_commit"] is None or isinstance(meta["git_commit"], str)
    assert meta["data_hash"] == "abc123"
    assert datetime.fromisoformat(meta["timestamp_utc"]).tzinfo is not None
    json.dumps(meta)
    assert run_metadata(dtype="float64")["dtype"] == "float64"
    with pytest.raises(ValueError, match="dtype"):
        run_metadata(dtype="float16")


def test_git_commit_follows_worktree_commondir(tmp_path):
    main_sha, packed_sha = "a" * 40, "b" * 40
    git = tmp_path / "main" / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "feature").write_text(main_sha + "\n", encoding="utf-8")
    (git / "packed-refs").write_text(f"# pack-refs\n{packed_sha} refs/heads/packed\n", encoding="utf-8")
    worktree_git = git / "worktrees" / "wt"
    worktree_git.mkdir(parents=True)
    (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
    (worktree_git / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")
    worktree = tmp_path / "wt"
    (worktree / "src").mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")

    assert _git_commit(worktree / "src") == main_sha  # loose branch ref lives in the common directory
    (worktree_git / "HEAD").write_text("ref: refs/heads/packed\n", encoding="utf-8")
    assert _git_commit(worktree) == packed_sha  # packed-refs of the common directory
    (worktree_git / "HEAD").write_text("ref: refs/heads/missing\n", encoding="utf-8")
    assert _git_commit(worktree) is None
    assert _git_commit(tmp_path / "main") is None  # main HEAD points at refs/heads/main, which is absent


# Acceptance test 5 (PRD S1a): long-running, run with -m slow
@pytest.mark.slow
def test_s1a_recovers_k_and_rho_with_states_observed(x64):
    ds = make_synthetic_dataset(jax.random.key(1))
    batch = batch_from_synthetic(ds, observe_states=True)
    model = PINN.init(jax.random.key(2), width=32, depth=2)  # starts from default_physics(), not the truth
    weights = LossWeights(data=1.0, thermal=1.0, wear=1.0, mono=0.1, reg=1e-6, state=1.0)
    cfg = TrainConfig(
        n_steps=4000,
        learning_rate=3e-3,
        n_warm=100,
        n_ramp=400,
        n_colloc=32,
        log_every=500,
        float64=True,
    )
    start = time.perf_counter()
    fitted, _ = train(model, batch, weights, cfg, jax.random.key(3))
    elapsed = time.perf_counter() - start
    assert elapsed < 180.0, f"training took {elapsed:.1f} s"

    theta = ds.theta_true
    k_hat, rho_hat = np.asarray(fitted.physics.k), np.asarray(fitted.physics.rho)
    k_true, rho_true = np.asarray(theta.k), np.asarray(theta.rho)
    for name in ("C2", "C3", "C4"):
        c = COMPOUNDS.index(name)
        assert abs(k_hat[c] / k_true[c] - 1.0) < 0.05, (name, k_hat[c], k_true[c])
        assert abs(rho_hat[c] / rho_true[c] - 1.0) < 0.05, (name, rho_hat[c], rho_true[c])
    assert not detect_collapse(fitted.physics).collapsed
