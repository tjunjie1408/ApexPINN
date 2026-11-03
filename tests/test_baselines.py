"""T12 model oracles and forecast-safe baseline contracts. No real-data G1 evaluation."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apexpinn.models.stints import Scenario, StintContext, StintHistory


def linear_history(n=12, *, slope=0.12, intercept=90.0):
    tau = np.arange(1, n + 1, dtype=float)
    return StintHistory(
        "crafted_linear",
        n,
        tau,
        intercept + slope * tau,
        np.full(n, 0.5),
        StintContext(0, 0.0, True, 35.0, 0.5),
        n,
    )


def test_b2_noiseless_linear_trend_is_exact_and_sigma_positive():
    from apexpinn.models.baselines import B2

    history = linear_history()
    model = B2(level_variance=0.0, slope_variance=0.0, observation_variance=0.0)
    prediction = model.forecast(history, 10, Scenario(0.5))
    np.testing.assert_allclose(prediction.mean_s, 90 + 0.12 * np.arange(13, 23), atol=1e-10)
    assert (prediction.sigma_s > 0).all()
    assert prediction.latent_h is None and prediction.latent_w_s is None


def test_b2_known_process_90_percent_coverage_on_2000_draws():
    from apexpinn.models.baselines import B2

    rng = np.random.default_rng(20261103)
    level_var, slope_var, noise_var = 0.005, 0.0004, 0.01
    model = B2(level_variance=level_var, slope_variance=slope_var, observation_variance=noise_var)
    inside = 0
    for _ in range(2000):
        state = np.array([90.0, 0.1])
        observed = []
        for _ in range(25):
            observed.append(state[0] + rng.normal(scale=np.sqrt(noise_var)))
            state = np.array([state[0] + state[1], state[1]]) + rng.normal(
                scale=np.sqrt([level_var, slope_var])
            )
        history = linear_history(20)
        history = StintHistory(
            history.stint_id, 20, history.tau, np.array(observed[:20]), history.q, history.ctx, 20
        )
        prediction = model.forecast(history, 5, Scenario(0.5))
        inside += abs(observed[24] - prediction.mean_s[-1]) <= 1.6448536269514722 * prediction.sigma_s[-1]
    assert 0.87 <= inside / 2000 <= 0.93


def test_b3_parameter_count_matches_m1_network():
    import equinox as eqx
    import jax

    from apexpinn.models.baselines import B3
    from apexpinn.models.pinn import PINN

    model = B3.init(jax.random.key(3), n_steps=2)
    comparator = PINN.init(jax.random.key(4))

    def count(net):
        return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(net, eqx.is_inexact_array)))

    assert abs(count(model.net) / count(comparator.net) - 1) <= 0.1


def baseline_fixture():
    from test_eval import evaluation_fixture

    from apexpinn.data.guards import batch_from_gold, slice_at_origin

    folds, cfg = evaluation_fixture()
    fold = folds["F1"]
    batch = batch_from_gold(fold.gold, fold.normalization)
    stint = fold.gold[fold.gold.session_id == "2024_bahrain"]
    history, scenario, _ = slice_at_origin(stint, 7, 5, circuit_q_mean=fold.normalization.training_q_mean)
    return batch, history, scenario.scenario


@pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3"])
def test_fitted_baselines_positive_scales_and_repeatable_future_safe_outputs(name):
    import jax

    from apexpinn.models.baselines import B0, B1, B2, B3

    batch, history, scenario = baseline_fixture()
    key = jax.random.key(11)
    constructors = {
        "B0": lambda: B0(),
        "B1": lambda: B1(n_estimators=8, min_data_in_leaf=2),
        "B2": lambda: B2(maxiter=100),
        "B3": lambda: B3.init(key, n_steps=20),
    }
    model = constructors[name]().fit(batch, key)
    first = model.forecast(history, 5, scenario)
    again = model.forecast(history, 5, scenario)
    assert np.isfinite(first.mean_s).all() and np.all(first.sigma_s > 0)
    np.testing.assert_array_equal(first.mean_s, again.mean_s)
    np.testing.assert_array_equal(first.sigma_s, again.sigma_s)
    assert first.latent_h is None and first.latent_w_s is None
    assert len(model.state_digest()) == 64


def test_training_batch_keeps_original_laps_and_excludes_test_circuits():
    from test_eval import evaluation_fixture

    from apexpinn.data.guards import batch_from_gold

    folds, _ = evaluation_fixture()
    fold = folds["F1"]
    original = batch_from_gold(fold.gold, fold.normalization)
    changed = fold.gold.copy()
    changed.loc[changed.circuit_id.eq("bahrain"), "proxy_y"] = 1e9
    after = batch_from_gold(changed, fold.normalization)
    np.testing.assert_array_equal(original.y, after.y)
    np.testing.assert_array_equal(original.tau, np.arange(1, 19))
    assert not np.asarray(original.mask)[0, 0]


def test_b2_gap_propagation_uses_original_lap_distance():
    from apexpinn.models.baselines import B2

    full = linear_history(12)
    keep = np.array([0, 1, 4, 7, 11])
    history = StintHistory(full.stint_id, 12, full.tau[keep], full.y[keep], full.q[keep], full.ctx, 12)
    model = B2(level_variance=0, slope_variance=0, observation_variance=0)
    np.testing.assert_allclose(
        model.forecast(history, 3, Scenario(0.5)).mean_s, 90 + 0.12 * np.arange(13, 16), atol=1e-10
    )


def test_inner_protocol_evaluates_declared_validation_group_only():
    import jax
    from test_eval import evaluation_fixture

    from apexpinn.data.gold import build_gold, fit_normalization
    from apexpinn.data.guards import batch_from_gold
    from apexpinn.eval.protocol import EvaluationFold, inner_circuit_split, run_protocol
    from apexpinn.models.baselines import B0

    folds, cfg = evaluation_fixture()
    frame = folds["F1"].gold
    split = inner_circuit_split(cfg.decisions, "F6")
    stats = fit_normalization(frame, cfg.decisions, test_fold="F6", training_circuits=split.training_circuits)
    gold = build_gold(frame, stats, cfg.decisions)
    model = B0().fit(batch_from_gold(gold, stats), jax.random.key(0))
    data = EvaluationFold(
        gold, stats, "fixture", model.state_digest(), inner_validation_fold=split.validation_fold
    )
    records = run_protocol(model, {"F6": data}, cfg, audit=[])
    assert records and all(r.event_id == "2024_bahrain" for r in records)
    assert all(r.conditions["evaluation_stage"] == "inner_validation" for r in records)


@pytest.fixture(scope="module")
def fitted_models_and_gold():
    import jax
    from test_eval import evaluation_fixture

    from apexpinn.data.guards import batch_from_gold
    from apexpinn.models.baselines import B0, B1, B2, B3

    folds, _ = evaluation_fixture()
    fold = folds["F1"]
    batch = batch_from_gold(fold.gold, fold.normalization)
    key = jax.random.key(12)
    models = {
        "B0": B0().fit(batch, key),
        "B1": B1(n_estimators=8, min_data_in_leaf=2).fit(batch, key),
        "B2": B2().fit(batch, key),
        "B3": B3.init(key, n_steps=20).fit(batch, key),
    }
    return models, fold


@settings(max_examples=8, deadline=None)
@given(value=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
@pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3"])
def test_future_perturbations_leave_each_fitted_forecaster_unchanged(fitted_models_and_gold, name, value):
    from apexpinn.data.guards import slice_at_origin

    models, fold = fitted_models_and_gold
    stint = fold.gold[fold.gold.session_id == "2024_bahrain"].copy(deep=True)
    history, design, _ = slice_at_origin(stint, 7, 5, circuit_q_mean=fold.normalization.training_q_mean)
    before = models[name].forecast(history, 5, design.scenario)
    for column in stint:
        if column.startswith(("obs_", "proxy_")):
            stint[column] = stint[column].astype(object)
            stint.loc[stint.stint_lap > 7, column] = value
    history, design, _ = slice_at_origin(stint, 7, 5, circuit_q_mean=fold.normalization.training_q_mean)
    after = models[name].forecast(history, 5, design.scenario)
    for field in ("mean_s", "member_means_s", "sigma_s", "member_sigma_s", "latent_h", "latent_w_s"):
        a, b = getattr(before, field), getattr(after, field)
        if a is None:
            assert b is None
        else:
            assert a.tobytes() == b.tobytes()


def test_trial_budget_counts_failures_and_records_actual_runtime():
    from test_eval import ToyForecaster, evaluation_fixture

    from apexpinn.train.baseline_study import candidate_configs, tune_baseline

    folds, cfg = evaluation_fixture()
    attempts = []

    class Candidate(ToyForecaster):
        def fit(self, train, key):
            return self

        def state_digest(self):
            return "fixture-model"

    def factory(name, params, key, **kwargs):
        attempts.append(dict(params))
        if len(attempts) == 1:
            raise ValueError("deliberate fixture failure")
        return Candidate(error=0.1)

    trials = []
    chosen = tune_baseline(
        folds["F1"].gold,
        cfg.decisions,
        model_name="B1",
        outer_fold="F6",
        seed=4,
        validation_only=True,
        trials=trials,
        factory=factory,
    )
    assert len(attempts) == len(trials) == 20
    assert trials[0].status == "failed" and trials[0].error
    assert sum(t.status == "finished" for t in trials) == 19
    assert all(t.elapsed_s >= 0 and np.isfinite(t.elapsed_s) for t in trials)
    assert chosen in candidate_configs("B1")
    assert len(candidate_configs("B0")) == len(candidate_configs("B2")) == 1

    resumed = []

    def forbidden(*args, **kwargs):
        raise AssertionError("completed trials must never be retried")

    repeated = tune_baseline(
        folds["F1"].gold,
        cfg.decisions,
        model_name="B1",
        outer_fold="F6",
        seed=4,
        validation_only=True,
        trials=resumed,
        factory=forbidden,
        completed=trials,
    )
    assert repeated == chosen
    assert resumed == trials
    assert len(attempts) == 20

    from dataclasses import replace

    with pytest.raises(ValueError, match="resume"):
        tune_baseline(
            folds["F1"].gold,
            cfg.decisions,
            model_name="B1",
            outer_fold="F6",
            seed=4,
            validation_only=True,
            trials=[],
            factory=forbidden,
            completed=[replace(trials[0], seed=999)],
        )


def test_fixture_study_preserves_trial_final_fit_separation_and_report(tmp_path):
    from test_eval import DECISIONS

    from apexpinn.train.baseline_study import fixture_silver, run_study

    output = tmp_path / "run"
    report = tmp_path / "report.md"
    receipt = run_study(
        fixture_silver(DECISIONS),
        DECISIONS,
        out=output,
        report=report,
        models=("B0",),
        folds=("F1", "F2"),
        validation_only=True,
        source_kind="crafted_fixture",
        mlp_steps=2,
    )
    assert not receipt["failures"]
    assert len(receipt["trials"]) == 2 and len(receipt["final_member_fits"]) == 10
    assert receipt["G1"] == "NOT ESTABLISHED"
    assert "No strongest real-data baseline" in report.read_text()
    assert all(
        r["member_sigma_s"] is not None and len(r["member_sigma_s"]) == 5 for r in receipt["forecasts"]
    )
    assert receipt["paired_comparisons"] or receipt["selected_reference_by_inner_validation"] == "B0"
    with pytest.raises(ValueError, match="already contains"):
        run_study(
            fixture_silver(DECISIONS),
            DECISIONS,
            out=output,
            report=report,
            models=("B0",),
            folds=("F1",),
            validation_only=True,
            source_kind="crafted_fixture",
        )


def test_unfrozen_real_study_stops_before_training(tmp_path):
    from dataclasses import replace

    from test_eval import DECISIONS

    from apexpinn.train.baseline_study import fixture_silver, run_study

    with pytest.raises(ValueError, match="G0"):
        run_study(
            fixture_silver(DECISIONS),
            replace(DECISIONS, status="draft"),
            out=tmp_path / "blocked",
            report=tmp_path / "report.md",
            coverage_complete=True,
        )
    assert not (tmp_path / "blocked").exists()


def test_report_cannot_overwrite_study_records_or_unrelated_reports(tmp_path):
    from test_eval import DECISIONS

    from apexpinn.train.baseline_study import fixture_silver, run_study

    data = fixture_silver(DECISIONS)
    with pytest.raises(ValueError, match="Markdown"):
        run_study(
            data,
            DECISIONS,
            out=tmp_path / "out",
            report=tmp_path / "out" / "study.json",
            models=("B0",),
            folds=("F1",),
            validation_only=True,
            source_kind="crafted_fixture",
        )
    report = tmp_path / "unrelated.md"
    report.write_text("# Existing unrelated result\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated"):
        run_study(
            data,
            DECISIONS,
            out=tmp_path / "out",
            report=report,
            models=("B0",),
            folds=("F1",),
            validation_only=True,
            source_kind="crafted_fixture",
        )
    assert report.read_text() == "# Existing unrelated result\n"
