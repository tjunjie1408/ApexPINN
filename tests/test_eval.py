"""T11 independent metric oracles and original-lap evaluation contracts. No race evaluation."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from apexpinn.contracts import FOUR_FIELDS, PredictionOutput, load_decisions

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = load_decisions(ROOT / "configs/decisions_v1.yaml")


def test_gaussian_crps_known_values_and_point_masses():
    from apexpinn.eval.metrics import crps_gaussian_mixture

    assert crps_gaussian_mixture(0.0, [0.0], [1.0]) == pytest.approx(
        (math.sqrt(2) - 1) / math.sqrt(math.pi), abs=1e-14
    )
    assert crps_gaussian_mixture(4.0, [1.0], [0.0]) == 3.0
    assert crps_gaussian_mixture(1.0, [0.0, 2.0], [0.0, 0.0]) == 0.5


def test_mixture_crps_matches_independent_200k_monte_carlo():
    from apexpinn.eval.metrics import crps_gaussian_mixture

    rng = np.random.default_rng(20261103)
    means, scales, weights = np.array([-1.0, 2.0]), np.array([0.5, 1.1]), [0.3, 0.7]
    i, j = rng.choice(2, size=(2, 200_000), p=weights)
    x = rng.normal(means[i], scales[i])
    independent = rng.normal(means[j], scales[j])
    estimate = np.mean(np.abs(x - 0.4)) - 0.5 * np.mean(np.abs(x - independent))
    actual = crps_gaussian_mixture(0.4, means, scales, weights=weights)
    assert actual == pytest.approx(estimate, rel=0.01)


def test_exact_mixture_intervals_and_inclusive_coverage():
    from apexpinn.eval.metrics import central_interval, coverage, interval_width

    lower, upper = central_interval([-5.0, 5.0], [0.0, 0.0], level=0.9)
    assert lower == pytest.approx(-5.0)
    assert upper == pytest.approx(5.0)
    assert coverage([-5.0, 0.0, 5.0], [lower] * 3, [upper] * 3) == 1.0
    assert interval_width([lower], [upper]) == pytest.approx(10.0)
    lower, upper = central_interval([0.0], [1.0], level=0.9)
    assert lower == pytest.approx(-1.6448536269514722)
    assert upper == pytest.approx(1.6448536269514722)


def test_point_metrics_validate_without_silent_nan_dropping():
    from apexpinn.eval.metrics import mae, rmse

    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    assert mae([0.0, 0.0], [3.0, 4.0]) == 3.5
    for target, predicted in [([], []), ([1], [1, 2]), ([np.nan], [1])]:
        with pytest.raises(ValueError):
            rmse(target, predicted)


@pytest.mark.parametrize(
    "means,scales,weights",
    [([], [], None), ([0], [-1], None), ([np.nan], [1], None), ([0, 1], [1, 1], [0.2, 0.2])],
)
def test_invalid_mixtures_raise(means, scales, weights):
    from apexpinn.eval.metrics import crps_gaussian_mixture

    with pytest.raises(ValueError):
        crps_gaussian_mixture(0.0, means, scales, weights=weights)


def test_hierarchical_bootstrap_contains_known_10_percent_and_repeats():
    from apexpinn.eval.bootstrap import paired_bootstrap_rel_rmse

    rng = np.random.default_rng(91)
    baseline = rng.uniform(0.2, 5.0, 120)
    races = np.repeat(["a", "b", "c", "d"], 30)
    stints = np.tile(np.repeat(["1", "2", "3"], 10), 4)
    kwargs = dict(race_ids=races, stint_ids=stints, seed=17)
    result = paired_bootstrap_rel_rmse(np.zeros(120), baseline * 0.9, baseline, **kwargs)
    assert result.estimate == pytest.approx(0.1)
    assert result.ci_low <= 0.1 + 1e-14 and result.ci_high >= 0.1 - 1e-14
    assert result.ci_high - result.ci_low < 1e-12  # unpaired indices would destroy proportionality
    assert result.n_stints == 12 and result.n_races == 4 and result.n_resamples == 2000
    assert result == paired_bootstrap_rel_rmse(np.zeros(120), baseline * 0.9, baseline, **kwargs)


def test_bootstrap_rejects_undefined_denominators_and_single_race():
    from apexpinn.eval.bootstrap import paired_bootstrap_rel_rmse

    kwargs = dict(race_ids=["a", "b"], stint_ids=["1", "1"], n_resamples=100, seed=0)
    with pytest.raises(ValueError, match="zero baseline"):
        paired_bootstrap_rel_rmse([0, 0], [1, 1], [0, 0], **kwargs)
    with pytest.raises(ValueError, match="replicate"):
        paired_bootstrap_rel_rmse([0, 0], [1, 1], [0, 1], **kwargs)
    kwargs["race_ids"] = ["a", "a"]
    with pytest.raises(ValueError, match="two races"):
        paired_bootstrap_rel_rmse([0, 0], [1, 1], [1, 1], **kwargs)


def test_prediction_output_optional_component_scales_and_moment_validation():
    means = np.array([[-1.0, 0.0], [1.0, 2.0]])
    scales = np.array([[0.5, 0.7], [1.0, 0.9]])
    sigma = np.sqrt(np.mean(scales**2, axis=0) + np.var(means, axis=0))
    prediction = PredictionOutput(np.mean(means, axis=0), means, sigma, None, None, member_sigma_s=scales)
    np.testing.assert_array_equal(prediction.member_sigma_s, scales)
    with pytest.raises(ValueError, match="member_sigma_s"):
        PredictionOutput(np.mean(means, axis=0), means, sigma * 0.5, None, None, member_sigma_s=scales)
    assert PredictionOutput([0], [[0]], [1], None, None).member_sigma_s is None


class ToyForecaster:
    """Fixture forecaster only. Does not fit or claim to be any B0-B3/M1/M2 candidate."""

    def __init__(self, error=0.0, *, members=1, explicit_scales=False):
        self.error, self.members, self.explicit_scales = error, members, explicit_scales
        self.calls = []

    def fit(self, *args, **kwargs):
        raise AssertionError("evaluation must not fit")

    def forecast(self, history, horizon, scenario):
        self.calls.append((history, horizon, scenario))
        # The crafted fixture has a linear corrected target. Only fit that line to history.
        slope, intercept = np.polyfit(history.tau, history.y, 1)
        mean = intercept + slope * np.arange(history.t0 + 1, history.t0 + horizon + 1) + self.error
        members = np.tile(mean, (self.members, 1))
        kwargs = {"member_sigma_s": np.full_like(members, 0.2)} if self.explicit_scales else {}
        return PredictionOutput(mean, members, np.full(horizon, 0.2), None, None, **kwargs)


def evaluation_fixture(*, model_name="toy"):
    from test_gold import silver_events

    from apexpinn.data.gold import build_gold, fit_normalization
    from apexpinn.eval.protocol import EvaluationFold, ProtocolConfig

    silver = silver_events()
    silver = silver[silver.dataset_split == "development"].copy()
    expanded = []
    for _, event in silver.groupby("session_id", sort=True):
        frame = event.iloc[[0] * 18].reset_index(drop=True).copy()
        laps = np.arange(1, 19)
        frame["lap_number"], frame["race_lap"], frame["stint_lap"] = laps, laps, laps
        frame["race_length"] = 18
        frame["is_scored_lap"] = laps > 1
        frame["exclusion_reason"] = np.where(laps == 1, "race_lap_1", "")
        frame["obs_tyre_life"] = laps + 2.0
        frame["obs_lap_start_s"], frame["obs_lap_end_s"] = 100 + (laps - 1) * 90.0, 100 + laps * 90.0
        frame["proxy_y"] = 90 - 0.06 * (18 - laps)
        frame["proxy_energy_index"] = float(event.proxy_energy_index.iloc[0]) + laps - 1
        expanded.append(frame)
    import pandas as pd

    silver = pd.concat(expanded, ignore_index=True)
    folds = {}
    for fold in ("F1", "F2"):
        stats = fit_normalization(silver, DECISIONS, test_fold=fold)
        gold = build_gold(silver, stats, DECISIONS)
        folds[fold] = EvaluationFold(gold, stats, f"fixture-gold-{fold}", f"fixture-model-{fold}")
    return folds, ProtocolConfig(DECISIONS, model_name, 0.06, validation_only=True)


def test_shared_anchor_is_model_class_independent():
    from apexpinn.data.guards import slice_at_origin
    from apexpinn.eval.protocol import anchor_offset
    from apexpinn.models.stints import anchor_offset as shared_array_anchor

    folds, _ = evaluation_fixture()
    frame = folds["F1"].gold
    stint = frame[frame.session_id == "2024_bahrain"]
    history, _, _ = slice_at_origin(stint, 4, 3, circuit_q_mean=0.5)

    class ShapeA:
        def shape(self, tau):
            return tau * 0.1

    class ShapeB:
        def shape(self, tau):
            return tau * 0.1

    assert anchor_offset(history, ShapeA().shape) == anchor_offset(history, ShapeB().shape)
    assert anchor_offset(history, ShapeA().shape) == shared_array_anchor(history.y, history.tau * 0.1)


def test_protocol_endpoint_only_original_laps_audit_and_four_fields():
    from apexpinn.eval.protocol import run_protocol, score_forecasts

    folds, cfg = evaluation_fixture()
    for data in folds.values():
        # Exclude lap 5; it must not shift origin=4,h=3 from lap 7 to lap 8.
        mask = data.gold.stint_lap.eq(5)
        data.gold.loc[mask, ["is_scored_lap", "exclusion_reason"]] = [False, "traffic"]
    models = {fold: ToyForecaster() for fold in folds}
    audit = []
    records = run_protocol(models, folds, cfg, audit=audit)
    selected = [r for r in records if r.t0 == 4 and r.horizon == 3]
    assert len(selected) == 2
    assert all(r.target_stint_lap == 7 and abs(r.y_pred - r.y_true) < 1e-10 for r in selected)
    assert any(r.reason == "history_not_consecutive_clean" for r in audit)
    assert any(r.reason == "target_not_scored" for r in audit)
    assert all(set(FOUR_FIELDS) <= set(asdict(r)) for r in records + audit)
    metrics = score_forecasts(records)
    assert all(set(FOUR_FIELDS) <= set(asdict(r)) for r in metrics)
    assert all(r.status == "Observed" for r in metrics)
    assert all(r.conditions["validation_only"] for r in metrics)
    json.dumps([asdict(r) for r in records + audit + metrics], allow_nan=False)
    assert all(
        horizon in cfg.decisions.horizons for model in models.values() for _, horizon, _ in model.calls
    )
    assert {r.horizon for r in records} == {1, 3, 5, 10}


def test_probabilities_are_unknown_for_unidentified_ensemble_variance():
    from apexpinn.eval.protocol import run_protocol, score_forecasts

    folds, cfg = evaluation_fixture()
    records = run_protocol(ToyForecaster(members=2), {"F1": folds["F1"]}, cfg, audit=[])
    metrics = score_forecasts(records)
    assert all(r.status == "Observed" for r in metrics if r.quantity["metric"] in ("rmse", "mae"))
    unknown = [r for r in metrics if r.quantity["metric"] == "crps"]
    assert unknown and all(r.value is None and r.status == "Unknown" for r in unknown)
    records = run_protocol(ToyForecaster(members=2, explicit_scales=True), {"F1": folds["F1"]}, cfg, audit=[])
    assert all(r.status == "Observed" for r in score_forecasts(records))


def test_protocol_draft_gate_and_model_mapping_fail_closed():
    from apexpinn.eval.protocol import run_protocol

    folds, cfg = evaluation_fixture()
    with pytest.raises(ValueError, match="draft"):
        run_protocol(ToyForecaster(), {"F1": folds["F1"]}, replace(cfg, validation_only=False), audit=[])
    with pytest.raises(ValueError, match="fold-specific"):
        run_protocol(ToyForecaster(), folds, cfg, audit=[])
    bad = replace(folds["F1"], gold=folds["F1"].gold.copy())
    bad.gold["normalization_test_fold"] = "F2"
    with pytest.raises(ValueError, match="normalization_test_fold"):
        run_protocol(ToyForecaster(), {"F1": bad}, cfg, audit=[])


def test_pairing_rejects_mismatched_population_instead_of_intersection():
    from apexpinn.eval.protocol import paired_comparison, run_protocol

    folds, cfg = evaluation_fixture()
    model = run_protocol({f: ToyForecaster(0.9) for f in folds}, folds, cfg, audit=[])
    baseline = run_protocol(
        {f: ToyForecaster(1.0) for f in folds}, folds, replace(cfg, model_name="baseline"), audit=[]
    )
    result = paired_comparison(model, baseline, horizon=3, phi=0.06, seed=11)
    assert result.estimate == pytest.approx(0.1)
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(model[1:], baseline, horizon=1, phi=0.06, seed=11)


def test_pairing_checks_history_conditions_and_normalization_provenance():
    from apexpinn.eval.protocol import paired_comparison, run_protocol

    folds, cfg = evaluation_fixture()
    model = run_protocol({f: ToyForecaster(0.9) for f in folds}, folds, cfg, audit=[])
    baseline = run_protocol(
        {f: ToyForecaster(1) for f in folds}, folds, replace(cfg, model_name="base"), audit=[]
    )
    altered = [replace(r, conditions={**r.conditions, "k_min": 4}) for r in baseline]
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(model, altered, horizon=3, phi=0.06, seed=0)
    altered = [replace(r, instrument={**r.instrument, "normalization_hash": "different"}) for r in baseline]
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(model, altered, horizon=3, phi=0.06, seed=0)


def test_declared_training_events_cannot_hide_a_test_circuit():
    from apexpinn.eval.protocol import run_protocol

    folds, cfg = evaluation_fixture()
    bad = replace(
        folds["F1"], normalization=replace(folds["F1"].normalization, training_events=("2024_bahrain",))
    )
    with pytest.raises(ValueError, match="training provenance"):
        run_protocol(ToyForecaster(), {"F1": bad}, cfg, audit=[])


def test_input_quality_is_audited_but_model_failures_propagate():
    from apexpinn.eval.protocol import run_protocol

    folds, cfg = evaluation_fixture()
    fold = folds["F1"]
    selected = fold.gold.session_id.eq("2024_bahrain") & fold.gold.stint_lap.eq(2)
    fold.gold.loc[selected, "proxy_q_heat"] = np.nan
    audit = []
    assert run_protocol(ToyForecaster(), {"F1": fold}, cfg, audit=audit) == []
    assert any(r.reason.startswith("input_quality:") for r in audit)
    folds, cfg = evaluation_fixture()

    class FailingModel(ToyForecaster):
        def forecast(self, history, horizon, scenario):
            raise RuntimeError("model failure")

    with pytest.raises(RuntimeError, match="model failure"):
        run_protocol(FailingModel(), {"F1": folds["F1"]}, cfg, audit=[])


def test_protocol_predictions_ignore_future_observations_at_fixed_origin():
    from apexpinn.eval.protocol import run_protocol

    folds, cfg = evaluation_fixture()
    before = run_protocol(ToyForecaster(), {"F1": folds["F1"]}, cfg, audit=[])
    gold = folds["F1"].gold
    mask = gold.session_id.eq("2024_bahrain") & gold.stint_lap.gt(4)
    gold.loc[mask, ["proxy_q_heat", "proxy_energy_index", "obs_track_temp_c", "obs_tyre_life"]] = 0.99
    gold.loc[mask, "proxy_y"] += 50.0
    after = run_protocol(ToyForecaster(), {"F1": folds["F1"]}, cfg, audit=[])
    left = {r.key: r for r in before if r.t0 == 4}
    right = {r.key: r for r in after if r.t0 == 4}
    assert left.keys() == right.keys() and left
    for key in left:
        assert left[key].y_pred == right[key].y_pred
        assert left[key].member_means_s == right[key].member_means_s
        assert left[key].y_true != right[key].y_true


def test_origin_summary_keeps_candidate_denominator():
    from apexpinn.eval.protocol import run_protocol, summarize_origins

    folds, cfg = evaluation_fixture()
    audit = []
    run_protocol(ToyForecaster(), {"F1": folds["F1"]}, cfg, audit=audit)
    summary = summarize_origins(audit)
    horizon_three = {r.quantity["metric"]: r for r in summary if r.conditions["horizon_actual_laps"] == 3}
    assert horizon_three["candidate_origins"].value == 13
    assert horizon_three["history_rejected_origins"].value == 1
    assert horizon_three["history_rejection_share"].value == pytest.approx(1 / 13)
    assert all(set(FOUR_FIELDS) <= set(asdict(r)) for r in summary)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_origins(audit + audit)


def test_temporal_holdout_is_never_evaluated_by_development_protocol():
    import pandas as pd

    from apexpinn.eval.protocol import run_protocol

    folds, cfg = evaluation_fixture()
    frame = folds["F1"].gold
    holdout = frame[frame.session_id == "2024_bahrain"].copy()
    holdout["session_id"], holdout["stint_id"], holdout["dataset_split"] = (
        "2025_bahrain",
        "2025_bahrain:1:1",
        "holdout",
    )
    contaminated = replace(folds["F1"], gold=pd.concat([frame, holdout], ignore_index=True))
    with pytest.raises(ValueError, match="holdout"):
        run_protocol(ToyForecaster(), {"F1": contaminated}, cfg, audit=[])


def test_inner_split_is_fixed_grouped_and_excludes_both_test_populations():
    from apexpinn.eval.protocol import inner_circuit_split

    order = tuple(DECISIONS.folds)
    for i, outer in enumerate(order):
        split = inner_circuit_split(DECISIONS, outer)
        assert split.validation_fold == order[(i + 1) % len(order)]
        train, validation, test = (
            set(split.training_circuits),
            set(split.validation_circuits),
            set(DECISIONS.folds[outer]),
        )
        assert not (train & validation or train & test or validation & test)
        assert train | validation | test == set(DECISIONS.circuit_to_fold)
        assert split == inner_circuit_split(DECISIONS, outer)
        assert set(FOUR_FIELDS) <= set(asdict(split))


def test_inner_normalization_ignores_inner_validation_circuits():
    from test_gold import silver_events

    from apexpinn.data.gold import fit_normalization
    from apexpinn.eval.protocol import inner_circuit_split

    split = inner_circuit_split(DECISIONS, "F6")
    assert split.validation_fold == "F1"
    silver = silver_events()
    stats = fit_normalization(silver, DECISIONS, test_fold="F6", training_circuits=split.training_circuits)
    silver.loc[silver.circuit_id.eq("bahrain"), "proxy_energy_index"] = 1e9
    assert (
        fit_normalization(silver, DECISIONS, test_fold="F6", training_circuits=split.training_circuits)
        == stats
    )
    assert stats.training_events == ("2024_monaco",)


def test_metric_pooling_rejects_inconsistent_protocol_conditions():
    from apexpinn.eval.protocol import run_protocol, score_forecasts

    folds, cfg = evaluation_fixture()
    records = run_protocol({f: ToyForecaster() for f in folds}, folds, cfg, audit=[])
    altered = [replace(r, conditions={**r.conditions, "k_min": 4}) if r.fold == "F2" else r for r in records]
    with pytest.raises(ValueError, match="different protocol conditions"):
        score_forecasts(altered)
