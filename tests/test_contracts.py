import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from apexpinn.contracts import (
    Decisions,
    MetricRecord,
    PredictionOutput,
    SimulationOutput,
    load_decisions,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "decisions_v1.yaml"


@pytest.fixture
def raw_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "decisions.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_default_config_loads():
    decisions = load_decisions(CONFIG_PATH)
    assert isinstance(decisions, Decisions)
    assert decisions.horizons == (1, 3, 5, 10)
    assert decisions.primary_horizon in decisions.horizons
    assert decisions.phi_low <= decisions.phi_mid <= decisions.phi_high
    assert decisions.circuit_to_fold["bahrain"] == "F1"


# Acceptance test 1
def test_rejects_circuit_in_two_folds(tmp_path, raw_config):
    cfg = copy.deepcopy(raw_config)
    cfg["o1_races"]["folds"]["F2"].append("bahrain")
    with pytest.raises(ValueError, match="bahrain"):
        load_decisions(_write(tmp_path, cfg))


# Acceptance test 2
def test_rejects_primary_horizon_not_in_horizons(tmp_path, raw_config):
    cfg = copy.deepcopy(raw_config)
    cfg["o5_protocol"]["primary_horizon"] = 7
    with pytest.raises(ValueError, match="primary_horizon"):
        load_decisions(_write(tmp_path, cfg))


# Acceptance test 3
@pytest.mark.parametrize(
    ("low", "mid", "high"),
    [(0.06, 0.03, 0.09), (0.03, 0.09, 0.06), (0.09, 0.06, 0.03)],
)
def test_rejects_unordered_phi(tmp_path, raw_config, low, mid, high):
    cfg = copy.deepcopy(raw_config)
    cfg["o3_race_lap_trend"].update(phi_low=low, phi_mid=mid, phi_high=high)
    with pytest.raises(ValueError, match="phi"):
        load_decisions(_write(tmp_path, cfg))


def test_accepts_equal_phi(tmp_path, raw_config):
    cfg = copy.deepcopy(raw_config)
    cfg["o3_race_lap_trend"].update(phi_low=0.05, phi_mid=0.05, phi_high=0.05)
    assert load_decisions(_write(tmp_path, cfg)).phi_mid == 0.05


# Acceptance test 4
def test_rejects_holdout_event_in_development_fold(tmp_path, raw_config):
    cfg = copy.deepcopy(raw_config)
    cfg["o1_races"]["holdout"]["events"].append("2024_monza")
    with pytest.raises(ValueError, match="2024_monza"):
        load_decisions(_write(tmp_path, cfg))


# Acceptance test 5
def test_prediction_output_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="member_means_s"):
        PredictionOutput(
            mean_s=np.zeros(5),
            member_means_s=np.zeros((3, 4)),
            sigma_s=np.ones(5),
            latent_h=None,
            latent_w_s=None,
        )
    with pytest.raises(ValueError, match="sigma_s"):
        PredictionOutput(
            mean_s=np.zeros(5),
            member_means_s=np.zeros((3, 5)),
            sigma_s=np.ones(4),
            latent_h=None,
            latent_w_s=None,
        )


def test_prediction_output_accepts_matching_shapes():
    out = PredictionOutput(
        mean_s=np.zeros(5),
        member_means_s=np.zeros((1, 5)),
        sigma_s=np.ones(5),
        latent_h=np.zeros((1, 5)),
        latent_w_s=np.zeros((1, 5)),
    )
    assert out.member_means_s.shape == (1, 5)


def _record(**overrides) -> MetricRecord:
    fields = dict(
        quantity="RMSE in s of y at h = 5, scored laps",
        population="LOCO folds F1-F6, 2023-2024 dry",
        conditions={"phi": 0.06, "seeds": [0, 1, 2], "dtype": "float32"},
        instrument="eval.metrics.rmse + race-level bootstrap, 2000 resamples",
        value=0.42,
        status="Observed",
    )
    fields.update(overrides)
    return MetricRecord(**fields)


# Acceptance test 6
@pytest.mark.parametrize(
    ("field", "new_value"),
    [
        ("quantity", "MAE in s of y at h = 5, scored laps"),
        ("population", "LOCO fold F1 only"),
        ("conditions", {"phi": 0.09, "seeds": [0, 1, 2], "dtype": "float32"}),
        ("instrument", "eval.metrics.rmse, no bootstrap"),
    ],
)
def test_metric_record_comparable_names_differing_field(field, new_value):
    a = _record()
    b = _record(**{field: new_value})
    assert a.comparable(b) is False
    with pytest.raises(ValueError, match=field):
        a.require_comparable(b)


def test_metric_record_comparable_when_only_value_differs():
    a = _record()
    b = _record(value=0.50)
    assert a.comparable(b) is True
    a.require_comparable(b)


def test_metric_record_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        _record(status="Proven")


def test_simulation_output_validates_shapes():
    finish = np.arange(12, dtype=float).reshape(3, 4)
    out = SimulationOutput(
        strategy_ids=("a", "b", "c"),
        finish_times_s=finish,
        mean_s=finish.mean(axis=1),
        std_s=finish.std(axis=1),
        p10_s=np.percentile(finish, 10, axis=1),
        p50_s=np.percentile(finish, 50, axis=1),
        p90_s=np.percentile(finish, 90, axis=1),
        rank=np.array([1, 2, 3]),
    )
    assert out.finish_times_s.shape == (3, 4)
    with pytest.raises(ValueError, match="mean_s"):
        SimulationOutput(
            strategy_ids=("a", "b", "c"),
            finish_times_s=finish,
            mean_s=np.zeros(2),
            std_s=finish.std(axis=1),
            p10_s=np.percentile(finish, 10, axis=1),
            p50_s=np.percentile(finish, 50, axis=1),
            p90_s=np.percentile(finish, 90, axis=1),
            rank=np.array([1, 2, 3]),
        )
