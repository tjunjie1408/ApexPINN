"""T9 population, original-lap horizon and feature-leakage contracts. No network."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apexpinn.contracts import load_decisions
from apexpinn.data import manifest as mf
from apexpinn.data.guards import OriginRejected, audit_origins, slice_at_origin
from apexpinn.data.silver import build_silver, main

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = load_decisions(ROOT / "configs/decisions_v1.yaml")
ALLOCATION = {"HARD": "C1", "MEDIUM": "C2", "SOFT": "C3"}


def bronze_fixture(n=18):
    laps = pd.DataFrame(
        {
            "DriverNumber": ["1"] * n,
            "LapNumber": np.arange(1, n + 1, dtype=float),
            "Stint": np.ones(n),
            "LapTime_s": np.full(n, 90.0),
            "LapStartTime_s": np.arange(n) * 90.0 + 100.0,
            "Time_s": np.arange(n) * 90.0 + 190.0,
            "PitInTime_s": np.full(n, np.nan),
            "PitOutTime_s": np.full(n, np.nan),
            "TrackStatus": ["1"] * n,
            "IsAccurate": [True] * n,
            "Deleted": [False] * n,
            "Compound": ["SOFT"] * n,
            "TyreLife": np.arange(1, n + 1, dtype=float) + 2,
            "FreshTyre": [False] * n,
        }
    )
    telemetry = laps[["DriverNumber", "LapNumber"]].copy()
    telemetry["energy_index"] = np.arange(n, dtype=float) + 10
    telemetry["brake_fraction"] = 0.2
    telemetry["brake_fraction_weighted"] = 0.1
    telemetry["telemetry_missing"] = False
    weather = pd.DataFrame({"Time_s": [0.0, 200.0, 9999.0], "TrackTemp": [30.0, 31.0, 99.0]})
    return laps, weather, telemetry


def silver_fixture(laps=None, *, event_id="2024_bahrain", phi=0.06):
    raw, weather, telemetry = bronze_fixture()
    return build_silver(
        raw if laps is None else laps,
        weather,
        telemetry,
        event_id=event_id,
        decisions=DECISIONS,
        allocation=ALLOCATION,
        phi=phi,
        source_version="test-fixture-sha256",
    )


def gold_fixture():
    frame = silver_fixture()
    # Explicit normalized fixture. T9 does not fit normalization statistics.
    frame["proxy_q_heat"] = np.linspace(0.1, 0.9, len(frame))
    return frame


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("LapTime_s", np.nan, "no_lap_time"),
        ("LapTime_s", np.inf, "invalid_lap_time"),
        ("LapTime_s", -1.0, "invalid_lap_time"),
        ("PitInTime_s", 440.0, "in_lap"),
        ("PitOutTime_s", 370.0, "out_lap"),
        ("TrackStatus", "14", "not_green"),
        ("TrackStatus", "16", "not_green"),
        ("TrackStatus", "17", "not_green"),
        ("TrackStatus", "2", "not_green"),
        ("TrackStatus", None, "not_green"),
        ("IsAccurate", False, "inaccurate"),
        ("IsAccurate", None, "inaccurate"),
        ("Deleted", True, "deleted"),
        ("Compound", "INTERMEDIATE", "wet_compound"),
        ("Compound", "WET", "wet_compound"),
        ("Compound", "UNKNOWN", "unknown_compound"),
    ],
)
def test_each_population_filter(column, value, reason):
    laps, _, _ = bronze_fixture()
    if value is None:
        laps[column] = laps[column].astype(object)
    laps.loc[3, column] = value
    frame = silver_fixture(laps)
    assert frame.loc[3, "exclusion_reason"] == reason
    assert not frame.loc[3, "is_scored_lap"]
    assert frame.loc[0, "exclusion_reason"] == "race_lap_1"


@pytest.mark.parametrize("phi", [0.03, 0.06, 0.09])
def test_target_keys_folds_and_causal_weather(phi):
    frame = silver_fixture(phi=phi)
    np.testing.assert_allclose(frame["proxy_y"], 90.0 - phi * (18 - np.arange(1, 19)))
    assert frame["compound_c"].eq("C3").all()
    assert frame["obs_track_temp_c"].iloc[:2].tolist() == [30.0, 30.0]
    assert frame["obs_track_temp_c"].iloc[2:].eq(31.0).all()
    assert frame["split_fold"].nunique() == 1
    assert silver_fixture(event_id="2023_bahrain")["split_fold"].iloc[0] == frame["split_fold"].iloc[0]
    assert silver_fixture(event_id="2025_bahrain")["split_fold"].iloc[0] == frame["split_fold"].iloc[0]
    assert silver_fixture(event_id="2025_bahrain")["dataset_split"].eq("holdout").all()
    assert not any(c.startswith("latent_") for c in frame)
    assert "proxy_q_heat" not in frame


def test_traffic_exact_threshold_and_missing_gap():
    laps, weather, telemetry = bronze_fixture()
    other = laps.copy()
    other["DriverNumber"] = "2"
    other["Time_s"] += 1.0
    other.loc[3, "Time_s"] -= 0.001
    other_telemetry = telemetry.copy()
    other_telemetry["DriverNumber"] = "2"
    frame = build_silver(
        pd.concat([laps, other]),
        weather,
        pd.concat([telemetry, other_telemetry]),
        event_id="2024_bahrain",
        decisions=DECISIONS,
        allocation=ALLOCATION,
        phi=0.06,
        source_version="fixture",
    )
    leader = frame[frame.driver_id == "1"]
    follower = frame[frame.driver_id == "2"].set_index("lap_number")
    assert leader["obs_gap_ahead_s"].isna().all()
    assert follower.loc[3, "exclusion_reason"] == ""
    assert follower.loc[4, "exclusion_reason"] == "traffic"


def test_duplicates_and_missing_schema_raise():
    laps, weather, telemetry = bronze_fixture()
    for raw, telem in [
        (pd.concat([laps, laps.iloc[[2]]]), telemetry),
        (laps, pd.concat([telemetry, telemetry.iloc[[2]]])),
    ]:
        with pytest.raises(ValueError, match="duplicate"):
            build_silver(
                raw,
                weather,
                telem,
                event_id="2024_bahrain",
                decisions=DECISIONS,
                allocation=ALLOCATION,
                phi=0.06,
                source_version="fixture",
            )
    with pytest.raises(ValueError, match="IsAccurate"):
        silver_fixture(laps.drop(columns="IsAccurate"))


def test_wet_race_cannot_supply_origins_and_unknown_compound_is_not_dry():
    laps, _, _ = bronze_fixture()
    laps.loc[10, "Compound"] = "WET"
    frame = silver_fixture(laps)
    assert not frame["event_is_dry"].any()
    assert not frame["is_scored_lap"].any()
    laps.loc[10, "Compound"] = None
    assert not silver_fixture(laps)["event_is_dry"].any()


def test_original_laps_and_stint_boundaries_are_not_renumbered():
    laps, weather, telemetry = bronze_fixture()
    laps.loc[laps.LapNumber >= 10, "Stint"] = 2.0
    laps = laps[laps.LapNumber != 12]
    telemetry = telemetry[telemetry.LapNumber != 12]
    frame = build_silver(
        laps,
        weather,
        telemetry,
        event_id="2024_bahrain",
        decisions=DECISIONS,
        allocation=ALLOCATION,
        phi=0.06,
        source_version="fixture",
    )
    second = frame[frame.lap_number >= 10]
    assert second["stint_lap"].tolist() == [1, 2, 4, 5, 6, 7, 8, 9]
    assert frame["stint_id"].nunique() == 2
    with pytest.raises(ValueError, match="one stint"):
        slice_at_origin(frame, 5, 1, circuit_q_mean=0.5)


def test_history_keeps_older_clean_laps_and_future_is_deterministic():
    frame = gold_fixture()
    history, future, labels = slice_at_origin(frame, 7, 5, circuit_q_mean=0.4)
    assert history.tau.tolist() == [2, 3, 4, 5, 6, 7]
    assert history.race_lap_t0 == 7
    assert history.ctx.age_at_start == 3
    assert history.ctx.track_temp_start == 30
    assert history.ctx.circuit_q_mean == 0.4
    assert future.tau.tolist() == [8, 9, 10, 11, 12]
    assert future.race_lap.tolist() == [8, 9, 10, 11, 12]
    assert future.tyre_age.tolist() == [10, 11, 12, 13, 14]
    np.testing.assert_array_equal(labels, frame.loc[7:11, "proxy_y"])
    assert future.scenario.q_bar == pytest.approx(frame.loc[4:6, "proxy_q_heat"].mean())
    assert not set(vars(future)) & {"y", "labels", "obs_track_temp_c"}


def test_endpoint_horizon_not_hth_scored_row_and_intermediate_labels_masked():
    frame = gold_fixture()
    frame.loc[7, ["is_scored_lap", "exclusion_reason"]] = [False, "traffic"]
    _, future, labels = slice_at_origin(frame, 7, 5, circuit_q_mean=0.5)
    assert future.tau[-1] == 12
    assert np.isnan(labels[0])
    assert labels[-1] == frame.loc[11, "proxy_y"]
    frame.loc[11, ["is_scored_lap", "exclusion_reason"]] = [False, "traffic"]
    with pytest.raises(OriginRejected, match="target_not_scored"):
        slice_at_origin(frame, 7, 5, circuit_q_mean=0.5)


@pytest.mark.parametrize("hole", [4, 5, 6])
def test_three_consecutive_original_clean_laps_required(hole):
    frame = gold_fixture()
    frame.loc[hole, ["is_scored_lap", "exclusion_reason"]] = [False, "traffic"]
    with pytest.raises(OriginRejected, match="history_not_consecutive_clean"):
        slice_at_origin(frame, 7, 1, circuit_q_mean=0.5)
    frame = gold_fixture().drop(index=hole)
    with pytest.raises(OriginRejected, match="history_not_consecutive_clean"):
        slice_at_origin(frame, 7, 1, circuit_q_mean=0.5)


def test_missing_future_and_invalid_inputs_raise():
    frame = gold_fixture()
    with pytest.raises(OriginRejected, match="target_missing"):
        slice_at_origin(frame, 16, 5, circuit_q_mean=0.5)
    with pytest.raises(OriginRejected, match="history_not_consecutive_clean"):
        slice_at_origin(frame, 3, 1, circuit_q_mean=0.5)
    frame.loc[6, "proxy_q_heat"] = 2.0
    with pytest.raises(ValueError, match="proxy_q_heat"):
        slice_at_origin(frame, 7, 1, circuit_q_mean=0.5)
    with pytest.raises(ValueError, match="horizon"):
        slice_at_origin(gold_fixture(), 7, 0, circuit_q_mean=0.5)


def test_unseen_c6_rejected_without_remapping():
    frame = gold_fixture()
    frame["compound_c"] = "C6"
    with pytest.raises(ValueError, match="C6"):
        slice_at_origin(frame, 7, 1, circuit_q_mean=0.5)


def assert_same_inputs(left, right):
    lh, lf, _ = left
    rh, rf, _ = right
    assert (lh.stint_id, lh.t0, lh.race_lap_t0, lh.ctx) == (rh.stint_id, rh.t0, rh.race_lap_t0, rh.ctx)
    for field in ("tau", "y", "q"):
        assert getattr(lh, field).tobytes() == getattr(rh, field).tobytes()
    for field in ("tau", "tyre_age", "race_lap", "q"):
        assert getattr(lf, field).tobytes() == getattr(rf, field).tobytes()


@settings(max_examples=40, deadline=None)
@given(
    t0=st.integers(4, 12),
    horizon=st.sampled_from([1, 3, 5]),
    value=st.floats(allow_nan=True, allow_infinity=True, width=64),
)
def test_future_observations_and_proxies_cannot_change_inputs(t0, horizon, value):
    frame = gold_fixture()
    original = slice_at_origin(frame, t0, horizon, circuit_q_mean=0.5)
    changed = frame.copy(deep=True)
    for column in changed:
        if column.startswith(("obs_", "proxy_")):
            changed[column] = changed[column].astype(object)
            changed.loc[changed.stint_lap > t0, column] = value
    assert_same_inputs(original, slice_at_origin(changed, t0, horizon, circuit_q_mean=0.5))
    changed.loc[changed.stint_lap <= t0, "proxy_y"] += 1.0
    altered = slice_at_origin(changed, t0, horizon, circuit_q_mean=0.5)
    assert original[0].y.tobytes() != altered[0].y.tobytes()


def test_origin_audit_preserves_rejection_denominators():
    frame = gold_fixture()
    frame.loc[5, ["is_scored_lap", "exclusion_reason"]] = [False, "traffic"]
    audit = audit_origins(frame, horizons=(1, 5), k_min=3)
    assert set(audit.horizon) == {1, 5}
    selected = audit[(audit.t0 == 7) & (audit.horizon == 5)].iloc[0]
    assert selected.reason == "history_not_consecutive_clean"
    assert not selected.eligible
    assert len(audit[audit.horizon == 5]) == 11  # possible original origins 3..13


def test_silver_cli_integrity_and_deterministic_rebuild(tmp_path, capsys):
    laps, weather, telemetry = bronze_fixture()
    bronze = tmp_path / "bronze" / "2024_bahrain"
    frames = dict(
        zip(
            ("laps.parquet", "weather.parquet", "lap_telemetry.parquet"),
            (laps, weather, telemetry),
            strict=True,
        )
    )
    files = {name: mf.write_parquet(frame, bronze / name) for name, frame in frames.items()}
    mf.write_manifest(bronze, {"schema": mf.MANIFEST_SCHEMA, "event_id": "2024_bahrain", "files": files})
    args = [
        "--event",
        "2024_bahrain",
        "--bronze-root",
        str(bronze.parent),
        "--out-root",
        str(tmp_path / "silver"),
        "--config",
        str(ROOT / "configs/decisions_v1.yaml"),
        "--allocations",
        str(ROOT / "configs/compound_allocation.yaml"),
    ]
    assert main(args) == 0
    output = tmp_path / "silver" / "2024_bahrain" / "phi_mid"
    hashes = {name: mf.sha256_file(output / name) for name in ("laps.parquet", "manifest.json")}
    assert main(args) == 0
    assert hashes == {name: mf.sha256_file(output / name) for name in hashes}
    assert mf.verify_event(bronze, required_files=tuple(files)) == []
    assert len(capsys.readouterr().out.splitlines()) == 2
    (bronze / "weather.parquet").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Bronze integrity"):
        main(args)


@pytest.mark.parametrize("field", ["stint_lap", "race_lap"])
def test_guard_rejects_duplicate_or_inconsistent_counters(field):
    frame = gold_fixture()
    frame.loc[4, field] = frame.loc[3, field]
    with pytest.raises(ValueError, match="duplicate"):
        slice_at_origin(frame, 7, 1, circuit_q_mean=0.5)
    frame = gold_fixture()
    frame.loc[5, "race_lap"] += 100
    with pytest.raises(ValueError, match="offset"):
        slice_at_origin(frame, 7, 1, circuit_q_mean=0.5)


def test_missing_q_requires_gold_and_history_is_not_an_input_view():
    with pytest.raises(ValueError, match="proxy_q_heat"):
        slice_at_origin(silver_fixture(), 7, 1, circuit_q_mean=0.5)
    frame = gold_fixture()
    history, future, labels = slice_at_origin(frame, 7, 1, circuit_q_mean=0.5)
    snapshot = history.y.tobytes(), history.q.tobytes(), labels.tobytes(), future.q.tobytes()
    frame.loc[:, "proxy_y"] = 0.0
    frame.loc[:, "proxy_q_heat"] = 0.0
    assert snapshot == (history.y.tobytes(), history.q.tobytes(), labels.tobytes(), future.q.tobytes())


def test_nullable_boolean_quality_flags_are_handled_explicitly():
    laps, _, _ = bronze_fixture()
    laps["IsAccurate"] = laps.IsAccurate.astype("boolean")
    laps["Deleted"] = laps.Deleted.astype("boolean")
    laps.loc[4, "IsAccurate"] = pd.NA
    laps.loc[5, "Deleted"] = pd.NA
    frame = silver_fixture(laps)
    assert frame.loc[4, "exclusion_reason"] == "inaccurate"
    assert frame.loc[5, "exclusion_reason"] == ""  # missing Deleted does not trigger the filter


def test_stint_ids_cannot_reappear_after_a_different_tire_set():
    laps, _, _ = bronze_fixture()
    laps.loc[5:9, "Stint"] = 2.0
    with pytest.raises(ValueError, match="Stint.*monotonic"):
        silver_fixture(laps)


def test_recorded_proxy_inputs_retained_and_input_order_does_not_matter():
    laps, weather, telemetry = bronze_fixture()
    laps["SpeedST"] = 301.0
    telemetry["car_coverage"] = 0.95
    before = laps.copy(deep=True)
    kwargs = dict(
        event_id="2024_bahrain",
        decisions=DECISIONS,
        allocation=ALLOCATION,
        phi=0.06,
        source_version="fixture",
    )
    frame = build_silver(laps, weather, telemetry, **kwargs)
    assert frame["obs_speed_st_kmh"].eq(301).all()
    assert frame["proxy_car_coverage"].eq(0.95).all()
    shuffled = build_silver(
        laps.sample(frac=1, random_state=2), weather, telemetry.sample(frac=1, random_state=3), **kwargs
    )
    pd.testing.assert_frame_equal(frame, shuffled)
    pd.testing.assert_frame_equal(before, laps)
