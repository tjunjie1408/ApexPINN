"""Gold normalization contracts and independent analytic proxy checks (PLAN T10)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apexpinn.contracts import load_decisions

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = load_decisions(ROOT / "configs/decisions_v1.yaml")


def silver_events():
    frames = []
    for event, offset in (("2024_bahrain", 100), ("2024_monaco", 10), ("2025_bahrain", 1000)):
        season, circuit = event.split("_", 1)
        lap = np.arange(1, 9)
        frames.append(
            pd.DataFrame(
                {
                    "session_id": event,
                    "season": int(season),
                    "circuit_id": circuit,
                    "driver_id": "1",
                    "stint_id": event + ":1:1",
                    "lap_number": lap,
                    "race_lap": lap,
                    "stint_lap": lap,
                    "race_length": 8,
                    "split_fold": DECISIONS.circuit_to_fold[circuit],
                    "dataset_split": "holdout" if season == "2025" else "development",
                    "event_is_dry": True,
                    "is_scored_lap": lap > 1,
                    "exclusion_reason": np.where(lap == 1, "race_lap_1", ""),
                    "compound_c": "C3",
                    "source_version": "crafted-test-fixture",
                    "phi": 0.06,
                    "obs_lap_time_s": 90.0,
                    "obs_lap_start_s": 100 + (lap - 1) * 90.0,
                    "obs_lap_end_s": 100 + lap * 90.0,
                    "obs_pit_in_s": np.nan,
                    "obs_pit_out_s": np.nan,
                    "obs_track_status": "1",
                    "obs_is_accurate": True,
                    "obs_deleted": False,
                    "obs_compound": "SOFT",
                    "obs_tyre_life": lap + 2.0,
                    "obs_fresh_tyre": False,
                    "obs_track_temp_c": 30.0,
                    "obs_speed_st_kmh": 300.0,
                    "proxy_y": 90 - 0.06 * (8 - lap),
                    "proxy_energy_index": offset + lap - 1.0,
                    "proxy_brake_fraction": 0.25,
                    "proxy_brake_fraction_weighted": 0.2,
                    "proxy_telemetry_missing": False,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_normalization_uses_training_circuits_only():
    from apexpinn.data.gold import fit_normalization

    silver = silver_events()
    stats = fit_normalization(silver, DECISIONS, test_fold="F1")
    assert (stats.lower, stats.upper, stats.n_train) == (11.0, 17.0, 7)
    assert stats.training_events == ("2024_monaco",)
    assert stats.training_circuits == ("monaco",)
    changed = silver.copy()
    heldout = changed.circuit_id.eq("bahrain")
    changed.loc[heldout, "proxy_energy_index"] = np.nan
    changed.loc[heldout, "proxy_y"] = -9999
    assert fit_normalization(changed, DECISIONS, test_fold="F1") == stats


def test_inner_split_and_empty_fit_are_explicit():
    from apexpinn.data.gold import fit_normalization

    silver = silver_events()
    with pytest.raises(ValueError, match="test circuits"):
        fit_normalization(silver, DECISIONS, test_fold="F1", training_circuits=("bahrain",))
    with pytest.raises(ValueError, match="no usable training"):
        fit_normalization(silver[silver.circuit_id == "bahrain"], DECISIONS, test_fold="F1")


def test_gold_preserves_rows_flags_and_clips_using_frozen_stats():
    from apexpinn.data.gold import build_gold, fit_normalization

    silver = silver_events()
    before = silver.copy(deep=True)
    stats = fit_normalization(silver, DECISIONS, test_fold="F1")
    gold = build_gold(silver, stats, DECISIONS)
    pd.testing.assert_frame_equal(before, silver)
    pd.testing.assert_frame_equal(gold[silver.columns], silver)
    assert gold.loc[gold.circuit_id == "bahrain", "proxy_q_heat"].eq(1).all()
    train = gold[(gold.session_id == "2024_monaco") & gold.is_scored_lap]
    np.testing.assert_allclose(train.proxy_q_heat, [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])
    assert gold.proxy_speed_trap.eq(300).all()
    assert gold.loc[gold.circuit_id == "bahrain", "proxy_q_clipped"].all()
    assert stats.context_q_mean("bahrain") == pytest.approx(0.5)


def test_constant_range_and_nonfinite_features_are_visible():
    from apexpinn.data.gold import build_gold, fit_normalization

    silver = silver_events()
    silver.loc[silver.session_id == "2024_monaco", "proxy_energy_index"] = 5.0
    stats = fit_normalization(silver, DECISIONS, test_fold="F1")
    assert stats.constant_range
    silver.loc[2, "proxy_energy_index"] = np.inf
    gold = build_gold(silver, stats, DECISIONS)
    assert np.isnan(gold.loc[2, "proxy_q_heat"])
    assert gold.loc[2, "feature_exclusion_reason"] == "invalid_proxy"
    assert gold.loc[1, "proxy_q_heat"] == 0
    assert gold.loc[2, "is_scored_lap"]  # Silver population flags are preserved


def test_event_missing_telemetry_boundary_and_temporal_holdout_isolation():
    from apexpinn.data.gold import event_eligibility, fit_normalization

    frame = silver_events().groupby("session_id", sort=False).head(5).copy()
    train_indices = frame.index[frame.session_id.eq("2024_monaco")]
    frame.loc[train_indices[0], "proxy_telemetry_missing"] = True
    eligibility = event_eligibility(frame).set_index("session_id")
    assert eligibility.loc["2024_monaco", "eligible"]
    assert eligibility.loc["2024_monaco", "telemetry_missing_share"] == 0.2
    stats = fit_normalization(frame, DECISIONS, test_fold="F1")
    temporal = frame[frame.session_id == "2024_monaco"].copy()
    temporal["session_id"] = "2025_monaco"
    temporal["stint_id"] = "2025_monaco:1:1"
    temporal["season"] = 2025
    temporal["dataset_split"] = "holdout"
    temporal["proxy_energy_index"] = 1e9
    assert fit_normalization(pd.concat([frame, temporal]), DECISIONS, test_fold="F1") == stats
    frame.loc[train_indices[1], "proxy_telemetry_missing"] = True
    with pytest.raises(ValueError, match="no usable training"):
        fit_normalization(frame, DECISIONS, test_fold="F1")


def test_gold_model_inputs_use_global_training_fallback_for_test_circuit():
    from apexpinn.data.gold import build_gold, fit_normalization
    from apexpinn.data.guards import slice_at_origin

    silver = silver_events()
    stats = fit_normalization(silver, DECISIONS, test_fold="F1")
    gold = build_gold(silver, stats, DECISIONS)
    stint = gold[gold.session_id == "2024_bahrain"]
    history, future, labels = slice_at_origin(stint, 4, 3, circuit_q_mean=stats.context_q_mean("bahrain"))
    assert history.ctx.circuit_q_mean == pytest.approx(0.5)
    assert history.q.tolist() == [1, 1, 1]
    assert future.tau.tolist() == [5, 6, 7]
    assert np.isfinite(labels).all()
    assert stint.ctx_q_mean_source.eq("global_training_fallback").all()


def test_circle_energy_and_boolean_braking_reach_gold_without_redefinition():
    from apexpinn.data import probe
    from apexpinn.data.gold import build_gold, fit_normalization

    t = np.arange(0, 60, 0.1)
    speed, radius = 50.0, 100.0
    car = pd.DataFrame({"t_s": t, "speed_kmh": speed * 3.6, "brake": np.arange(len(t)) % 4 == 0})
    pos = pd.DataFrame(
        {"t_s": t, "x_m": radius * np.cos(t * speed / radius), "y_m": radius * np.sin(t * speed / radius)}
    )
    aggregates = probe.per_lap_telemetry(pd.DataFrame({"lap_start_s": [0.0], "lap_end_s": [60.0]}), car, pos)
    frame = silver_events()
    frame["proxy_energy_index"] = aggregates.energy_index.iloc[0]
    frame["proxy_brake_fraction"] = aggregates.brake_fraction.iloc[0]
    gold = build_gold(frame, fit_normalization(frame, DECISIONS, test_fold="F1"), DECISIONS)
    np.testing.assert_allclose(gold.proxy_energy_index, speed**2 / radius, rtol=0.02)
    assert gold.proxy_brake_fraction.eq(0.25).all()


def write_bronze_events(root):
    from apexpinn.data import manifest as mf
    from apexpinn.data.silver import LAP_COLUMNS

    for event, group in silver_events().groupby("session_id"):
        if event.startswith("2025"):
            continue
        laps = group.rename(columns={v: k for k, v in LAP_COLUMNS.items()}).copy()
        laps["DriverNumber"], laps["LapNumber"], laps["Stint"] = "1", group.lap_number, 1.0
        laps["SpeedST"] = 300.0
        telemetry = pd.DataFrame(
            {
                "DriverNumber": "1",
                "LapNumber": group.lap_number,
                "energy_index": group.proxy_energy_index,
                "brake_fraction": 0.25,
                "brake_fraction_weighted": 0.2,
                "telemetry_missing": False,
            }
        )
        weather = pd.DataFrame({"Time_s": [0.0], "TrackTemp": [30.0]})
        directory = root / event
        frames = {
            "laps.parquet": laps[["DriverNumber", "LapNumber", "Stint", "SpeedST", *LAP_COLUMNS]],
            "weather.parquet": weather,
            "lap_telemetry.parquet": telemetry,
        }
        files = {name: mf.write_parquet(frame, directory / name) for name, frame in frames.items()}
        mf.write_manifest(
            directory, {"schema": mf.MANIFEST_SCHEMA, "event_id": event, "fixture": True, "files": files}
        )


def test_full_bronze_rebuild_is_deterministic_and_report_is_generated(tmp_path):
    from apexpinn.data import manifest as mf
    from apexpinn.data.gold import build_dataset

    bronze = tmp_path / "bronze"
    write_bronze_events(bronze)
    kwargs = dict(
        bronze_root=bronze,
        out_root=tmp_path / "gold",
        config=ROOT / "configs/decisions_v1.yaml",
        allocations=ROOT / "configs/compound_allocation.yaml",
        event_ids=("2024_bahrain", "2024_monaco"),
        folds=("F1", "F2"),
        phi_setting="mid",
        report_path=tmp_path / "data_report.md",
    )
    first = build_dataset(**kwargs)
    assert not first["fold_failures"]
    assert not first["coverage"]["configured_dev_events_complete"]
    hashes = {p: mf.sha256_file(p) for p in (tmp_path / "gold").rglob("*") if p.is_file()}
    report = (tmp_path / "data_report.md").read_bytes()
    assert build_dataset(**kwargs)["dataset_sha256"] == first["dataset_sha256"]
    assert hashes == {p: mf.sha256_file(p) for p in hashes}
    assert (tmp_path / "data_report.md").read_bytes() == report
    text = report.decode()
    assert "PARTIAL" in text and "NOT ESTABLISHED" in text
    assert "exclusion" in text.lower() and "variance" in text.lower()
    assert first["folds"]["F1"]["normalization"]["training_events"] == ["2024_monaco"]


def test_pipeline_fails_closed_for_bad_bronze_and_empty_fold(tmp_path):
    from apexpinn.data.gold import build_dataset

    bronze = tmp_path / "bronze"
    write_bronze_events(bronze)
    kwargs = dict(
        bronze_root=bronze,
        out_root=tmp_path / "gold",
        config=ROOT / "configs/decisions_v1.yaml",
        allocations=ROOT / "configs/compound_allocation.yaml",
        event_ids=("2024_bahrain",),
        folds=("F1",),
        phi_setting="mid",
        report_path=tmp_path / "data_report.md",
    )
    result = build_dataset(**kwargs)
    assert "no usable training" in result["fold_failures"]["F1"]
    assert not result["folds"]
    (bronze / "2024_bahrain" / "laps.parquet").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Bronze integrity"):
        build_dataset(**kwargs)


def test_inner_training_subset_and_phi_mismatch():
    from apexpinn.data.gold import build_gold, fit_normalization

    silver = silver_events()
    full = fit_normalization(silver, DECISIONS, test_fold="F4")
    inner = fit_normalization(silver, DECISIONS, test_fold="F4", training_circuits=("monaco",))
    assert full.upper == 107
    assert inner.upper == 17
    assert inner.training_events == ("2024_monaco",)
    silver["phi"] = 0.03
    with pytest.raises(ValueError, match="phi"):
        build_gold(silver, full, DECISIONS)


def test_future_test_circuit_values_cannot_change_gold_origin_inputs():
    from apexpinn.data.gold import build_gold, fit_normalization
    from apexpinn.data.guards import slice_at_origin

    frame = silver_events()
    stats = fit_normalization(frame, DECISIONS, test_fold="F1")
    gold = build_gold(frame, stats, DECISIONS)
    first = slice_at_origin(
        gold[gold.session_id == "2024_bahrain"], 4, 3, circuit_q_mean=stats.context_q_mean("bahrain")
    )
    mask = frame.session_id.eq("2024_bahrain") & frame.stint_lap.gt(4)
    frame.loc[mask, ["proxy_energy_index", "obs_track_temp_c", "obs_tyre_life", "proxy_y"]] = np.nan
    assert fit_normalization(frame, DECISIONS, test_fold="F1") == stats
    gold = build_gold(frame, stats, DECISIONS)
    second = slice_at_origin(
        gold[gold.session_id == "2024_bahrain"], 4, 3, circuit_q_mean=stats.context_q_mean("bahrain")
    )
    assert first[0].ctx == second[0].ctx
    for column in ("tau", "y", "q"):
        assert getattr(first[0], column).tobytes() == getattr(second[0], column).tobytes()
    for column in ("tau", "tyre_age", "race_lap", "q"):
        assert getattr(first[1], column).tobytes() == getattr(second[1], column).tobytes()


def test_cli_partial_failure_and_temporal_holdout_protection(tmp_path, capsys):
    from apexpinn.data.gold import main

    bronze = tmp_path / "bronze"
    write_bronze_events(bronze)
    args = [
        "--bronze-root",
        str(bronze),
        "--out-root",
        str(tmp_path / "gold"),
        "--event",
        "2024_bahrain",
        "--fold",
        "F1",
        "--report",
        str(tmp_path / "report.md"),
        "--config",
        str(ROOT / "configs/decisions_v1.yaml"),
        "--allocations",
        str(ROOT / "configs/compound_allocation.yaml"),
    ]
    assert main(args) == 1
    assert "no usable training" in capsys.readouterr().out
    args[args.index("2024_bahrain")] = "2025_bahrain"
    with pytest.raises(ValueError, match="holdout processing disabled"):
        main(args)


def test_latent_columns_cannot_be_stored_as_gold():
    from apexpinn.data.gold import build_gold, fit_normalization

    silver = silver_events()
    stats = fit_normalization(silver, DECISIONS, test_fold="F1")
    silver["latent_w"] = 0.0
    with pytest.raises(ValueError, match="latent_"):
        build_gold(silver, stats, DECISIONS)
