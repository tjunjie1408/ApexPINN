"""Offline tests for the Bronze extract and manifest (PLAN T8). FastF1 sessions are faked, no network."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apexpinn.contracts import load_decisions
from apexpinn.data import extract
from apexpinn.data import manifest as mf

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "decisions_v1.yaml"
LAP_S = 60.0
RATE_HZ = 4.0
N_LAPS = 4
V_MS = 50.0
RADIUS_M = 100.0
DRIVERS = (("AAA", "1", 100.0), ("BBB", "2", 103.0))  # (code, number, session time of lap 1 start)
T0 = datetime(2024, 3, 2, 18, 0, tzinfo=UTC)


def _lap_window(start0: float, lap: int) -> tuple[float, float]:
    return start0 + (lap - 1) * LAP_S, start0 + lap * LAP_S


class _FakeSession:
    """Small deterministic race: 2 drivers, 4 laps, 4 Hz streams that extend past the race on both ends.

    Driver 1 brakes only during lap 2, driver 2 never brakes.
    """

    def __init__(self, brake_lap: int = 2):
        rows, self.car_data, self.pos_data = [], {}, {}
        for code, number, start0 in DRIVERS:
            for lap in range(1, N_LAPS + 1):
                ls, le = _lap_window(start0, lap)
                rows.append(
                    {
                        "Time": pd.Timedelta(le, unit="s"),
                        "Driver": code,
                        "DriverNumber": number,
                        "LapTime": pd.Timedelta(LAP_S, unit="s"),
                        "LapNumber": float(lap),
                        "Stint": 1.0,
                        "PitOutTime": pd.NaT,
                        "PitInTime": pd.NaT,
                        "IsPersonalBest": False,
                        "Compound": "MEDIUM",
                        "TyreLife": float(lap),
                        "FreshTyre": True,
                        "LapStartTime": pd.Timedelta(ls, unit="s"),
                        "LapStartDate": pd.Timestamp("2024-03-02 15:00:00") + pd.Timedelta(ls, unit="s"),
                        "TrackStatus": "1",
                        "Deleted": False,
                        "DeletedReason": "",
                        "IsAccurate": True,
                    }
                )
            t = np.arange(start0 - 10.0, start0 + N_LAPS * LAP_S + 10.0, 1.0 / RATE_HZ)
            angle = V_MS / RADIUS_M * t
            brake_lo, brake_hi = _lap_window(start0, brake_lap)
            brake = (t >= brake_lo) & (t < brake_hi) if number == "1" else np.zeros(t.size, dtype=bool)
            self.car_data[number] = pd.DataFrame(
                {
                    "SessionTime": pd.to_timedelta(t, unit="s"),
                    "Speed": np.full_like(t, V_MS * 3.6),
                    "Throttle": np.full_like(t, 80.0),
                    "Brake": brake,
                }
            )
            self.pos_data[number] = pd.DataFrame(
                {
                    "SessionTime": pd.to_timedelta(t, unit="s"),
                    "X": RADIUS_M * np.cos(angle) * 10.0,  # FastF1 positions are in 1/10 m
                    "Y": RADIUS_M * np.sin(angle) * 10.0,
                }
            )
        laps = pd.DataFrame(rows)
        for col in ("PitOutTime", "PitInTime"):  # no pit stops in this race
            laps[col] = pd.Series([pd.NaT] * len(laps), index=laps.index, dtype="timedelta64[ns]")
        self.laps = laps
        self.weather_data = pd.DataFrame(
            {
                "Time": pd.to_timedelta([0.0, 60.0, 120.0], unit="s"),
                "AirTemp": [25.0, 25.1, 25.3],
                "TrackTemp": [40.0, 40.5, 41.0],
                "Rainfall": [False, False, False],
            }
        )
        self.event = pd.Series(
            {"EventName": "Bahrain Grand Prix", "Location": "Sakhir", "EventDate": "2024-03-02"}
        )

    def load(self, **kwargs):
        return None


class _Loader:
    """Fake ``fastf1.get_session`` that records its calls."""

    def __init__(self, **session_kwargs):
        self.calls: list[tuple[int, str, str]] = []
        self.session_kwargs = session_kwargs

    def __call__(self, season: int, grand_prix: str, kind: str) -> _FakeSession:
        self.calls.append((season, grand_prix, kind))
        return _FakeSession(**self.session_kwargs)


def _file_state(event_dir: Path) -> dict[str, tuple[int, bytes]]:
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(event_dir.iterdir())}


# Acceptance test 1: the manifest SHA-256 is stable across two writes of the same frame.


def test_parquet_sha256_stable_across_two_writes(tmp_path):
    frame = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.5],
            "name": ["a", None, "c"],
            "flag": [True, False, True],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", None]),
        }
    )
    first = mf.write_parquet(frame, tmp_path / "a.parquet")
    second = mf.write_parquet(frame.copy(), tmp_path / "b.parquet")
    again = mf.write_parquet(frame, tmp_path / "a.parquet")
    assert first == second == again
    assert first["sha256"] == mf.sha256_file(tmp_path / "b.parquet")
    assert first["rows"] == 3 and first["columns"] == ["x", "name", "flag", "when"]
    assert not list(tmp_path.glob("*.tmp"))


def test_forced_reextract_keeps_file_hashes_and_updates_timestamp(tmp_path):
    loader = _Loader()
    first = extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=T0)
    later = datetime(2024, 3, 3, 9, 30, tzinfo=UTC)
    second = extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=later, force=True)
    assert first.status == second.status == "written"
    assert first.manifest["files"] == second.manifest["files"]
    assert set(first.manifest["files"]) == set(extract.BRONZE_FILES)
    assert first.manifest["downloaded_utc"] == "2024-03-02T18:00:00Z"
    assert second.manifest["downloaded_utc"] == "2024-03-03T09:30:00Z"
    assert mf.read_manifest(second.event_dir) == second.manifest
    assert mf.verify_event(second.event_dir, extract.BRONZE_FILES) == []


def test_manifest_records_versions_and_event(tmp_path):
    result = extract.extract_event("2024_bahrain", tmp_path, get_session=_Loader(), now=T0)
    m = result.manifest
    assert m["schema"] == mf.MANIFEST_SCHEMA
    assert (m["event_id"], m["season"], m["circuit"], m["session"]) == ("2024_bahrain", 2024, "bahrain", "R")
    assert m["grand_prix_query"] == "Bahrain Grand Prix" and m["event_name"] == "Bahrain Grand Prix"
    assert m["versions"]["fastf1"] and m["versions"]["pyarrow"] and m["versions"]["pandas"]
    assert m["drivers_without_car_data"] == [] and m["drivers_without_pos_data"] == []
    assert m["files"][extract.LAPS_FILE]["rows"] == len(DRIVERS) * N_LAPS


# Acceptance test 2: re-running the extract on an existing event is a no-op unless --force.


def test_rerun_is_noop_unless_force(tmp_path):
    loader = _Loader()
    first = extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=T0)
    before = _file_state(first.event_dir)

    again = extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=T0)
    assert again.status == "skipped"
    assert len(loader.calls) == 1  # FastF1 was not even asked for the session
    assert _file_state(first.event_dir) == before
    assert again.manifest == first.manifest

    forced = extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=T0, force=True)
    assert forced.status == "written"
    assert len(loader.calls) == 2


def test_corrupted_extract_is_redone_without_force(tmp_path):
    loader = _Loader()
    first = extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=T0)
    (first.event_dir / extract.WEATHER_FILE).write_bytes(b"not parquet")
    assert mf.verify_event(first.event_dir, extract.BRONZE_FILES) == [
        f"{extract.WEATHER_FILE} SHA-256 does not match the manifest"
    ]
    redo = extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=T0)
    assert redo.status == "written" and len(loader.calls) == 2
    assert redo.manifest["files"] == first.manifest["files"]


def test_missing_manifest_counts_as_incomplete(tmp_path):
    loader = _Loader()
    first = extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=T0)
    (first.event_dir / mf.MANIFEST_NAME).unlink()
    assert extract.extract_event("2024_bahrain", tmp_path, get_session=loader, now=T0).status == "written"


def test_cli_rerun_is_noop_unless_force(tmp_path, monkeypatch):
    import fastf1

    loader = _Loader()
    monkeypatch.setattr(fastf1, "get_session", loader)
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda *a, **k: None)
    args = [
        "--event", "2024_bahrain",
        "--event", "2023_spa:Belgium",
        "--out-root", str(tmp_path / "bronze"),
        "--cache-dir", str(tmp_path / "cache"),
    ]  # fmt: skip
    assert extract.main(args) == 0
    assert loader.calls == [(2024, "Bahrain Grand Prix", "R"), (2023, "Belgium", "R")]
    before = {e: _file_state(tmp_path / "bronze" / e) for e in ("2024_bahrain", "2023_spa")}

    assert extract.main(args) == 0
    assert len(loader.calls) == 2
    assert {e: _file_state(tmp_path / "bronze" / e) for e in before} == before

    assert extract.main([*args, "--force"]) == 0
    assert len(loader.calls) == 4


def test_cli_reports_failed_event_and_keeps_going(tmp_path, monkeypatch):
    import fastf1

    def loader(season, grand_prix, kind):
        if season == 2023:
            raise RuntimeError("no such session")
        return _FakeSession()

    monkeypatch.setattr(fastf1, "get_session", loader)
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda *a, **k: None)
    args = ["--event", "2023_spa", "--event", "2024_bahrain", "--out-root", str(tmp_path)]
    assert extract.main([*args, "--cache-dir", str(tmp_path / "cache")]) == 1
    assert mf.verify_event(tmp_path / "2024_bahrain", extract.BRONZE_FILES) == []
    assert not (tmp_path / "2023_spa").exists()


# Acceptance test 3: per-lap telemetry aggregates are computed from that lap's samples only.


def _telemetry(tmp_path: Path, session: _FakeSession) -> pd.DataFrame:
    result = extract.extract_event("2024_bahrain", tmp_path, get_session=lambda *a: session, force=True)
    return pd.read_parquet(result.event_dir / extract.LAP_TELEMETRY_FILE)


def test_lap_telemetry_uses_only_that_laps_samples(tmp_path):
    tele = _telemetry(tmp_path, _FakeSession(brake_lap=2))
    assert len(tele) == len(DRIVERS) * N_LAPS
    driver1 = tele[tele["DriverNumber"] == "1"].sort_values("LapNumber")
    driver2 = tele[tele["DriverNumber"] == "2"].sort_values("LapNumber")
    assert driver1["brake_fraction"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert driver1["brake_fraction_weighted"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert driver2["brake_fraction"].tolist() == [0.0] * N_LAPS  # driver 1's streams never leak in
    assert (tele["n_car"] == LAP_S * RATE_HZ).all()  # streams extend past the race; only in-lap samples
    assert not tele["telemetry_missing"].any()
    np.testing.assert_allclose(tele["speed_mean_kmh"], V_MS * 3.6)
    np.testing.assert_allclose(tele["energy_index"], V_MS**2 / RADIUS_M, rtol=0.02)


def test_lap_telemetry_unchanged_by_later_samples(tmp_path):
    base = _telemetry(tmp_path / "base", _FakeSession())
    perturbed_session = _FakeSession()
    _, _, start0 = DRIVERS[0]
    lap3_start, _ = _lap_window(start0, 3)
    rng = np.random.default_rng(0)
    streams = (
        (perturbed_session.car_data["1"], ("Speed", "Brake")),
        (perturbed_session.pos_data["1"], ("X", "Y")),
    )
    for frame, cols in streams:
        later = frame["SessionTime"].dt.total_seconds() >= lap3_start  # values only, same timestamps
        for col in cols:
            if col == "Brake":
                frame.loc[later, col] = True
            else:
                frame.loc[later, col] = frame.loc[later, col] * rng.uniform(0.5, 1.5, int(later.sum()))
    perturbed = _telemetry(tmp_path / "perturbed", perturbed_session)
    early = (perturbed["DriverNumber"] == "1") & (perturbed["LapNumber"] <= 2)
    pd.testing.assert_frame_equal(perturbed[early], base[early], check_exact=True)
    assert perturbed.loc[~early & (perturbed["DriverNumber"] == "1"), "brake_fraction"].gt(0.9).all()


def test_drivers_without_streams_are_flagged(tmp_path):
    session = _FakeSession()
    del session.car_data["2"]
    del session.pos_data["2"]
    result = extract.extract_event("2024_bahrain", tmp_path, get_session=lambda *a: session, now=T0)
    assert result.manifest["drivers_without_car_data"] == ["2"]
    tele = pd.read_parquet(result.event_dir / extract.LAP_TELEMETRY_FILE)
    assert tele.loc[tele["DriverNumber"] == "2", "telemetry_missing"].all()
    assert not tele.loc[tele["DriverNumber"] == "1", "telemetry_missing"].any()


# Bronze layout and event selection.


def test_laps_and_weather_store_timedeltas_as_seconds(tmp_path):
    session = _FakeSession()
    result = extract.extract_event("2024_bahrain", tmp_path, get_session=lambda *a: session, now=T0)
    laps = pd.read_parquet(result.event_dir / extract.LAPS_FILE)
    assert "LapTime" not in laps.columns
    np.testing.assert_allclose(laps["LapTime_s"], LAP_S)
    np.testing.assert_allclose(laps["Time_s"], session.laps["Time"].dt.total_seconds())
    assert laps["PitInTime_s"].isna().all()
    assert laps["Compound"].tolist() == session.laps["Compound"].tolist()
    weather = pd.read_parquet(result.event_dir / extract.WEATHER_FILE)
    assert weather.columns.tolist() == ["Time_s", "AirTemp", "TrackTemp", "Rainfall"]
    np.testing.assert_allclose(weather["Time_s"], [0.0, 60.0, 120.0])


def test_missing_weather_writes_empty_frame(tmp_path):
    session = _FakeSession()
    session.weather_data = None
    result = extract.extract_event("2024_bahrain", tmp_path, get_session=lambda *a: session, now=T0)
    weather = pd.read_parquet(result.event_dir / extract.WEATHER_FILE)
    assert weather.empty and weather.columns.tolist() == list(extract.WEATHER_EMPTY_COLUMNS)


def test_no_laps_raises_and_writes_no_manifest(tmp_path):
    session = _FakeSession()
    session.laps = session.laps.iloc[0:0]
    with pytest.raises(ValueError, match="no laps"):
        extract.extract_event("2024_bahrain", tmp_path, get_session=lambda *a: session, now=T0)
    assert mf.read_manifest(tmp_path / "2024_bahrain") is None


def test_parse_event():
    assert extract.parse_event("2023_mexico_city") == ("2023_mexico_city", "Mexico City Grand Prix")
    assert extract.parse_event("2024_imola:Imola") == ("2024_imola", "Imola")
    with pytest.raises(ValueError, match="season"):
        extract.parse_event("spa_2023")
    with pytest.raises(ValueError, match="no FastF1 event name"):
        extract.parse_event("2023_nurburgring")


def test_every_config_event_has_a_fastf1_name():
    decisions = load_decisions(CONFIG)
    events = extract.config_events(decisions, "all")
    assert len(events) == len(decisions.dev_events) + len(decisions.holdout_events)
    assert extract.config_events(decisions, "dev") == list(decisions.dev_events)
    assert extract.config_events(decisions, "holdout") == list(decisions.holdout_events)
    for event in events:
        extract.parse_event(event)  # raises if a circuit has no FastF1 event name
    with pytest.raises(ValueError):
        extract.config_events(decisions, "test")
