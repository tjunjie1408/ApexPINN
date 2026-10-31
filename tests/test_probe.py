"""Offline tests for the FastF1 data probe (PLAN T6). No network: FastF1 sessions are faked."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apexpinn.data import probe

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_fastf1.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("probe_fastf1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_column_report_flags_absent_and_missing():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
    report = probe.column_report(frame, ["a", "b"])
    assert report["a"] == {"present": True, "nan_fraction": 0.5}
    assert report["b"] == {"present": False, "nan_fraction": None}
    assert probe.column_report(None, ["a"])["a"]["present"] is False


@pytest.mark.parametrize(
    ("status", "green"), [("1", True), ("11", True), ("12", False), ("4", False), (None, False)]
)
def test_is_green(status, green):
    assert probe.is_green(status) is green


def test_gap_to_car_ahead_per_lap_number():
    laps = pd.DataFrame({"LapNumber": [1, 1, 1, 2, 2], "lap_end_s": [100.0, 100.4, 103.0, 190.0, 190.8]})
    gaps = probe.gap_to_car_ahead(laps)
    assert np.isnan(gaps[0]) and np.isnan(gaps[3])
    np.testing.assert_allclose(gaps[[1, 2, 4]].to_numpy(), [0.4, 2.6, 0.8])


def test_exclusion_reason_first_matching_filter():
    laps = pd.DataFrame(
        {
            "LapNumber": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "lap_time_s": [95, 90, 90, 90, 90, 90, 90, 90, 90, np.nan],
            "pit_in_s": [np.nan, 50.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "pit_out_s": [np.nan, np.nan, 60.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "TrackStatus": ["1", "1", "1", "14", "1", "1", "1", "1", "1", "1"],
            "IsAccurate": [True, True, True, True, False, True, True, True, True, True],
            "Deleted": [False, False, False, False, False, True, False, False, False, False],
            "Compound": ["SOFT"] * 6 + ["INTERMEDIATE", "SOFT", "SOFT", "SOFT"],
            "gap_ahead_s": [np.nan, 5, 5, 5, 5, 5, 5, 0.4, 3.0, 5],
        }
    )
    reasons = probe.exclusion_reason(laps, traffic_gap_threshold_s=1.0).tolist()
    assert reasons == [
        "race_lap_1",
        "in_lap",
        "out_lap",
        "not_green",
        "inaccurate",
        "deleted",
        "wet_compound",
        "traffic",
        "",
        "no_lap_time",
    ]


def test_pit_lane_times_pairs_in_lap_with_next_out_lap():
    laps = pd.DataFrame(
        {
            "Driver": ["A", "A", "A", "B", "B"],
            "LapNumber": [10, 11, 12, 10, 11],
            "pit_in_s": [1000.0, np.nan, np.nan, 2000.0, np.nan],
            "pit_out_s": [np.nan, 1022.5, np.nan, np.nan, 2400.0],  # B: 400 s garage stop is dropped
        }
    )
    np.testing.assert_allclose(probe.pit_lane_times(laps), [22.5])


def test_lateral_accel_on_circle_matches_v2_over_r():
    v, r = 60.0, 120.0
    t = np.arange(0.0, 30.0, 0.25)
    angle = v / r * t
    a_lat = probe.lateral_accel(t, r * np.cos(angle), r * np.sin(angle), np.full_like(t, v))
    np.testing.assert_allclose(a_lat[2:-2], v**2 / r, rtol=1e-6)  # central differences are exact here
    np.testing.assert_allclose(np.nanmean(a_lat), v**2 / r, rtol=0.02)  # one-sided ends included


def test_lateral_accel_masks_stationary_samples():
    t = np.arange(10.0)
    a_lat = probe.lateral_accel(t, np.zeros(10), np.zeros(10), np.zeros(10))
    assert np.isnan(a_lat).all()


def test_per_lap_telemetry_uses_only_samples_inside_the_lap():
    laps = pd.DataFrame({"lap_start_s": [0.0, 10.0], "lap_end_s": [10.0, 20.0]})
    t = np.arange(-5.0, 25.0, 0.25)  # streams extend past both laps, as for a real driver
    brake = (t >= 10.0) & (t < 20.0)  # braking only in lap 2
    car = pd.DataFrame({"t_s": t, "speed_kmh": np.full_like(t, 180.0), "brake": brake})
    r, v = 100.0, 50.0
    pos = pd.DataFrame({"t_s": t, "x_m": r * np.cos(v / r * t), "y_m": r * np.sin(v / r * t)})
    out = probe.per_lap_telemetry(laps, car, pos)
    assert out["brake_fraction"].tolist() == [0.0, 1.0]
    assert out["brake_fraction_weighted"].tolist() == [0.0, 1.0]
    assert not out["telemetry_missing"].any()
    np.testing.assert_allclose(out["energy_index"], v**2 / r, rtol=0.02)
    np.testing.assert_allclose(out["speed_mean_kmh"], 180.0)


def test_per_lap_telemetry_ignores_samples_after_the_lap():
    # PROJECT section 5.1: no proxy for lap k uses telemetry from laps > k. Perturb every sample from the
    # first sample of lap 2 on (t = 10.0, exactly the end of lap 1) and lap 1 must be bitwise unchanged.
    laps = pd.DataFrame({"lap_start_s": [0.0, 10.0, 20.0], "lap_end_s": [10.0, 20.0, 30.0]})
    t = np.arange(-5.0, 35.0, 0.25)
    r, v = 100.0, 50.0
    angle = v / r * t
    v_kmh = 180.0 + 20.0 * np.sin(0.3 * t)  # varying speed, so |a_lon| is not zero
    car = pd.DataFrame({"t_s": t, "speed_kmh": v_kmh, "brake": np.sin(t) > 0.5})
    pos = pd.DataFrame({"t_s": t, "x_m": r * np.cos(angle), "y_m": r * np.sin(angle)})
    base = probe.per_lap_telemetry(laps, car, pos)

    later = t >= laps.loc[1, "lap_start_s"]
    n_later = int(later.sum())
    rng = np.random.default_rng(0)
    car_p, pos_p = car.copy(), pos.copy()
    car_p.loc[later, "speed_kmh"] = car_p.loc[later, "speed_kmh"] * rng.uniform(0.5, 1.5, n_later)
    car_p.loc[later, "brake"] = True
    for col in ("x_m", "y_m"):
        pos_p.loc[later, col] = pos_p.loc[later, col] * rng.uniform(0.5, 1.5, n_later)
    perturbed = probe.per_lap_telemetry(laps, car_p, pos_p)

    pd.testing.assert_frame_equal(perturbed.loc[[0]], base.loc[[0]], check_exact=True)
    assert np.isfinite(base.loc[0, "energy_index"])
    assert perturbed.loc[1, "brake_fraction"] == 1.0  # the perturbation does reach the later laps
    assert perturbed.loc[1, "energy_index"] != base.loc[1, "energy_index"]


def test_per_lap_telemetry_flags_missing_streams():
    laps = pd.DataFrame({"lap_start_s": [0.0, 10.0], "lap_end_s": [10.0, 20.0]})
    t = np.arange(0.0, 10.0, 0.25)  # no samples in lap 2
    car = pd.DataFrame({"t_s": t, "speed_kmh": np.full_like(t, 180.0), "brake": False})
    out = probe.per_lap_telemetry(laps, car, None)
    assert out["telemetry_missing"].tolist() == [True, True]
    assert out.loc[1, "n_car"] == 0


def _proxy_frame() -> pd.DataFrame:
    rows = []
    for event, base_q in (("2023_a", 10.0), ("2024_b", 20.0)):
        for stint in range(3):
            for lap in range(1, 9):
                q = base_q + 0.1 * lap
                rows.append(
                    {
                        "event_id": event,
                        "stint_key": f"{event}/{stint}",
                        "race_lap": lap + 10 * stint,
                        "total_laps": 40,
                        "lap_time_s": 90.0 + 0.5 * q,
                        "good": q,
                        "flat": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_compare_proxies_statistics():
    result = probe.compare_proxies(_proxy_frame(), phi=0.0, proxies=["good", "flat", "absent"])
    good = result["good"]
    assert good["n_laps"] == 48 and good["n_stints"] == 6 and good["n_events"] == 2
    np.testing.assert_allclose(good["across_event_std"], np.std([10.45, 20.45], ddof=1))
    np.testing.assert_allclose(good["corr_y_within_stint"], 1.0)
    np.testing.assert_allclose(good["l2_ratio"], good["within_stint_std"] / good["across_event_std"])
    assert np.isnan(result["flat"]["corr_y_within_stint"])
    assert result["absent"] == {"available": False}
    assert probe.suggest_proxy(result) == "good"


def _warmup_frame(phi: float, total: float = 50.0) -> pd.DataFrame:
    shapes = {
        "dip": (2, [92.0, 91.0, 90.5, 90.8, 91.2]),  # min at stint lap 4, gain 1.5 s
        "dip_at_first_scored": (2, [90.0, 90.2, 90.4, 90.6]),  # min at stint lap 2, gain 0
        "min_at_lap_1": (1, [89.0, 90.0, 90.1, 90.2, 90.3]),
        "still_improving": (2, [92.0, 91.5, 91.0, 90.5, 90.0]),  # last lap is the minimum
        "too_short": (2, [92.0, 90.0]),
        "min_too_late": (2, [95.0 - 0.5 * k for k in range(9)] + [91.5, 92.0]),  # min at stint lap 10
    }
    rows = []
    for stint, (first, ys) in shapes.items():
        for k, y in enumerate(ys):
            lap = first + k
            race_lap = 10.0 + lap
            rows.append(
                {
                    "stint_key": stint,
                    "stint_lap": float(lap),
                    "TyreLife": float(lap + 3),
                    "race_lap": race_lap,
                    "total_laps": total,
                    "lap_time_s": y + phi * (total - race_lap),
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # order must not matter


def test_warmup_visibility_counts_early_dips_on_corrected_time():
    phi = 0.1
    out = probe.warmup_visibility(_warmup_frame(phi), phi)
    assert out["n_stints"] == 5 and out["n_dip"] == 2
    assert out["dip_share"] == pytest.approx(0.4)
    gain = out["dip_gain_s"]
    assert gain["n"] == 2
    assert (gain["median"], gain["q25"], gain["q75"]) == pytest.approx((0.75, 0.375, 1.125))
    assert out["first_scored_stint_lap"] == {"median": 2.0, "counts": {"1": 1, "2": 4}}
    assert out["first_scored_tyre_life_median"] == pytest.approx(5.0)


def test_warmup_visibility_empty_frame():
    cols = ["stint_key", "stint_lap", "race_lap", "total_laps", "lap_time_s"]
    out = probe.warmup_visibility(pd.DataFrame(columns=cols), 0.06)
    assert out["n_stints"] == 0 and out["dip_share"] is None and out["dip_gain_s"]["median"] is None


def test_to_jsonable_handles_numpy_and_nan():
    value = {
        "a": np.float64(1.5),
        "b": np.int64(2),
        "c": float("nan"),
        "d": np.array([1, 2]),
        "e": np.bool_(True),
    }
    assert json.dumps(probe.to_jsonable(value)) == '{"a": 1.5, "b": 2, "c": null, "d": [1, 2], "e": true}'


# End-to-end through the CLI with a fake FastF1 session.

V_MS = 50.0
RADIUS = {2023: 100.0, 2024: 125.0, 2025: 80.0}


class _FakeSession:
    def __init__(self, season: int, n_laps: int = 10, lap_s: float = 90.0, rate_hz: float = 4.0):
        radius = RADIUS[season]
        rows, self.car_data, self.pos_data = [], {}, {}
        for i, (driver, number) in enumerate((("AAA", "1"), ("BBB", "2"))):
            start0 = 100.0 + 3.0 * i  # BBB crosses the line 3 s behind AAA: no traffic exclusion
            # Lap time is 1 s slower on stint lap 2 (race laps 2 and 7), so most stints show an early dip.
            for lap in range(1, n_laps + 1):
                ls, le = start0 + (lap - 1) * lap_s, start0 + lap * lap_s
                pit_in = pd.Timedelta(le - 5.0, unit="s") if (i == 0 and lap == 5) else pd.NaT
                pit_out = pd.Timedelta(ls + 20.0, unit="s") if (i == 0 and lap == 6) else pd.NaT
                rows.append(
                    {
                        "Driver": driver,
                        "DriverNumber": number,
                        "LapNumber": float(lap),
                        "LapTime": pd.Timedelta(lap_s + (1.0 if lap in (2, 7) else 0.0), unit="s"),
                        "LapStartTime": pd.Timedelta(ls, unit="s"),
                        "Time": pd.Timedelta(le, unit="s"),
                        "Stint": 1.0 if lap <= 5 else 2.0,
                        "Compound": "MEDIUM" if lap <= 5 else "HARD",
                        "TyreLife": float(lap if lap <= 5 else lap - 5),
                        "FreshTyre": True,
                        "TrackStatus": "1",
                        "IsAccurate": True,
                        "Deleted": False,
                        "PitInTime": pit_in,
                        "PitOutTime": pit_out,
                        "SpeedI1": 250.0 + lap,
                        "SpeedI2": 260.0 + lap,
                        "SpeedFL": 270.0 + lap,
                        "SpeedST": 300.0 + lap,
                    }
                )
            t = np.arange(start0, start0 + n_laps * lap_s, 1.0 / rate_hz)
            angle = V_MS / radius * t
            self.car_data[number] = pd.DataFrame(
                {
                    "SessionTime": pd.to_timedelta(t, unit="s"),
                    "Speed": np.full_like(t, V_MS * 3.6),
                    "Throttle": np.full_like(t, 80.0),
                    "Brake": (np.arange(t.size) % 4) == 0,
                }
            )
            self.pos_data[number] = pd.DataFrame(
                {
                    "SessionTime": pd.to_timedelta(t, unit="s"),
                    "X": radius * np.cos(angle) * 10.0,
                    "Y": radius * np.sin(angle) * 10.0,
                }
            )
        self.laps = pd.DataFrame(rows)
        self.weather_data = pd.DataFrame(
            {
                "Time": pd.to_timedelta([0, 60], unit="s"),
                "AirTemp": 25.0,
                "TrackTemp": 40.0,
                "Rainfall": False,
            }
        )
        self.event = pd.Series({"EventName": f"Fake GP {season}", "EventDate": f"{season}-05-01"})
        self.total_laps = n_laps

    def load(self, **kwargs):
        return None


def test_cli_end_to_end_with_fake_sessions(tmp_path, monkeypatch):
    import fastf1

    monkeypatch.setattr(fastf1, "get_session", lambda season, gp, kind: _FakeSession(season))
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda *a, **k: None)
    script = _load_script()
    json_out, report_out = tmp_path / "probe.json", tmp_path / "data_probe.md"
    code = script.main(
        [
            "--race", "2023:Spain:barcelona",
            "--race", "2024:Bahrain:bahrain",
            "--race", "2025:Japan:suzuka",
            "--cache-dir", str(tmp_path / "cache"),
            "--json-out", str(json_out),
            "--report-out", str(report_out),
        ]
    )  # fmt: skip
    assert code == 0
    result = json.loads(json_out.read_text(encoding="utf-8"))
    races = {r["event_id"]: r for r in result["races"]}
    for race in races.values():
        assert "error" not in race, race.get("error")
        assert race["telemetry_missing_fraction"] == 0.0
        assert race["o1_keep"] is True
    assert races["2023_barcelona"]["role"] == "dev" and races["2023_barcelona"]["fold"] == "F3"
    assert races["2025_suzuka"]["role"] == "holdout"
    assert "raw_stint_slope_s_per_lap" not in races["2025_suzuka"]
    counts = races["2023_barcelona"]["exclusion_counts"]
    assert counts == {"scored": 16, "race_lap_1": 2, "in_lap": 1, "out_lap": 1}
    assert races["2023_barcelona"]["pit_lane_time_s"]["median"] == pytest.approx(25.0)

    # Stints: AAA 1 (stint laps 2-4 scored), AAA 2 (2-5), BBB 1 (2-5) dip at stint lap 3; BBB 2 has no
    # out-lap exclusion, so its stint lap 1 is scored and fastest. Gain = 1 s - phi_mid.
    phi = result["proxy_comparison"]["phi"]
    for event_id in ("2023_barcelona", "2024_bahrain"):
        warm = races[event_id]["warmup_visibility"]
        assert (warm["n_stints"], warm["n_dip"]) == (4, 3)
        assert warm["dip_gain_s"]["median"] == pytest.approx(1.0 - phi)
        assert warm["first_scored_stint_lap"] == {"median": 2.0, "counts": {"1": 1, "2": 3}}
        assert warm["first_scored_tyre_life_median"] == pytest.approx(2.0)
    assert "warmup_visibility" not in races["2025_suzuka"]

    comparison = result["proxy_comparison"]
    assert comparison["events"] == ["2023_barcelona", "2024_bahrain"]  # holdout never feeds O2
    energy = comparison["by_proxy"]["energy_index"]["event_means"]
    assert energy["2023_barcelona"] == pytest.approx(V_MS**2 / RADIUS[2023], rel=0.02)
    assert energy["2024_bahrain"] == pytest.approx(V_MS**2 / RADIUS[2024], rel=0.02)
    assert result["records"] and all(r["quantity"] and r["instrument"] for r in result["records"])
    report = report_out.read_text(encoding="utf-8")
    assert "## O2 heat-load proxy candidates" in report and "2025_suzuka" in report
    warm_section = report.split("## Warm-up visibility (S4 on real stints)")[1].split("## O2")[0]
    assert "2023_barcelona" in warm_section and "2025_suzuka" not in warm_section
    warm_records = [r for r in result["records"] if "early corrected-pace dip" in r["quantity"]]
    assert sorted(r["population"].split()[0] for r in warm_records) == ["2023_barcelona", "2024_bahrain"]
    assert all(r["value"] == pytest.approx(0.75) for r in warm_records)
