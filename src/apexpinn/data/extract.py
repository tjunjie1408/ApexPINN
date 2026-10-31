"""Bronze extract (PLAN T8, PROJECT section 3.1): FastF1 race sessions to Parquet plus a SHA-256 manifest.

Per event the output directory ``<out-root>/<event_id>/`` holds:

- ``laps.parquet``: FastF1 laps as loaded. Timedelta columns become float seconds named ``<column>_s``
  (for example ``LapTime`` becomes ``LapTime_s``); other columns keep their FastF1 names.
- ``weather.parquet``: FastF1 weather, same timedelta rule (``Time`` becomes ``Time_s``).
- ``lap_telemetry.parquet``: per-lap telemetry aggregates from ``apexpinn.data.probe.per_lap_telemetry``,
  computed per driver from the samples inside each lap, keyed by ``DriverNumber`` and ``LapNumber``.
- ``manifest.json``: see ``apexpinn.data.manifest``. Written last, so a directory without a valid manifest
  is an incomplete extract.

Re-running on an event whose manifest verifies is a no-op unless ``--force`` is given. Needs network
access for uncached sessions. Example (from the repository root):

    python -m apexpinn.data.extract --set dev --out-root data/bronze --cache-dir .fastf1_cache
    python -m apexpinn.data.extract --event 2023_spa --event "2024_imola:Emilia Romagna Grand Prix" --force

Events are ``<season>_<circuit>`` ids as in configs/decisions_v1.yaml. ``GRAND_PRIX_QUERY`` maps each
circuit id to the FastF1 event name used for the lookup; append ``:<name>`` to an event to override it.
The manifest records the event name FastF1 resolved, so a wrong lookup shows up there.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apexpinn.contracts import EVENT_ID, Decisions, load_decisions
from apexpinn.data import manifest as mf
from apexpinn.data import probe

SESSION_KIND = "R"
LAPS_FILE = "laps.parquet"
WEATHER_FILE = "weather.parquet"
LAP_TELEMETRY_FILE = "lap_telemetry.parquet"
BRONZE_FILES = (LAPS_FILE, WEATHER_FILE, LAP_TELEMETRY_FILE)
WEATHER_EMPTY_COLUMNS = ("Time_s", "AirTemp", "TrackTemp", "Rainfall")  # written when FastF1 has none
TELEMETRY_KEYS = ("DriverNumber", "Driver", "LapNumber", "lap_start_s", "lap_end_s")

# Circuit id (configs/decisions_v1.yaml) -> FastF1 event name for fastf1.get_session.
GRAND_PRIX_QUERY: dict[str, str] = {
    "bahrain": "Bahrain Grand Prix",
    "jeddah": "Saudi Arabian Grand Prix",
    "melbourne": "Australian Grand Prix",
    "suzuka": "Japanese Grand Prix",
    "shanghai": "Chinese Grand Prix",
    "miami": "Miami Grand Prix",
    "imola": "Emilia Romagna Grand Prix",
    "monaco": "Monaco Grand Prix",
    "barcelona": "Spanish Grand Prix",
    "montreal": "Canadian Grand Prix",
    "spielberg": "Austrian Grand Prix",
    "silverstone": "British Grand Prix",
    "budapest": "Hungarian Grand Prix",
    "spa": "Belgian Grand Prix",
    "zandvoort": "Dutch Grand Prix",
    "monza": "Italian Grand Prix",
    "baku": "Azerbaijan Grand Prix",
    "singapore": "Singapore Grand Prix",
    "austin": "United States Grand Prix",
    "mexico_city": "Mexico City Grand Prix",
    "interlagos": "São Paulo Grand Prix",
    "las_vegas": "Las Vegas Grand Prix",
    "lusail": "Qatar Grand Prix",
    "yas_marina": "Abu Dhabi Grand Prix",
}

SessionLoader = Callable[[int, str, str], Any]

log = logging.getLogger("apexpinn.extract")


# FastF1 frame conversion. Same conventions as scripts/probe_fastf1.py: float seconds of session time,
# speed in km/h, positions in m.


def _seconds(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_timedelta(frame[col]).dt.total_seconds()


def laps_to_frame(laps: pd.DataFrame) -> pd.DataFrame:
    """FastF1 laps plus float-second columns lap_time_s, lap_end_s, lap_start_s, pit_in_s, pit_out_s."""
    df = pd.DataFrame(laps).reset_index(drop=True)
    df["lap_time_s"] = _seconds(df, "LapTime")
    df["lap_end_s"] = _seconds(df, "Time")
    df["lap_start_s"] = _seconds(df, "LapStartTime")
    df["pit_in_s"] = _seconds(df, "PitInTime")
    df["pit_out_s"] = _seconds(df, "PitOutTime")
    return df


def car_to_frame(car: pd.DataFrame | None) -> pd.DataFrame | None:
    """FastF1 car data to ``t_s``, ``speed_kmh``, ``brake``; None if absent or empty."""
    if car is None or car.empty:
        return None
    return pd.DataFrame(
        {
            "t_s": _seconds(car, "SessionTime"),
            "speed_kmh": car["Speed"].astype(float),
            "brake": car["Brake"],
        }
    )


def pos_to_frame(pos: pd.DataFrame | None) -> pd.DataFrame | None:
    """FastF1 position data (1/10 m) to ``t_s``, ``x_m``, ``y_m``; None if absent or empty."""
    if pos is None or pos.empty:
        return None
    return pd.DataFrame(
        {"t_s": _seconds(pos, "SessionTime"), "x_m": pos["X"] / 10.0, "y_m": pos["Y"] / 10.0}
    )


def _safe_attr(session: Any, name: str) -> Any:
    """Session attribute, or None if FastF1 could not load it (it raises DataNotLoadedError then)."""
    try:
        return getattr(session, name)
    except Exception:  # FastF1 raises its own exception types for data it failed to load
        log.warning("session.%s is not available", name)
        return None


def timedeltas_to_seconds(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy with every timedelta column replaced by float seconds named ``<column>_s``, in place order."""
    out = {}
    for col in frame.columns:
        if frame[col].dtype.kind == "m":
            out[f"{col}_s"] = frame[col].dt.total_seconds()
        else:
            out[str(col)] = frame[col]
    return pd.DataFrame(out, index=frame.index).reset_index(drop=True)


def lap_telemetry_frame(
    laps: pd.DataFrame, car_data: Mapping[str, pd.DataFrame], pos_data: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Per-lap telemetry aggregates, one row per lap in ``laps`` order.

    ``laps`` is FastF1 laps (``DriverNumber``, ``LapNumber``, ``LapStartTime``, ``Time``). Each driver's
    laps use only that driver's streams, and each lap only the samples inside its window
    (``probe.per_lap_telemetry``).
    """
    frame = laps_to_frame(laps)
    numbers = frame["DriverNumber"].astype(str)
    parts = [
        probe.per_lap_telemetry(
            group, car_to_frame(car_data.get(number)), pos_to_frame(pos_data.get(number))
        )
        for number, group in frame.groupby(numbers, sort=True)
    ]
    keys = frame[[c for c in TELEMETRY_KEYS if c in frame.columns]].copy()
    keys["DriverNumber"] = numbers
    if not parts:
        return keys
    return keys.join(pd.concat(parts)).reset_index(drop=True)


def _weather_frame(weather: pd.DataFrame | None) -> pd.DataFrame:
    if weather is None or weather.empty:
        return pd.DataFrame(columns=list(WEATHER_EMPTY_COLUMNS), dtype=float)
    return timedeltas_to_seconds(pd.DataFrame(weather))


def parse_event(spec: str) -> tuple[str, str]:
    """``<season>_<circuit>[:<FastF1 event name>]`` to (event_id, FastF1 event name)."""
    event_id, _, query = spec.partition(":")
    match = EVENT_ID.match(event_id)
    if match is None:
        raise ValueError(f"event {spec!r} must look like '<season>_<circuit>[:<FastF1 event name>]'")
    circuit = match.group(2)
    if not query:
        if circuit not in GRAND_PRIX_QUERY:
            raise ValueError(f"no FastF1 event name for circuit {circuit!r}; append ':<event name>'")
        query = GRAND_PRIX_QUERY[circuit]
    return event_id, query


def config_events(decisions: Decisions, which: str) -> list[str]:
    """Event ids of the ``dev``, ``holdout``, or ``all`` set in the decisions config."""
    if which == "dev":
        return list(decisions.dev_events)
    if which == "holdout":
        return list(decisions.holdout_events)
    if which == "all":
        return [*decisions.dev_events, *decisions.holdout_events]
    raise ValueError(f"event set must be dev, holdout, or all, got {which!r}")


def _default_loader(season: int, grand_prix: str, kind: str) -> Any:
    import fastf1

    return fastf1.get_session(season, grand_prix, kind)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class ExtractResult:
    event_id: str
    status: str  # "written" or "skipped"
    event_dir: Path
    manifest: Mapping[str, Any]


def extract_event(
    event_id: str,
    out_root: str | Path,
    grand_prix: str | None = None,
    *,
    force: bool = False,
    get_session: SessionLoader | None = None,
    now: datetime | None = None,
) -> ExtractResult:
    """Extract one race to ``out_root/event_id`` and return what happened.

    If the event directory already verifies against its manifest and ``force`` is False, nothing is loaded
    or written. ``get_session(season, grand_prix, kind)`` defaults to ``fastf1.get_session``; tests pass a
    fake. ``now`` overrides the download timestamp.
    """
    if grand_prix is None:
        event_id, grand_prix = parse_event(event_id)
    match = EVENT_ID.match(event_id)
    if match is None:
        raise ValueError(f"event id {event_id!r} must look like '<season>_<circuit>'")
    season, circuit = int(match.group(1)), match.group(2)
    event_dir = Path(out_root) / event_id

    if not force:
        problems = mf.verify_event(event_dir, BRONZE_FILES)
        if not problems:
            log.info("%s: Bronze extract exists and verifies, skipping (use --force to redo)", event_id)
            return ExtractResult(event_id, "skipped", event_dir, mf.read_manifest(event_dir) or {})
        if event_dir.exists():
            log.warning("%s: existing extract is incomplete (%s), extracting again", event_id, problems)

    loader = get_session or _default_loader
    session = loader(season, grand_prix, SESSION_KIND)
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    downloaded = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    raw_laps = _safe_attr(session, "laps")
    if raw_laps is None or len(raw_laps) == 0:
        raise ValueError(f"{event_id}: FastF1 returned no laps for {season} {grand_prix!r}")
    car_data = {str(k): v for k, v in (_safe_attr(session, "car_data") or {}).items()}
    pos_data = {str(k): v for k, v in (_safe_attr(session, "pos_data") or {}).items()}
    laps = pd.DataFrame(raw_laps)
    frames = {
        LAPS_FILE: timedeltas_to_seconds(laps),
        WEATHER_FILE: _weather_frame(_safe_attr(session, "weather_data")),
        LAP_TELEMETRY_FILE: lap_telemetry_frame(laps, car_data, pos_data),
    }

    # Drop the old manifest first: until the new one is written the directory reads as incomplete.
    (event_dir / mf.MANIFEST_NAME).unlink(missing_ok=True)
    files = {name: mf.write_parquet(frame, event_dir / name) for name, frame in frames.items()}
    event = _safe_attr(session, "event")
    event = event if event is not None else {}
    driver_numbers = set(laps["DriverNumber"].astype(str)) if "DriverNumber" in laps else set()
    manifest = {
        "schema": mf.MANIFEST_SCHEMA,
        "event_id": event_id,
        "season": season,
        "circuit": circuit,
        "session": SESSION_KIND,
        "grand_prix_query": grand_prix,
        "event_name": str(event.get("EventName", "")),
        "location": str(event.get("Location", "")),
        "event_date": str(event.get("EventDate", "")),
        "downloaded_utc": downloaded,
        "versions": {
            "fastf1": _package_version("fastf1"),
            "pandas": pd.__version__,
            "pyarrow": _package_version("pyarrow"),
            "apexpinn": _package_version("apexpinn"),
        },
        "drivers_without_car_data": sorted(driver_numbers - set(car_data)),
        "drivers_without_pos_data": sorted(driver_numbers - set(pos_data)),
        "files": files,
    }
    mf.write_manifest(event_dir, manifest)
    log.info("%s: wrote %s (%d laps)", event_id, event_dir, files[LAPS_FILE]["rows"])
    return ExtractResult(event_id, "written", event_dir, manifest)


def extract_events(
    events: Iterable[tuple[str, str]],
    out_root: str | Path,
    *,
    force: bool = False,
    get_session: SessionLoader | None = None,
) -> tuple[list[ExtractResult], dict[str, str]]:
    """Extract each (event_id, FastF1 event name). One failing event does not stop the others."""
    results: list[ExtractResult] = []
    errors: dict[str, str] = {}
    for event_id, grand_prix in events:
        try:
            result = extract_event(event_id, out_root, grand_prix, force=force, get_session=get_session)
            results.append(result)
        except Exception as exc:  # one bad race must not lose the others
            log.exception("%s: extract failed", event_id)
            errors[event_id] = repr(exc)
    return results, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--event", action="append", default=[], help="<season>_<circuit>[:<FastF1 event name>], repeatable"
    )
    parser.add_argument("--set", choices=("dev", "holdout", "all"), help="events from the decisions config")
    parser.add_argument("--config", type=Path, default=Path("configs/decisions_v1.yaml"))
    parser.add_argument("--out-root", type=Path, default=Path("data/bronze"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".fastf1_cache"))
    parser.add_argument("--force", action="store_true", help="extract again even if a valid extract exists")
    args = parser.parse_args(argv)

    specs = list(args.event)
    if args.set:
        specs += config_events(load_decisions(args.config), args.set)
    if not specs:
        parser.error("give --event and/or --set")
    try:
        events = list(dict.fromkeys(parse_event(s) for s in specs))
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import fastf1

    fastf1.set_log_level("WARNING")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(args.cache_dir))

    results, errors = extract_events(events, args.out_root, force=args.force)
    written = sum(r.status == "written" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    log.info("written %d, skipped %d, failed %d", written, skipped, len(errors))
    for event_id, error in errors.items():
        log.error("failed: %s: %s", event_id, error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
