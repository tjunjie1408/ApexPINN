from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "configs" / "compound_allocation.yaml"


@pytest.fixture(scope="module")
def table() -> dict:
    assert TABLE_PATH.is_file(), "T7 compound allocation table is missing"
    return yaml.safe_load(TABLE_PATH.read_text(encoding="utf-8"))


def test_table_covers_every_configured_event(table):
    decisions = yaml.safe_load((ROOT / "configs" / "decisions_v1.yaml").read_text(encoding="utf-8"))
    races = decisions["o1_races"]
    expected = set(races["dev_events"]) | set(races["holdout"]["events"])
    assert set(table["events"]) == expected


def test_table_schema_and_evidence(table):
    assert table["version"] == 1
    assert table["evidence"] == "Observed"


def test_event_ids_are_not_duplicated(table):
    document = yaml.compose(TABLE_PATH.read_text(encoding="utf-8"))
    events = next(value for key, value in document.value if key.value == "events")
    names = [key.value for key, _ in events.value]
    assert len(names) == len(set(names))


def test_each_event_has_distinct_ordered_season_valid_compounds(table):
    ranges = {2023: range(0, 6), 2024: range(1, 6), 2025: range(1, 7)}
    for event_id, allocation in table["events"].items():
        assert set(allocation) == {"HARD", "MEDIUM", "SOFT", "source_url"}, event_id
        labels = [allocation[name] for name in ("HARD", "MEDIUM", "SOFT")]
        assert all(isinstance(label, str) and len(label) == 2 and label[0] == "C" for label in labels)
        numbers = [int(label[1]) for label in labels]
        assert numbers[0] < numbers[1] < numbers[2], event_id
        assert all(number in ranges[int(event_id[:4])] for number in numbers), event_id


def test_each_event_names_a_pirelli_article(table):
    for event_id, allocation in table["events"].items():
        source = urlparse(allocation["source_url"])
        assert source.scheme == "https" and source.netloc == "press.pirelli.com", event_id
        assert source.path not in ("", "/", "/en/"), event_id
        assert not source.query and not source.fragment, event_id


@pytest.mark.parametrize(
    ("event_id", "expected"),
    [
        ("2023_melbourne", ("C2", "C3", "C4")),
        ("2024_melbourne", ("C3", "C4", "C5")),
        ("2025_imola", ("C4", "C5", "C6")),
        ("2025_spa", ("C1", "C3", "C4")),
        ("2025_austin", ("C1", "C3", "C4")),
        ("2025_mexico_city", ("C2", "C4", "C5")),
    ],
)
def test_source_checked_season_changes_and_nonconsecutive_choices(table, event_id, expected):
    allocation = table["events"][event_id]
    assert tuple(allocation[name] for name in ("HARD", "MEDIUM", "SOFT")) == expected
