"""Shared contracts: frozen decisions (O1-O8), prediction, metric, and simulation records.

See PROJECT.md section 5 for the module contracts and GLOSSARY.md section 2 for the four-field rule.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EVENT_ID = re.compile(r"^(\d{4})_([a-z][a-z0-9_]*)$")
STATUSES = ("Observed", "Inferred", "Assumed", "Unknown", "Superseded")
FOUR_FIELDS = ("quantity", "population", "conditions", "instrument")


def _event_parts(event_id: str) -> tuple[int, str]:
    match = EVENT_ID.match(event_id)
    if match is None:
        raise ValueError(f"event id {event_id!r} must look like '<season>_<circuit>'")
    return int(match.group(1)), match.group(2)


@dataclass(frozen=True)
class Decisions:
    """Validated O1-O8 decisions. Construction raises ValueError on an inconsistent config."""

    version: int
    status: str
    dev_seasons: tuple[int, ...]
    folds: Mapping[str, tuple[str, ...]]
    dev_events: tuple[str, ...]
    holdout_season: int
    holdout_events: tuple[str, ...]
    phi_low: float
    phi_mid: float
    phi_high: float
    traffic_gap_threshold_s: float
    horizons: tuple[int, ...]
    primary_horizon: int
    k_min: int
    heat_load_proxy: Mapping[str, Any] = field(default_factory=dict)
    target: Mapping[str, Any] = field(default_factory=dict)
    synthetic: Mapping[str, Any] = field(default_factory=dict)
    ensemble: Mapping[str, Any] = field(default_factory=dict)
    strategy: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ("draft", "frozen"):
            raise ValueError(f"status must be 'draft' or 'frozen', got {self.status!r}")
        self._check_folds()
        self._check_events()
        self._check_phi()
        self._check_protocol()

    @property
    def circuit_to_fold(self) -> dict[str, str]:
        return {circuit: fold for fold, circuits in self.folds.items() for circuit in circuits}

    def _check_folds(self) -> None:
        if not self.folds:
            raise ValueError("o1_races.folds must define at least one fold")
        seen: dict[str, str] = {}
        for fold, circuits in self.folds.items():
            if not circuits:
                raise ValueError(f"fold {fold!r} has no circuits")
            for circuit in circuits:
                if circuit in seen:
                    raise ValueError(
                        f"circuit {circuit!r} appears in folds {seen[circuit]!r} and {fold!r}; "
                        "each circuit must belong to exactly one fold"
                    )
                seen[circuit] = fold

    def _check_events(self) -> None:
        circuit_to_fold = self.circuit_to_fold
        dev_event_fold: dict[str, str] = {}
        for event in self.dev_events:
            season, circuit = _event_parts(event)
            if season not in self.dev_seasons:
                raise ValueError(f"development event {event!r} is not in dev_seasons {self.dev_seasons}")
            if circuit not in circuit_to_fold:
                raise ValueError(f"development event {event!r}: circuit {circuit!r} is in no fold")
            if event in dev_event_fold:
                raise ValueError(f"development event {event!r} is listed twice")
            dev_event_fold[event] = circuit_to_fold[circuit]
        if self.holdout_season in self.dev_seasons:
            raise ValueError(f"holdout season {self.holdout_season} is also a development season")
        for event in self.holdout_events:
            if event in dev_event_fold:
                raise ValueError(
                    f"holdout event {event!r} also appears in development fold {dev_event_fold[event]!r}"
                )
            season, _ = _event_parts(event)
            if season != self.holdout_season:
                raise ValueError(f"holdout event {event!r} is not in holdout season {self.holdout_season}")
        if len(set(self.holdout_events)) != len(self.holdout_events):
            raise ValueError("holdout events must be unique")

    def _check_phi(self) -> None:
        phis = (self.phi_low, self.phi_mid, self.phi_high)
        if not all(np.isfinite(phis)):
            raise ValueError(f"phi values must be finite, got {phis}")
        if not self.phi_low <= self.phi_mid <= self.phi_high:
            raise ValueError(
                f"phi_low <= phi_mid <= phi_high must hold, got "
                f"phi_low={self.phi_low}, phi_mid={self.phi_mid}, phi_high={self.phi_high}"
            )

    def _check_protocol(self) -> None:
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError(f"horizons must be positive integers, got {self.horizons}")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError(f"horizons must be unique, got {self.horizons}")
        if self.primary_horizon not in self.horizons:
            raise ValueError(f"primary_horizon {self.primary_horizon} is not in horizons {self.horizons}")
        if self.k_min < 1:
            raise ValueError(f"k_min must be at least 1, got {self.k_min}")


def load_decisions(path: str | Path) -> Decisions:
    """Read and validate a decisions YAML file (configs/decisions_v1.yaml layout)."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        o1 = raw["o1_races"]
        o3 = raw["o3_race_lap_trend"]
        o5 = raw["o5_protocol"]
        return Decisions(
            version=int(raw["version"]),
            status=str(raw["status"]),
            dev_seasons=tuple(int(s) for s in o1["dev_seasons"]),
            folds={str(k): tuple(str(c) for c in v) for k, v in o1["folds"].items()},
            dev_events=tuple(str(e) for e in o1["dev_events"]),
            holdout_season=int(o1["holdout"]["season"]),
            holdout_events=tuple(str(e) for e in o1["holdout"]["events"]),
            phi_low=float(o3["phi_low"]),
            phi_mid=float(o3["phi_mid"]),
            phi_high=float(o3["phi_high"]),
            traffic_gap_threshold_s=float(o3["traffic_gap_threshold_s"]),
            horizons=tuple(int(h) for h in o5["horizons"]),
            primary_horizon=int(o5["primary_horizon"]),
            k_min=int(o5["k_min"]),
            heat_load_proxy=dict(raw.get("o2_heat_load_proxy") or {}),
            target=dict(raw.get("o4_target") or {}),
            synthetic=dict(raw.get("o6_synthetic") or {}),
            ensemble=dict(raw.get("o7_ensemble") or {}),
            strategy=dict(raw.get("o8_strategy") or {}),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing or malformed decisions key: {exc}") from exc


def _float_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True, eq=False)  # eq=False: elementwise array == has no single truth value
class PredictionOutput:
    """Forecast for one origin (PROJECT section 5.2). E = 1 for single models."""

    mean_s: np.ndarray  # (horizon,)
    member_means_s: np.ndarray  # (E, horizon)
    sigma_s: np.ndarray  # (horizon,), total predictive std
    latent_h: np.ndarray | None  # None for models without latent states (B0-B3)
    latent_w_s: np.ndarray | None
    member_sigma_s: np.ndarray | None = None  # optional exact Gaussian component scales, (E, horizon)

    def __post_init__(self) -> None:
        mean = _float_array("mean_s", self.mean_s)
        members = _float_array("member_means_s", self.member_means_s)
        sigma = _float_array("sigma_s", self.sigma_s)
        if mean.ndim != 1 or mean.shape[0] < 1:
            raise ValueError(f"mean_s must have shape (horizon,), got {mean.shape}")
        horizon = mean.shape[0]
        if members.ndim != 2 or members.shape[0] < 1 or members.shape[1] != horizon:
            raise ValueError(f"member_means_s must have shape (E, {horizon}), got {members.shape}")
        if sigma.shape != (horizon,):
            raise ValueError(f"sigma_s must have shape ({horizon},), got {sigma.shape}")
        if np.any(sigma < 0):
            raise ValueError("sigma_s must be non-negative")
        object.__setattr__(self, "mean_s", mean)
        object.__setattr__(self, "member_means_s", members)
        object.__setattr__(self, "sigma_s", sigma)
        if self.member_sigma_s is not None:
            component = _float_array("member_sigma_s", self.member_sigma_s)
            if component.shape != members.shape or np.any(component < 0):
                raise ValueError("member_sigma_s must match member_means_s and be nonnegative")
            variance = np.mean(component**2, axis=0) + np.var(members, axis=0)
            if not np.allclose(mean, members.mean(axis=0), rtol=1e-6, atol=1e-8):
                raise ValueError("member_sigma_s mixture mean disagrees with mean_s")
            if not np.allclose(sigma**2, variance, rtol=1e-6, atol=1e-8):
                raise ValueError("member_sigma_s mixture variance disagrees with sigma_s")
            object.__setattr__(self, "member_sigma_s", component)
        for name in ("latent_h", "latent_w_s"):
            value = getattr(self, name)
            if value is None:
                continue
            arr = _float_array(name, value)
            if arr.ndim < 1 or arr.shape[-1] != horizon:
                raise ValueError(f"{name} must end in the horizon axis ({horizon}), got {arr.shape}")
            object.__setattr__(self, name, arr)


@dataclass(frozen=True)
class MetricRecord:
    """One reported number with its four fields (GLOSSARY section 2) and epistemic status."""

    quantity: str | Mapping[str, Any]
    population: str | Mapping[str, Any]
    conditions: str | Mapping[str, Any]
    instrument: str | Mapping[str, Any]
    value: float | None
    status: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        for name in FOUR_FIELDS:
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

    def differing_fields(self, other: MetricRecord) -> tuple[str, ...]:
        return tuple(name for name in FOUR_FIELDS if getattr(self, name) != getattr(other, name))

    def comparable(self, other: MetricRecord) -> bool:
        """True only if all four fields match."""
        return not self.differing_fields(other)

    def require_comparable(self, other: MetricRecord) -> None:
        """Raise ValueError naming every field that differs."""
        differing = self.differing_fields(other)
        if differing:
            details = "; ".join(
                f"{name}: {getattr(self, name)!r} != {getattr(other, name)!r}" for name in differing
            )
            raise ValueError(f"records are not comparable, differing fields: {details}")


@dataclass(frozen=True, eq=False)
class SimulationOutput:
    """Per-strategy finish-time distribution summary (PROJECT section 4.8)."""

    strategy_ids: tuple[str, ...]
    finish_times_s: np.ndarray  # (n_strategies, n_rollouts)
    mean_s: np.ndarray  # (n_strategies,)
    std_s: np.ndarray
    p10_s: np.ndarray
    p50_s: np.ndarray
    p90_s: np.ndarray
    rank: np.ndarray  # (n_strategies,), 1 = best

    def __post_init__(self) -> None:
        finish = _float_array("finish_times_s", self.finish_times_s)
        n = len(self.strategy_ids)
        if finish.ndim != 2 or finish.shape[0] != n or finish.shape[1] < 1:
            raise ValueError(f"finish_times_s must have shape ({n}, n_rollouts), got {finish.shape}")
        object.__setattr__(self, "strategy_ids", tuple(self.strategy_ids))
        object.__setattr__(self, "finish_times_s", finish)
        for name in ("mean_s", "std_s", "p10_s", "p50_s", "p90_s"):
            arr = _float_array(name, getattr(self, name))
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
            object.__setattr__(self, name, arr)
        if np.any(self.std_s < 0):
            raise ValueError("std_s must be non-negative")
        if np.any(self.p10_s > self.p50_s) or np.any(self.p50_s > self.p90_s):
            raise ValueError("p10_s <= p50_s <= p90_s must hold for every strategy")
        rank = np.asarray(self.rank)
        if rank.shape != (n,) or sorted(rank.tolist()) != list(range(1, n + 1)):
            raise ValueError(f"rank must be a permutation of 1..{n}, got {rank.tolist()}")
        object.__setattr__(self, "rank", rank)
