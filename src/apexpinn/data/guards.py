"""Original-lap origin eligibility and leakage-safe model inputs (PLAN T9, approved O5).

Gold adds normalized ``proxy_q_heat``. Context circuit means must be supplied from training-only
statistics. Targets are returned separately; an excluded intermediate lap is NaN, never shifted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from apexpinn.models.stints import Scenario, StintContext, StintHistory, validate_horizon
from apexpinn.physics.residuals import COMPOUNDS


class OriginRejected(ValueError):
    """Model-independent population rejection, with a stable reason for origin audits."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, eq=False)
class FutureDesign:
    """Only deterministic counters and assumed heat load, one entry per original future lap."""

    tau: np.ndarray
    tyre_age: np.ndarray
    race_lap: np.ndarray
    q: np.ndarray

    @property
    def scenario(self) -> Scenario:
        return Scenario(q_bar=float(self.q[0]))


def _one_stint(stint: pd.DataFrame) -> pd.DataFrame:
    required = {"session_id", "driver_id", "stint_id", "stint_lap", "race_lap", "is_scored_lap"}
    if missing := required - set(stint.columns):
        raise ValueError(f"stint missing columns: {sorted(missing)}")
    if stint.empty or any(
        stint[c].nunique(dropna=False) != 1 for c in ("session_id", "driver_id", "stint_id")
    ):
        raise ValueError("slice_at_origin requires exactly one stint")
    for name in ("stint_lap", "race_lap"):
        values = stint[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 1) or np.any(values % 1):
            raise ValueError(f"{name} must be positive integers")
        if stint[name].duplicated().any():
            raise ValueError(f"duplicate {name} in stint")
    ordered = stint.sort_values("stint_lap").copy()
    if (ordered.race_lap - ordered.stint_lap).nunique() != 1:
        raise ValueError("race_lap and stint_lap must preserve the same original-lap offset")
    if ordered.is_scored_lap.isna().any() or not ordered.is_scored_lap.isin([True, False]).all():
        raise ValueError("is_scored_lap must contain nonmissing boolean flags")
    return ordered.set_index("stint_lap", drop=False)


def _validate_origin(t0: int, horizon: int, k_min: int) -> None:
    for name, value, minimum in (("t0", t0, 1), ("k_min", k_min, 3)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    validate_horizon(horizon)


def _rejection_reason(stint: pd.DataFrame, t0: int, horizon: int, k_min: int) -> str:
    window = np.arange(t0 - k_min + 1, t0 + 1)
    history = stint.reindex(window)
    if t0 < k_min or not history.is_scored_lap.eq(True).all():
        return "history_not_consecutive_clean"
    if t0 + horizon not in stint.index:
        return "target_missing"
    if not stint.loc[t0 + horizon, "is_scored_lap"]:
        return "target_not_scored"
    return ""


def slice_at_origin(
    stint: pd.DataFrame,
    t0: int,
    horizon: int,
    *,
    circuit_q_mean: float,
    k_min: int = 3,
) -> tuple[StintHistory, FutureDesign, np.ndarray]:
    """Return past-only inputs and separate labels on original laps t0+1..t0+horizon.

    Require clean consecutive history and a clean exact endpoint. Older clean history is retained.
    Intermediate excluded/missing targets are NaN. Future observations never participate in input
    validation or construction; population flags may still select which targets are scored.
    """
    _validate_origin(t0, horizon, k_min)
    ordered = _one_stint(stint)
    if reason := _rejection_reason(ordered, t0, horizon, k_min):
        raise OriginRejected(reason)
    past = ordered.loc[ordered.stint_lap <= t0]
    clean = past.loc[past.is_scored_lap.eq(True)]
    required = {
        "compound_c",
        "obs_tyre_life",
        "obs_fresh_tyre",
        "obs_track_temp_c",
        "proxy_q_heat",
        "proxy_y",
    }
    if missing := required - set(past.columns):
        raise ValueError(f"model inputs missing columns: {sorted(missing)}")
    compounds = past.compound_c.dropna().unique()
    if len(compounds) != 1 or compounds[0] not in COMPOUNDS:
        raise ValueError(
            f"unsupported or inconsistent compound_c {compounds.tolist()}; no implicit remapping"
        )
    q = clean.proxy_q_heat.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("historical proxy_q_heat must be finite and normalized to [0, 1] by T10")
    if not np.isfinite(circuit_q_mean) or not 0 <= circuit_q_mean <= 1:
        raise ValueError("circuit_q_mean must be a training-only normalized mean in [0, 1]")
    first = past.iloc[0]
    if first.stint_lap != 1 or pd.isna(first.obs_fresh_tyre):
        raise ValueError("stint-start context is missing")
    ctx = StintContext(
        compound_idx=COMPOUNDS.index(compounds[0]),
        age_at_start=float(first.obs_tyre_life),
        fresh=bool(first.obs_fresh_tyre),
        track_temp_start=float(first.obs_track_temp_c),
        circuit_q_mean=float(circuit_q_mean),
    )
    history = StintHistory(
        stint_id=str(first.stint_id),
        t0=int(t0),
        tau=clean.stint_lap.to_numpy(dtype=np.float64, copy=True),
        y=clean.proxy_y.to_numpy(dtype=np.float64, copy=True),
        q=q,
        ctx=ctx,
        race_lap_t0=int(ordered.loc[t0, "race_lap"]),
    )
    age_at_origin = float(past.loc[t0, "obs_tyre_life"])
    if not np.isfinite(age_at_origin) or age_at_origin < 0:
        raise ValueError("obs_tyre_life at the origin must be finite and nonnegative")
    future_tau = np.arange(t0 + 1, t0 + horizon + 1, dtype=np.int64)
    scenario = Scenario.from_history(history)
    future = FutureDesign(
        tau=future_tau,
        tyre_age=age_at_origin + np.arange(1, horizon + 1),
        race_lap=history.race_lap_t0 + np.arange(1, horizon + 1),
        q=np.full(horizon, scenario.q_bar, dtype=np.float64),
    )
    targets = ordered.reindex(future_tau)
    labels = targets.proxy_y.where(targets.is_scored_lap.eq(True)).to_numpy(dtype=np.float64, copy=True)
    return history, future, labels


def audit_origins(stints: pd.DataFrame, *, horizons: Sequence[int], k_min: int = 3) -> pd.DataFrame:
    """One population decision per (stint, origin, horizon), including rejected candidates.

    Denominator: original origins k_min..max_stint_lap-horizon, whether clean or filtered. Missing
    internal rows are tested as candidate origins too. Input-quality/model errors are not population
    rejections; T10/T11 must record those separately. No normalized proxy is needed for this audit.
    """
    for horizon in horizons:
        _validate_origin(k_min, horizon, k_min)
    records = []
    for _, group in stints.groupby("stint_id", sort=True):
        stint = _one_stint(group)
        for horizon in horizons:
            for t0 in range(k_min, int(stint.stint_lap.max()) - horizon + 1):
                reason = _rejection_reason(stint, t0, horizon, k_min)
                records.append(
                    {
                        "session_id": stint.session_id.iloc[0],
                        "stint_id": stint.stint_id.iloc[0],
                        "split_fold": stint.split_fold.iloc[0],
                        "horizon": horizon,
                        "t0": t0,
                        "eligible": not reason,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(
        records,
        columns=[
            "session_id",
            "stint_id",
            "split_fold",
            "horizon",
            "t0",
            "eligible",
            "reason",
        ],
    )
