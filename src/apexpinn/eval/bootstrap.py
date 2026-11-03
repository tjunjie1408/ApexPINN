"""Paired hierarchical bootstrap: races, then stints within sampled races (PRD section 6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from apexpinn.eval.metrics import rmse


@dataclass(frozen=True)
class BootstrapResult:
    """Relative RMSE reduction as a fraction; positive favors the model over the baseline."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    seed: int
    n_races: int
    n_stints: int
    quantity: dict[str, Any]
    population: dict[str, Any]
    conditions: dict[str, Any]
    instrument: dict[str, Any]


def _ids(values: Any, length: int, name: str) -> np.ndarray:
    ids = np.asarray(values, dtype=object)
    if ids.shape != (length,) or any(not isinstance(v, str) or not v.strip() for v in ids):
        raise ValueError(f"{name} must be nonempty string identifiers aligned to the targets")
    return ids


def paired_bootstrap_rel_rmse(
    target: Any,
    prediction: Any,
    baseline: Any,
    *,
    race_ids: Any,
    stint_ids: Any,
    n_resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Return 1-RMSE(model)/RMSE(baseline), preserving paired indices and all within-stint origins.

    Undefined baseline denominators raise, including in a replicate. No failed draws are discarded.
    Arrays must already refer to the identical model-independent population and one horizon/phi.
    """
    model_rmse, base_rmse = rmse(target, prediction), rmse(target, baseline)
    if base_rmse == 0:
        raise ValueError("relative RMSE is undefined for zero baseline RMSE")
    for name, value, minimum in (("n_resamples", n_resamples, 2), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    y, predicted, reference = (np.asarray(v, dtype=np.float64) for v in (target, prediction, baseline))
    races, stints = _ids(race_ids, len(y), "race_ids"), _ids(stint_ids, len(y), "stint_ids")
    unique_races = sorted(set(races))
    if len(unique_races) < 2:
        raise ValueError("race-level bootstrap requires at least two races")
    clusters = []
    for race in unique_races:
        within = sorted(set(stints[races == race]))
        clusters.append([np.flatnonzero((races == race) & (stints == stint)) for stint in within])
    rng = np.random.default_rng(seed)
    reductions = np.empty(n_resamples, dtype=np.float64)
    squared_model, squared_baseline = (predicted - y) ** 2, (reference - y) ** 2
    for draw in range(n_resamples):
        chosen = []
        for race_index in rng.integers(len(clusters), size=len(clusters)):
            race_stints = clusters[race_index]
            chosen.extend(race_stints[i] for i in rng.integers(len(race_stints), size=len(race_stints)))
        indices = np.concatenate(chosen)
        denominator = float(np.mean(squared_baseline[indices]))
        if denominator == 0:
            raise ValueError("zero baseline RMSE in a bootstrap replicate; no draws discarded")
        reductions[draw] = 1 - np.sqrt(float(np.mean(squared_model[indices])) / denominator)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(reductions, [tail, 1 - tail])
    return BootstrapResult(
        1 - model_rmse / base_rmse,
        float(lower),
        float(upper),
        confidence,
        n_resamples,
        seed,
        len(unique_races),
        sum(len(c) for c in clusters),
        quantity={"metric": "paired_relative_rmse_reduction", "unit": "fraction"},
        population={"n_records": len(y), "races": unique_races, "n_stints": sum(len(c) for c in clusters)},
        conditions={"confidence": confidence, "seed": seed, "n_resamples": n_resamples},
        instrument={
            "method": "paired race-then-stint percentile bootstrap/v1",
            "formula": "1 - RMSE(model)/RMSE(baseline)",
        },
    )
