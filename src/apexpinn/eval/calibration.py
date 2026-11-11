"""Clean, horizon-specific affine mixture recalibration. No conformal coverage guarantee."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from apexpinn.cloud.artifacts import identity
from apexpinn.eval.metrics import central_interval


def split_plan(decisions, outer_fold: str) -> dict:
    order = list(decisions.folds)
    if outer_fold not in order or len(order) < 4:
        raise ValueError("clean calibration needs a known outer fold and at least four circuit groups")
    all_circuits = [c for f in order for c in decisions.folds[f]]
    if len(all_circuits) != len(set(all_circuits)):
        raise ValueError("circuit groups overlap")
    i = order.index(outer_fold)
    validation, calibration = order[(i + 1) % len(order)], order[(i + 2) % len(order)]
    return {
        "outer_fold": outer_fold,
        "validation_fold": validation,
        "calibration_fold": calibration,
        "outer_circuits": list(decisions.folds[outer_fold]),
        "validation_circuits": list(decisions.folds[validation]),
        "calibration_circuits": list(decisions.folds[calibration]),
        "tuning_circuits": sorted(
            c for f in order if f not in (outer_fold, validation, calibration) for c in decisions.folds[f]
        ),
        "final_circuits": sorted(
            c for f in order if f not in (outer_fold, calibration) for c in decisions.folds[f]
        ),
    }


def event_weights(events: list[str]) -> np.ndarray:
    if not events:
        raise ValueError("empty calibration population")
    counts = Counter(events)
    return np.array([1 / (len(counts) * counts[e]) for e in events], dtype=float)


def weighted_quantile(values, weights, probability: float) -> float:
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if (
        values.ndim != 1
        or not values.size
        or values.shape != weights.shape
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
        or not 0 < probability < 1
    ):
        raise ValueError(
            "finite values, nonnegative positive-total weights and probability in (0,1) required"
        )
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = min(int(np.searchsorted(cumulative, probability, side="left")), len(values) - 1)
    return float(values[order[index]])


def validate_calibration_rows(rows: list[dict], plan: dict, *, horizons, minimum_events: int) -> None:
    if not rows:
        raise ValueError("empty calibration rows")
    keys = set()
    for row in rows:
        if (
            row["fold"] != plan["calibration_fold"]
            or row["conditions"].get("data_role") != "calibration"
            or row["conditions"].get("outer_fold") != plan["outer_fold"]
        ):
            raise ValueError("only the declared clean calibration population may fit scales")
        key = (row["event_id"], row["stint_id"], row["t0"], row["horizon"], row["phi"])
        if key in keys:
            raise ValueError("duplicate calibration endpoint")
        keys.add(key)
    if {r["horizon"] for r in rows} != set(horizons):
        raise ValueError("calibration horizons differ from declaration")
    if any(len({r["event_id"] for r in rows if r["horizon"] == h}) < minimum_events for h in horizons):
        raise ValueError("too few calibration events in a horizon")
    for field in ("model_hash", "normalization_numeric_sha256"):
        if len({r["instrument"][field] for r in rows}) != 1:
            raise ValueError("calibration rows must use one fixed ensemble and numerical normalization")
    if len({r["phi"] for r in rows}) != 1:
        raise ValueError("mixed phi values in calibration data")


def _remote_guard() -> None:
    from apexpinn.train.cloud_checkpoint import require_colab_gpu

    require_colab_gpu()


def fit_calibrator(rows: list[dict], plan: dict, method: dict) -> dict:
    """Estimate four scales on hosted Colab, from independent calibration circuits only."""
    _remote_guard()
    spec = method["calibrator"]
    validate_calibration_rows(
        rows, plan, horizons=method["horizons"], minimum_events=spec["minimum_events_per_horizon"]
    )
    factors, populations = {}, {}
    for h in method["horizons"]:
        group = [r for r in rows if r["horizon"] == h]
        scores = []
        for row in group:
            lo, hi = central_interval(
                row["member_means_s"], row["member_sigma_s"], level=spec["target_level"]
            )
            mu, target = row["y_pred"], row["y_true"]
            if not lo < mu < hi:
                raise ValueError("raw central interval must bracket the mixture mean")
            score = (mu - target) / (mu - lo) if target < mu else (target - mu) / (hi - mu)
            if not np.isfinite(score):
                raise ValueError("nonfinite interval ratio")
            scores.append(score)
        factor = max(
            weighted_quantile(scores, event_weights([r["event_id"] for r in group]), spec["target_level"]),
            spec["factor_floor"],
        )
        factors[str(h)] = factor
        populations[str(h)] = {"origins": len(group), "events": sorted({r["event_id"] for r in group})}
    return {
        "schema": "apexpinn.mixture_recalibrator/v1",
        "method_id": method["method_id"],
        "factors": factors,
        "populations": populations,
        "plan": plan,
        "model_hash": rows[0]["instrument"]["model_hash"],
        "phi": rows[0]["phi"],
        "normalization_numeric_sha256": rows[0]["instrument"]["normalization_numeric_sha256"],
        "calibration_rows_sha256": identity(rows),
        "guarantee": "none under dependent origins or circuit shift",
    }


def rescale_row(row: dict, factor: float, *, recipe_sha256: str) -> dict:
    """Scale a specified distribution around its unchanged mean; never fit a scale here."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be finite and positive")
    means = np.asarray(row["member_means_s"], dtype=float)
    scales = np.asarray(row["member_sigma_s"], dtype=float)
    if (
        means.shape != (5,)
        or scales.shape != means.shape
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or not np.isclose(means.mean(), row["y_pred"], atol=1e-10, rtol=1e-10)
    ):
        raise ValueError("a valid five-member mixture with positive scales is required")
    variance = float(np.mean(scales**2) + np.var(means))
    if not np.isclose(variance, row["sigma_s"] ** 2, atol=1e-10, rtol=1e-10):
        raise ValueError("raw mixture variance disagrees with its components")
    result = copy.deepcopy(row)
    result["model"] = "M2/calibrated"
    result["conditions"]["model"] = result["model"]
    result["member_means_s"] = (row["y_pred"] + factor * (means - row["y_pred"])).tolist()
    result["member_sigma_s"] = (factor * scales).tolist()
    result["sigma_s"] = float(
        np.sqrt(np.mean(np.square(result["member_sigma_s"])) + np.var(result["member_means_s"]))
    )
    result["instrument"]["calibrator_sha256"] = recipe_sha256
    if not np.isfinite(result["sigma_s"]) or not np.isclose(
        result["sigma_s"] ** 2, factor**2 * variance, rtol=1e-10, atol=1e-10
    ):
        raise ValueError("calibrated mixture is nonfinite or violates the affine variance identity")
    return result


def apply_calibrator(rows: list[dict], recipe: dict) -> list[dict]:
    recipe_sha = identity(recipe)
    result = []
    for row in rows:
        if (
            row["fold"] != recipe["plan"]["outer_fold"]
            or row["conditions"].get("data_role") != "outer_test"
            or row["instrument"]["model_hash"] != recipe["model_hash"]
            or row["instrument"]["normalization_numeric_sha256"] != recipe["normalization_numeric_sha256"]
            or row["phi"] != recipe["phi"]
        ):
            raise ValueError("recipe is bound to a different test population, ensemble or normalization")
        result.append(rescale_row(row, recipe["factors"][str(row["horizon"])], recipe_sha256=recipe_sha))
    return result
