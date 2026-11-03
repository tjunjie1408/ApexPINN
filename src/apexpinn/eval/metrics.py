"""Point errors and exact univariate Gaussian-mixture scores. No fitting or implicit row filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri


def _paired_vectors(target: Any, prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    y, predicted = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or predicted.shape != y.shape:
        raise ValueError("target and prediction must be equal nonempty 1-D arrays")
    if not np.isfinite(y).all() or not np.isfinite(predicted).all():
        raise ValueError("target and prediction must be finite; select the common scored population first")
    return y, predicted


def rmse(target: Any, prediction: Any) -> float:
    y, predicted = _paired_vectors(target, prediction)
    return float(np.sqrt(np.mean((predicted - y) ** 2)))


def mae(target: Any, prediction: Any) -> float:
    y, predicted = _paired_vectors(target, prediction)
    return float(np.mean(np.abs(predicted - y)))


def _mixture(means: Any, scales: Any, weights: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, scales = np.asarray(means, dtype=np.float64), np.asarray(scales, dtype=np.float64)
    if means.ndim != 1 or means.size == 0 or scales.shape != means.shape:
        raise ValueError("means and scales must be equal nonempty 1-D arrays")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("means/scales must be finite; scales must be nonnegative")
    weights = (
        np.full(means.size, 1 / means.size) if weights is None else np.asarray(weights, dtype=np.float64)
    )
    if (
        weights.shape != means.shape
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("weights must be nonnegative, match the members and sum to one")
    keep = weights > 0
    return means[keep], scales[keep], weights[keep] / weights[keep].sum()


def _normal_absolute_mean(delta: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """E|N(delta, scale^2)|, including point masses at scale=0."""
    z = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
    value = scale * np.sqrt(2 / np.pi) * np.exp(-0.5 * z**2) + delta * (2 * ndtr(z) - 1)
    return np.where(scale > 0, value, np.abs(delta))


def crps_gaussian_mixture(target: float, means: Any, scales: Any, *, weights: Any = None) -> float:
    """E|X-y| - E|X-X'|/2 for independent draws from the explicit mixture, in target units."""
    means, scales, weights = _mixture(means, scales, weights)
    if not np.isfinite(target):
        raise ValueError("target must be finite")
    first = weights @ _normal_absolute_mean(means - target, scales)
    differences = means[:, None] - means[None, :]
    pair_scale = np.hypot(scales[:, None], scales[None, :])
    second = weights @ _normal_absolute_mean(differences, pair_scale) @ weights
    value = float(first - 0.5 * second)
    if not np.isfinite(value):
        raise ValueError("CRPS overflowed for the supplied mixture")
    return max(value, 0.0)  # roundoff can leave a tiny negative value at a degenerate point mass


def mixture_quantile(probability: float, means: Any, scales: Any, *, weights: Any = None) -> float:
    """Generalized-inverse mixture CDF, including atoms; does not use a moment-matched Gaussian."""
    means, scales, weights = _mixture(means, scales, weights)
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("probability must be strictly between zero and one")
    component_quantiles = means + scales * ndtri(probability)
    lower, upper = float(component_quantiles.min()), float(component_quantiles.max())

    def cdf(value: float) -> float:
        z = np.divide(value - means, scales, out=np.zeros_like(means), where=scales > 0)
        return float(weights @ np.where(scales > 0, ndtr(z), value >= means))

    if cdf(lower) >= probability:
        return lower
    for atom in np.unique(means[scales == 0]):
        if cdf(np.nextafter(atom, -np.inf)) < probability <= cdf(float(atom)):
            return float(atom)
    for _ in range(128):
        midpoint = lower + (upper - lower) / 2
        if midpoint == lower or midpoint == upper:
            break
        if cdf(midpoint) >= probability:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def central_interval(means: Any, scales: Any, *, level: float, weights: Any = None) -> tuple[float, float]:
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be strictly between zero and one")
    tail = (1 - level) / 2
    return (
        mixture_quantile(tail, means, scales, weights=weights),
        mixture_quantile(1 - tail, means, scales, weights=weights),
    )


def interval_width(lower: Any, upper: Any) -> float:
    lower, upper = _paired_vectors(lower, upper)
    if np.any(lower > upper):
        raise ValueError("interval lower bound exceeds upper bound")
    return float(np.mean(upper - lower))


def coverage(target: Any, lower: Any, upper: Any) -> float:
    y, lower = _paired_vectors(target, lower)
    _, upper = _paired_vectors(target, upper)
    interval_width(lower, upper)
    return float(np.mean((y >= lower) & (y <= upper)))
