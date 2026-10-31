"""Physics diagnostics for M1/M2: residual RMS, wear violation rate, and parameter-collapse detection.

See PRD.md section 6 (physics metrics) and GLOSSARY.md (wear violation rate, parameter collapse).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from apexpinn.physics.residuals import COMPOUNDS, POSITIVITY_FLOOR, PhysicsParams

VIOLATION_TOL = 1e-6


def residual_rms(residual: ArrayLike) -> jax.Array:
    """Root mean square of a residual over collocation points."""
    r = jnp.asarray(residual)
    return jnp.sqrt(jnp.mean(r**2))


def wear_violation_rate(dw: ArrayLike, tol: float = VIOLATION_TOL) -> jax.Array:
    """Share of collocation points where dw/dtau < -tol."""
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    dw = jnp.asarray(dw)
    if dw.size == 0:
        raise ValueError("dw must contain at least one collocation point")
    return jnp.mean(dw < -tol)


@dataclass(frozen=True)
class CollapseReport:
    """Positive rates within rel_tol of POSITIVITY_FLOOR, named like 'k_c[C3]' or 'rho_c[C1]'."""

    flagged: tuple[str, ...]
    rel_tol: float

    @property
    def collapsed(self) -> bool:
        return bool(self.flagged)


def detect_collapse(params: PhysicsParams, rel_tol: float = 0.01) -> CollapseReport:
    """Flag any k_c or rho_c with value <= POSITIVITY_FLOOR * (1 + rel_tol).

    Collapse is reported, not prevented (GLOSSARY: parameter collapse). The caller decides whether the data
    loss is high enough for the flag to mean collapse.
    """
    if rel_tol < 0:
        raise ValueError(f"rel_tol must be non-negative, got {rel_tol}")
    threshold = POSITIVITY_FLOOR * (1.0 + rel_tol)
    flagged = []
    for name, values in (("k_c", params.k), ("rho_c", params.rho)):
        arr = np.asarray(values)
        flagged.extend(f"{name}[{COMPOUNDS[i]}]" for i in np.flatnonzero(arr <= threshold))
    return CollapseReport(flagged=tuple(flagged), rel_tol=rel_tol)
