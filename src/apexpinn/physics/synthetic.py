"""Synthetic stints with known physics parameters, for identifiability studies (PRD S1, S4; PLAN T3).

Each stint follows the gauge-fixed model of PROJECT.md section 4.4: h(0) = w(0) = 0, the true ODE is
solved with Diffrax, and lap time comes from the shared observation equation plus Gaussian noise.

A Misspecification (or one of the named MISSPEC_VARIANTS) generates data from a slightly different process
than the fitted ODE, so the T5 identifiability results can be checked under misfit. The default is the
well-specified model, and its output does not change when misspecification support is unused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from apexpinn.physics.residuals import COMPOUNDS, PhysicsParams, observation, thermal_rhs, wear_rhs

MIN_LAPS = 15
MAX_LAPS = 35
DEFAULT_N_STINTS = 48
DEFAULT_COMPOUNDS = ("C2", "C3", "C4")
DEFAULT_NOISE_S = 0.05  # first O6 noise level
Q_LEVEL_RANGE = (0.45, 0.85)  # circuit-level heat load, uniform per stint
Q_LAP_SD = 0.03  # small per-lap variation around the circuit level
B_RANGE_S = (80.0, 100.0)  # stint offset b_s, uniform per stint
RACE_LENGTH = 60  # L_race; each stint gets a start race lap in [2, RACE_LENGTH - n_laps + 1]
TRAFFIC_MIN_START = 3  # earliest stint lap of a traffic step


@dataclass(frozen=True)
class Misspecification:
    """How the synthetic data departs from the fitted gauge-fixed ODE. The defaults are well specified.

    - phi_true, phi_used: the lap time carries a race-lap trend phi_true * (RACE_LENGTH - race_lap),
      and the target is corrected with phi_used (PROJECT section 4.2). The leftover term rises by
      phi_used - phi_true per stint lap, so a fitted rho_c is expected to shift by about that amount
      (PROJECT L3). The shift is exact for the base wear rate and approximate under overheating wear.
    - noise_df: Student-t observation noise with this many degrees of freedom (> 2), scaled so its
      standard deviation is noise_s. None keeps Gaussian noise.
    - wear_slope_ratio: replace the wear ODE by piecewise-linear wear whose slope changes by this factor
      at mid-stint (tau = n_laps / 2). The two slopes are set so that the stint-mean slope
      w(n_laps) / n_laps equals rho_c. That mean slope is the truth target, because no exact ODE rho_c
      exists for this data. The thermal state h still follows the ODE. None keeps ODE wear.
    - drop_first_laps: stint laps 1..drop_first_laps (0, 1, or 2) are unscored in the mask, as the real
      out-lap filter does (PROJECT section 4.5 and L1). The default 0 scores lap 1, where warm-up is
      fastest. True states and targets of dropped laps are still stored.
    - traffic_step_s: (low, high) range of a step offset in seconds, drawn uniformly per affected stint
      and added to traffic_laps consecutive laps. round(traffic_stint_fraction * n_stints) stints are
      affected. None adds no traffic. The truth stays theta_true.
    """

    phi_true: float = 0.0
    phi_used: float = 0.0
    noise_df: float | None = None
    wear_slope_ratio: float | None = None
    drop_first_laps: int = 0
    traffic_step_s: tuple[float, float] | None = None
    traffic_stint_fraction: float = 0.25
    traffic_laps: int = 3

    def __post_init__(self) -> None:
        if not (np.isfinite(self.phi_true) and np.isfinite(self.phi_used)):
            raise ValueError(
                f"phi_true and phi_used must be finite, got {self.phi_true!r} and {self.phi_used!r}"
            )
        if self.noise_df is not None and not (np.isfinite(self.noise_df) and self.noise_df > 2):
            raise ValueError(f"noise_df must be finite and > 2 (finite variance), got {self.noise_df!r}")
        if self.wear_slope_ratio is not None and not (
            np.isfinite(self.wear_slope_ratio) and self.wear_slope_ratio > 0
        ):
            raise ValueError(f"wear_slope_ratio must be finite and > 0, got {self.wear_slope_ratio!r}")
        if self.drop_first_laps not in (0, 1, 2):
            raise ValueError(f"drop_first_laps must be 0, 1, or 2, got {self.drop_first_laps!r}")
        if self.traffic_step_s is not None:
            low, high = self.traffic_step_s
            if not (np.isfinite(low) and np.isfinite(high) and 0 <= low <= high):
                raise ValueError(
                    f"traffic_step_s must be (low, high) with 0 <= low <= high, got {self.traffic_step_s!r}"
                )
        if not 0.0 <= self.traffic_stint_fraction <= 1.0:
            raise ValueError(f"traffic_stint_fraction must be in [0, 1], got {self.traffic_stint_fraction!r}")
        if not 1 <= self.traffic_laps <= MIN_LAPS - TRAFFIC_MIN_START:
            raise ValueError(
                f"traffic_laps must be in [1, {MIN_LAPS - TRAFFIC_MIN_START}], got {self.traffic_laps!r}"
            )

    @property
    def rho_shift(self) -> float:
        """Expected shift of a fitted rho_c caused by the wrong race-lap trend: phi_used - phi_true."""
        return self.phi_used - self.phi_true


WELL_SPECIFIED = Misspecification()
MISSPEC_VARIANTS: dict[str, Misspecification] = {
    "well_specified": WELL_SPECIFIED,
    "wrong_phi": Misspecification(phi_true=0.05, phi_used=0.06),
    "student_t": Misspecification(noise_df=5.0),
    "piecewise_wear": Misspecification(wear_slope_ratio=3.0),
    "drop_early": Misspecification(drop_first_laps=1),
    "traffic": Misspecification(traffic_step_s=(0.3, 0.5)),
}


def default_theta_true() -> PhysicsParams:
    """Ground-truth Theta for C1..C5: softer compounds warm up faster, wear faster, and peak at lower h.

    With these values most stints have their fastest lap between stint laps 2 and 8, more than 0.1 s faster
    than stint lap 1 (the early dip of S4).
    """
    return PhysicsParams.from_constrained(
        k=[0.22, 0.28, 0.35, 0.43, 0.52],
        rho=[0.035, 0.045, 0.06, 0.08, 0.10],
        gamma=[3.5, 3.2, 3.0, 2.8, 2.5],
        h_opt=[0.68, 0.64, 0.60, 0.57, 0.55],
        kappa=1.0,
    )


class SyntheticDataset(eqx.Module):
    """A batch of synthetic stints, padded to MAX_LAPS. Row s is stint s.

    Lap arrays have shape (n_stints, MAX_LAPS). Column k - 1 holds stint lap k (tau = k).
    Entries past n_laps[s] are padding set to 0, so always select with mask. The mask can also exclude
    early laps (Misspecification.drop_first_laps), whose stored values are real, not padding.
    """

    compound_idx: jax.Array  # (n_stints,) int, index into COMPOUNDS
    n_laps: jax.Array  # (n_stints,) int, in [MIN_LAPS, MAX_LAPS]
    mask: jax.Array  # (n_stints, MAX_LAPS) bool, True for scored laps drop_first_laps < k <= n_laps
    tau: jax.Array  # (MAX_LAPS,) float, 1..MAX_LAPS
    q: jax.Array  # (n_stints, MAX_LAPS + 1) heat load at tau = 0..MAX_LAPS, linear in between
    h: jax.Array  # true thermal state
    w: jax.Array  # true cumulative wear, in seconds (piecewise linear under wear_slope_ratio)
    b: jax.Array  # (n_stints,) stint offset b_s, in seconds
    race_lap: jax.Array  # (n_stints, MAX_LAPS) int race lap of each stint lap, in [2, RACE_LENGTH]
    traffic: jax.Array  # (n_stints, MAX_LAPS) traffic step offset included in y_clean, in seconds
    y_clean: jax.Array  # noise-free target, including any misspecification terms, in seconds
    y: jax.Array  # y_clean + noise (Gaussian, or Student-t under noise_df), std noise_s
    rho_target: jax.Array  # (N_COMPOUNDS,) rho_c a fit should recover: theta_true.rho + misspec.rho_shift
    theta_true: PhysicsParams
    noise_s: float
    misspec: Misspecification = eqx.field(static=True)


def _solve_stint(
    params: PhysicsParams, compound_idx: jax.Array, q_knots: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Solve the true thermal and wear ODEs on tau in [0, MAX_LAPS] and return (h, w) at tau = 1..MAX_LAPS.

    q has a slope change at every knot, so the interior knots are jump times: no Tsit5 step spans one.
    Steps across kinks make the embedded error estimate unreliable. Without jump times, this solve and
    ODEModel.solve disagreed by up to 3.5e-4 in h at rtol = atol = 1e-6.
    """
    knots = jnp.arange(MAX_LAPS + 1, dtype=q_knots.dtype)
    q_of_tau = diffrax.LinearInterpolation(ts=knots, ys=q_knots)

    def vector_field(t, y, args):
        h, _ = y
        dh = thermal_rhs(params, compound_idx, h, q_of_tau.evaluate(t))
        return dh, wear_rhs(params, compound_idx, h)

    zero = jnp.zeros((), dtype=q_knots.dtype)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=knots[0],
        t1=knots[-1],
        dt0=0.1,
        y0=(zero, zero),
        saveat=diffrax.SaveAt(ts=knots[1:]),
        stepsize_controller=diffrax.ClipStepSizeController(
            diffrax.PIDController(rtol=1e-6, atol=1e-6), jump_ts=knots[1:-1]
        ),
        max_steps=8192,
    )
    h, w = sol.ys
    return h, w


_solve_stints = eqx.filter_jit(jax.vmap(_solve_stint, in_axes=(None, 0, 0)))


def _compound_indices(compounds: Sequence[str]) -> np.ndarray:
    compounds = tuple(compounds)
    if not compounds:
        raise ValueError("compounds must not be empty")
    unknown = [c for c in compounds if c not in COMPOUNDS]
    if unknown:
        raise ValueError(f"unknown compounds {unknown}, expected names from {COMPOUNDS}")
    return np.array([COMPOUNDS.index(c) for c in compounds])


def _resolve_misspec(misspec: Misspecification | str | None) -> Misspecification:
    if misspec is None:
        return WELL_SPECIFIED
    if isinstance(misspec, str):
        if misspec not in MISSPEC_VARIANTS:
            raise ValueError(
                f"unknown misspec variant {misspec!r}, expected one of {sorted(MISSPEC_VARIANTS)}"
            )
        return MISSPEC_VARIANTS[misspec]
    if not isinstance(misspec, Misspecification):
        raise ValueError(f"misspec must be a Misspecification, a variant name, or None, got {misspec!r}")
    return misspec


def _piecewise_wear(rho: jax.Array, n_laps: jax.Array, tau: jax.Array, ratio: float) -> jax.Array:
    """Wear with slope s1 up to tau = n_laps / 2 and ratio * s1 after it, where w(n_laps) / n_laps = rho."""
    slope1 = 2.0 * rho / (1.0 + ratio)
    tau_break = n_laps.astype(tau.dtype) / 2.0
    before = jnp.minimum(tau[None, :], tau_break[:, None])
    after = jnp.maximum(tau[None, :] - tau_break[:, None], 0.0)
    return slope1[:, None] * (before + ratio * after)


def make_synthetic_dataset(
    key: jax.Array,
    n_stints: int = DEFAULT_N_STINTS,
    compounds: Sequence[str] = DEFAULT_COMPOUNDS,
    theta_true: PhysicsParams | None = None,
    noise_s: float = DEFAULT_NOISE_S,
    misspec: Misspecification | str | None = None,
) -> SyntheticDataset:
    """Draw n_stints synthetic stints. The same key always gives the same dataset.

    Per stint: length uniform in [MIN_LAPS, MAX_LAPS], a compound (balanced over `compounds`, then
    shuffled), a circuit-level q plus per-lap variation clipped to [0, 1], an offset b_s, and a start race
    lap. theta_true defaults to default_theta_true(). noise_s is the observation noise standard deviation in
    seconds.

    misspec is a Misspecification, a name from MISSPEC_VARIANTS, or None (well specified). The extra draws
    for race laps and traffic come from jax.random.fold_in(key, 1), so the well-specified draws are the
    same with or without misspecification. By default every lap 1..n_laps is scored, including the out-lap
    that real filters drop; use drop_first_laps to exclude it.
    """
    if isinstance(n_stints, bool) or not isinstance(n_stints, int | np.integer) or n_stints < 1:
        raise ValueError(f"n_stints must be a positive integer, got {n_stints!r}")
    if not np.isfinite(noise_s) or noise_s < 0:
        raise ValueError(f"noise_s must be finite and >= 0, got {noise_s!r}")
    choices = _compound_indices(compounds)
    params = default_theta_true() if theta_true is None else theta_true
    spec = _resolve_misspec(misspec)

    k_len, k_comp, k_level, k_lap, k_b, k_noise = jax.random.split(key, 6)
    k_race, k_t_pick, k_t_start, k_t_size = jax.random.split(jax.random.fold_in(key, 1), 4)
    n_laps = jax.random.randint(k_len, (n_stints,), MIN_LAPS, MAX_LAPS + 1)
    balanced = jnp.asarray(choices[np.arange(n_stints) % len(choices)])
    compound_idx = jax.random.permutation(k_comp, balanced)

    q_level = jax.random.uniform(k_level, (n_stints, 1), minval=Q_LEVEL_RANGE[0], maxval=Q_LEVEL_RANGE[1])
    q = jnp.clip(q_level + Q_LAP_SD * jax.random.normal(k_lap, (n_stints, MAX_LAPS + 1)), 0.0, 1.0)
    b = jax.random.uniform(k_b, (n_stints,), minval=B_RANGE_S[0], maxval=B_RANGE_S[1])
    tau = jnp.arange(1, MAX_LAPS + 1, dtype=q.dtype)
    valid = tau[None, :] <= n_laps[:, None]
    mask = valid & (tau[None, :] > spec.drop_first_laps)

    # Race laps: the stint starts after race lap 1 and ends by RACE_LENGTH.
    race_start = jax.random.randint(k_race, (n_stints,), 2, RACE_LENGTH - n_laps + 2)
    race_lap = race_start[:, None] + jnp.arange(MAX_LAPS)[None, :]

    h, w = _solve_stints(params, compound_idx, q)
    if spec.wear_slope_ratio is not None:
        w = _piecewise_wear(params.rho[compound_idx], n_laps, tau, spec.wear_slope_ratio)
    y_clean = observation(params, compound_idx[:, None], h, w, b[:, None])

    traffic = jnp.zeros_like(y_clean)
    if spec.traffic_step_s is not None:
        n_hit = int(round(spec.traffic_stint_fraction * n_stints))
        hit_rows = jax.random.permutation(k_t_pick, n_stints)[:n_hit]
        hit = jnp.zeros(n_stints, dtype=bool).at[hit_rows].set(True)
        last_start = n_laps - spec.traffic_laps  # the step ends before the last lap
        start = jax.random.randint(k_t_start, (n_stints,), TRAFFIC_MIN_START, last_start + 1)
        low, high = spec.traffic_step_s
        size = jax.random.uniform(k_t_size, (n_stints,), dtype=y_clean.dtype, minval=low, maxval=high)
        in_step = (tau[None, :] >= start[:, None]) & (tau[None, :] < start[:, None] + spec.traffic_laps)
        traffic = jnp.where(hit[:, None] & in_step, size[:, None], 0.0)
        y_clean = y_clean + traffic

    # Lap time carries phi_true * (L - race_lap); the target subtracts phi_used * (L - race_lap).
    if spec.phi_true != spec.phi_used:
        leftover = (spec.phi_true - spec.phi_used) * (RACE_LENGTH - race_lap)
        y_clean = y_clean + leftover.astype(y_clean.dtype)

    if spec.noise_df is None:
        unit_noise = jax.random.normal(k_noise, y_clean.shape, dtype=y_clean.dtype)
    else:
        df = spec.noise_df
        t_draw = jax.random.t(k_noise, df, y_clean.shape, dtype=y_clean.dtype)
        unit_noise = t_draw * np.sqrt((df - 2.0) / df)  # Student-t variance is df / (df - 2)
    y = y_clean + noise_s * unit_noise

    def pad(x: jax.Array) -> jax.Array:
        return jnp.where(valid, x, 0.0)

    return SyntheticDataset(
        compound_idx=compound_idx,
        n_laps=n_laps,
        mask=mask,
        tau=tau,
        q=q,
        h=pad(h),
        w=pad(w),
        b=b,
        race_lap=jnp.where(valid, race_lap, 0),
        traffic=pad(traffic),
        y_clean=pad(y_clean),
        y=pad(y),
        rho_target=params.rho + spec.rho_shift,
        theta_true=params,
        noise_s=float(noise_s),
        misspec=spec,
    )
