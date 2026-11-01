"""Exploratory M1 diagnostics (PLAN T5 method declarations, `M1` exploratory diagnostics).

The runs test five candidate causes of the v1 M1 failure (k_c collapsed on 10 of 10 main fits, and biased
low in S1a with true states observed): (a) a longer budget, (b) the physics weight, (c) a bounded latent
h = tanh(tau * n_h), (d) a no-network least-squares check that fits k_c and rho_c directly from the true
ODE states and rates, and (e) network smoothing and collocation resolution (n_colloc 128 or m1_width 128).
Each variant is the v1 M1 setup of full_config with one change, at seed 0 (seed 1 optional) and noise
0.05 s. The runs are exploratory: they are not S1b evidence, they change no v1 or v2 result or label, and
no M1 method change is declared from them. Records go to experiments/s1_m1_diag/<variant>/, one folder per
variant, so they never mix with the S1 records in experiments/s1/; the report is reports/s1_m1_diag.md
(apexpinn.eval.m1_diag_report).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from apexpinn.models.ode_model import ODEModel
from apexpinn.models.stints import batch_from_synthetic
from apexpinn.physics.diagnostics import detect_collapse, residual_rms, wear_violation_rate
from apexpinn.physics.residuals import (
    COMPOUNDS,
    PhysicsParams,
    states_and_rates,
    thermal_residual,
    wear_residual,
)
from apexpinn.train.s1_study import (
    DEFAULT_DECISIONS,
    Log,
    Runner,
    RunSpec,
    StudyConfig,
    build_fit_record,
    colloc_key,
    dataset_for,
    full_config,
    run_study,
    sample_collocation,
    theta_values,
    variant_truth,
)

DIAG_OUT_DIR = "experiments/s1_m1_diag"
DIAG_REPORT = "reports/s1_m1_diag.md"
DIAG_NOISE = 0.05
DEFAULT_SEEDS = (0,)
V1_OUT_DIR = "experiments/s1"  # the frozen S1b v1 records, read as the baseline row of the report


def _budget_20k(cfg: StudyConfig) -> StudyConfig:
    return replace(cfg, m1_train=replace(cfg.m1_train, n_steps=20000, log_every=1000))


def _physics_off(cfg: StudyConfig) -> StudyConfig:
    return replace(cfg, m1_weights=replace(cfg.m1_weights, thermal=0.0, wear=0.0, mono=0.0))


def _physics_x10(cfg: StudyConfig) -> StudyConfig:
    return replace(cfg, m1_weights=replace(cfg.m1_weights, thermal=10.0, wear=10.0))


def _bounded_h(cfg: StudyConfig) -> StudyConfig:
    return replace(cfg, m1_latent="tanh")


def _true_state_ls(cfg: StudyConfig) -> StudyConfig:
    return replace(cfg, m1_train=replace(cfg.m1_train, n_steps=1, log_every=1))


def _colloc_128(cfg: StudyConfig) -> StudyConfig:
    return replace(cfg, m1_train=replace(cfg.m1_train, n_colloc=128))


def _width_128(cfg: StudyConfig) -> StudyConfig:
    return replace(cfg, m1_width=128)


VARIANTS: dict[str, tuple[str, Callable[[StudyConfig], StudyConfig]]] = {
    "budget_20k": (
        "cause (a): 20000 training steps instead of 5000 (log_every 1000; n_warm 200 and n_ramp 800 "
        "unchanged)",
        _budget_20k,
    ),
    "physics_off": (
        "cause (b): thermal, wear and mono weights 0, so only the data and reg terms train; k_c, rho_c "
        "and kappa then get no gradient and stay at the prior draw",
        _physics_off,
    ),
    "physics_x10": ("cause (b): thermal and wear weights 10 instead of 1", _physics_x10),
    "bounded_h": ("cause (c): bounded latent h = tanh(tau * n_h) (PINN latent_mode 'tanh')", _bounded_h),
    "true_state_ls": (
        "cause (d): no-network least-squares check fitting k_c and rho_c directly from the true ODE "
        "states and autodiff rates at collocation points",
        _true_state_ls,
    ),
    "colloc_128": ("cause (e): 128 collocation points per stint instead of 32", _colloc_128),
    "width_128": ("cause (e): MLP width 128 instead of 32", _width_128),
}


def run_true_state_ls(cfg: StudyConfig, spec: RunSpec) -> dict[str, Any]:
    """Fit k_c and rho_c by closed-form least squares on the true ODE states and autodiff rates.

    At the fixed collocation sample (colloc_key(seed)), states_and_rates of ODEModel.dense_latent at
    theta_true gives (h, w, dh/dtau, dw/dtau). Minimizing sum((dh - k_c (q - h))^2) and
    sum((dw - rho_c (1 + kappa ReLU(h - h_opt_c)))^2) per compound gives the exact least-squares k_c and
    rho_c with no network.
    """
    t0 = time.perf_counter()
    ds = dataset_for(cfg, spec)
    batch = batch_from_synthetic(ds)
    theta = ds.theta_true
    ode = ODEModel.init(physics=theta, rtol=cfg.ode_rtol, atol=cfg.ode_atol)
    colloc = sample_collocation(
        colloc_key(spec.seed), batch.n_laps, cfg.m1_train.n_colloc, cfg.m1_train.colloc_extension
    ).astype(batch.tau.dtype)
    knots = jnp.arange(batch.q.shape[1], dtype=batch.tau.dtype)

    def per_stint(c, q_knots, t_col, n_s):
        h, _, dh, dw = states_and_rates(ode.dense_latent(c, q_knots), t_col)
        q_c = jnp.interp(jnp.minimum(t_col, n_s.astype(t_col.dtype)), knots, q_knots)
        return h, dh, dw, q_c

    h, dh, dw, q_c = jax.vmap(per_stint)(batch.compound_idx, batch.q, colloc, batch.n_laps)
    h_np, dh_np, dw_np, q_np = (np.asarray(a, dtype=np.float64) for a in (h, dh, dw, q_c))
    c_np = np.asarray(batch.compound_idx)
    k_hat = np.asarray(theta.k, dtype=np.float64).copy()
    rho_hat = np.asarray(theta.rho, dtype=np.float64).copy()
    h_opt_np = np.asarray(theta.h_opt, dtype=np.float64)
    kappa_val = float(np.asarray(theta.kappa, dtype=np.float64))
    for cname in cfg.compounds:
        idx = COMPOUNDS.index(cname)
        rows = c_np == idx
        if not np.any(rows):
            continue
        x_t = (q_np[rows] - h_np[rows]).ravel()
        y_t = dh_np[rows].ravel()
        denom_t = float(np.dot(x_t, x_t))
        if denom_t > 0:
            k_hat[idx] = float(np.dot(x_t, y_t) / denom_t)
        over = np.maximum(h_np[rows] - h_opt_np[idx], 0.0)
        x_w = (1.0 + kappa_val * over).ravel()
        y_w = dw_np[rows].ravel()
        denom_w = float(np.dot(x_w, x_w))
        if denom_w > 0:
            rho_hat[idx] = float(np.dot(x_w, y_w) / denom_w)

    fitted_phys = PhysicsParams.from_constrained(
        k=k_hat, rho=rho_hat, gamma=np.asarray(theta.gamma), h_opt=h_opt_np, kappa=kappa_val
    )
    r_t = jax.vmap(thermal_residual, in_axes=(None, 0, 0, 0, 0))(fitted_phys, batch.compound_idx, h, dh, q_c)
    r_w = jax.vmap(wear_residual, in_axes=(None, 0, 0, 0))(fitted_phys, batch.compound_idx, h, dw)
    resid = jnp.where(batch.mask, batch.y - ds.y_clean, 0.0)
    n_scored = int(np.asarray(batch.mask).sum())
    rms = float(np.sqrt(float(jnp.sum(resid**2)) / max(n_scored, 1)))
    flagged = detect_collapse(fitted_phys).flagged
    diagnostics = {
        "scored_residual_rms_s": rms,
        "residual_ratio": rms / spec.noise_s if spec.noise_s > 0 else None,
        "collapse_flags": [f for f in flagged if any(f.endswith(f"[{c}]") for c in cfg.compounds)],
        "thermal_residual_rms": float(residual_rms(r_t)),
        "wear_residual_rms": float(residual_rms(r_w)),
        "wear_violation_rate": float(wear_violation_rate(dw)),
        "n_colloc_points": int(colloc.size),
    }
    elapsed = time.perf_counter() - t0
    return build_fit_record(
        cfg,
        spec,
        estimates=theta_values(fitted_phys, cfg.compounds),
        truth=variant_truth(ds, cfg.compounds),
        n_scored=n_scored,
        diagnostics=diagnostics,
        convergence={"steps": [0], "objective": [0.0], "normalized": {}, "objective_rel_change_last20": None},
        wall_time_s={"fit": elapsed, "total": elapsed},
    )


def _check_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    out = tuple(seeds)
    if not out or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in out):
        raise ValueError(f"seeds must be non-negative integers, got {seeds!r}")
    return tuple(sorted(set(out)))


def base_config(decisions: str | Path = DEFAULT_DECISIONS) -> StudyConfig:
    """full_config: the v1 M1 setup every variant changes in one place (and the baseline row's config)."""
    return full_config(decisions)


def variant_config(
    name: str, seeds: Sequence[int] = DEFAULT_SEEDS, decisions: str | Path = DEFAULT_DECISIONS
) -> StudyConfig:
    """StudyConfig of one diagnostic variant: full_config with the variant's change, M1 only, noise 0.05 s.

    Records go to experiments/s1_m1_diag/<name>/.
    """
    if name not in VARIANTS:
        raise ValueError(f"variant must be one of {tuple(VARIANTS)}, got {name!r}")
    cfg = replace(
        base_config(decisions),
        models_main=("M1",),
        noise_levels=(DIAG_NOISE,),
        seeds_main=_check_seeds(seeds),
        out_dir=f"{DIAG_OUT_DIR}/{name}",
        report_path=DIAG_REPORT,
        name=f"m1_diag_{name}",
    )
    return VARIANTS[name][1](cfg)


def diag_specs(cfg: StudyConfig) -> list[RunSpec]:
    """The M1 main fits of a variant config: main_fit_M1_well_specified_noise0.05_seed<s> per seed."""
    return [RunSpec("main", "fit", "M1", "well_specified", DIAG_NOISE, s) for s in cfg.seeds_main]


def summary_line(name: str, summary: Mapping[str, Any]) -> str:
    """One console line for a run_diag summary entry (a run_study summary or a refusal)."""
    if "refused" in summary:
        return f"{name}: stopped: {summary['refused']}"
    return (
        f"{name}: planned {summary['planned']}, run {summary['run']}, skipped {summary['skipped']}, "
        f"errors {summary['errors']}; wall time {summary['wall_time_s']:.1f} s"
    )


def run_diag(
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    root: str | Path = ".",
    force: bool = False,
    log: Log = print,
    runner: Runner | None = None,
    decisions: str | Path = DEFAULT_DECISIONS,
) -> dict[str, dict[str, Any]]:
    """Run the M1 fits of each variant (all if None) with run_study; variant name -> summary.

    Only the M1 fit runs are run (run_study part 'main' with only), never the profiles. A finished record
    with the same config_hash is skipped. When run_study refuses a variant (ValueError, for example a
    finished record whose config_hash differs and force=False), that variant's summary is
    {'refused': message} and the remaining variants still run. Each summary is logged with summary_line
    as soon as its variant finishes. Unknown variant names raise ValueError before anything runs.
    """
    names = list(VARIANTS) if variants is None else list(dict.fromkeys(variants))
    unknown = [n for n in names if n not in VARIANTS]
    if unknown or not names:
        raise ValueError(f"variants must be names from {tuple(VARIANTS)}, got {variants!r}")
    configs = {name: variant_config(name, seeds, decisions) for name in names}
    summaries: dict[str, dict[str, Any]] = {}
    for name, cfg in configs.items():
        log(f"variant {name}: {VARIANTS[name][0]}")
        var_runner = (run_true_state_ls if name == "true_state_ls" else None) if runner is None else runner
        try:
            summaries[name] = run_study(
                cfg,
                part="main",
                force=force,
                runner=var_runner,
                root=root,
                log=log,
                only=[s.run_id for s in diag_specs(cfg)],
            )
        except ValueError as exc:
            summaries[name] = {"refused": str(exc)}
        log(summary_line(name, summaries[name]))
    return summaries


def variant_configs(
    seeds: Sequence[int] = DEFAULT_SEEDS, decisions: str | Path = DEFAULT_DECISIONS
) -> Mapping[str, StudyConfig]:
    """variant name -> variant_config, in VARIANTS order."""
    return {name: variant_config(name, seeds, decisions) for name in VARIANTS}
