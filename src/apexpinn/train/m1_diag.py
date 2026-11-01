"""Exploratory M1 diagnostics (PLAN T5 method declarations, `M1` exploratory diagnostics).

The runs separate three candidate causes of the v1 M1 failure (k_c collapsed on 10 of 10 main fits): a
longer budget, the physics weight, and a bounded latent h = tanh(tau * n_h). Each variant is the v1 M1
setup of full_config with one change, at seed 0 (seed 1 optional) and noise 0.05 s. The runs are
exploratory: they are not S1b evidence, they change no v1 or v2 result or label, and no M1 method change
is declared from them. Records go to experiments/s1_m1_diag/<variant>/, one folder per variant, so they
never mix with the S1 records in experiments/s1/; the report is reports/s1_m1_diag.md
(apexpinn.eval.m1_diag_report).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from apexpinn.train.s1_study import (
    DEFAULT_DECISIONS,
    Log,
    RunSpec,
    Runner,
    StudyConfig,
    full_config,
    run_study,
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
}


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
        try:
            summaries[name] = run_study(
                cfg,
                part="main",
                force=force,
                runner=runner,
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
