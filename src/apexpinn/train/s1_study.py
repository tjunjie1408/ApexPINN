"""S1 identifiability study runner: S1b recovery fits, S1c profiles and spectra, S1d misspecification.

See PRD.md section 6.4 (S1b, S1c, S1d), PLAN.md T5 (study run, misspecification study, pass criteria) and
the T5 'Method declarations' block, which this module implements:

- S1b: M1 (PINN) and M2 (ODE model) are trained on lap time only from a prior draw, never from the truth.
- S1c: profiles and the Gauss-Newton spectrum use the exact ODE likelihood with b_s concentrated out
  (physics.identifiability.ode_least_squares), per dataset, in log / logit coordinates. The MLE and the
  profile solves keep the free coordinates inside identifiability.S1C_BOUNDS, and a failed ODE solve
  gives a failed point instead of stopping the run (PROFILE_METHOD_VERSION 2).
- Keys: the data key is jax.random.key(seed), shared by every noise level and variant of a seed (paired
  design); the prior initialization draw comes from fold_in(key, 101) and is shared by M1, M2 and the
  first profile start of that seed.
- M2 joins the misspecification study only if the slowest main-study M2 fit took under 600 s. While any
  main M2 fit record is missing or failed, M2 is undecided: part misspec runs the M1 and profile runs and
  adds the M2 runs on a later call, once every main M2 fit has finished.

Every run writes one JSON record <out_dir>/<run_id>.json (atomic: tmp file plus os.replace). A record
stores the config hash of the settings the run used (including a digest of the synthetic generator and
residual equation sources, the prior ranges and the key fold offsets; for profile runs also the optimizer
box, the ODE max_steps and the profile method version), so a rerun skips completed runs and refuses to
mix finished records from a changed config unless force=True. Error records never block and are rerun.
Runs can be split across machines with shards, and `only` restricts a call to named runs (for example to
time one full profile run).

Paths in StudyConfig are relative to the repository root; run_study and scripts/run_s1.py resolve them
against a root argument. Call jax.config.update('jax_enable_x64', True) before running: the configs train
in float64.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from apexpinn.contracts import MetricRecord, load_decisions
from apexpinn.models.ode_model import ODEModel
from apexpinn.models.pinn import PINN
from apexpinn.models.stints import StintBatch, batch_from_synthetic
from apexpinn.physics.diagnostics import detect_collapse, residual_rms, wear_violation_rate
from apexpinn.physics.identifiability import (
    PRIOR_RANGES,
    S1C_BOUNDS,
    Interval,
    ProfileResult,
    ThetaLayout,
    gauss_newton_spectrum,
    likelihood_interval,
    ode_least_squares,
    ode_mle,
    profile_likelihood,
    profile_shape_label,
    sample_prior_physics,
    sse_deviance,
)
from apexpinn.physics.residuals import COMPOUNDS, PhysicsParams, thermal_residual, wear_residual
from apexpinn.physics.synthetic import (
    DEFAULT_COMPOUNDS,
    DEFAULT_N_STINTS,
    MISSPEC_VARIANTS,
    SyntheticDataset,
    make_synthetic_dataset,
)
from apexpinn.train.run_meta import run_metadata
from apexpinn.train.trainer import LossWeights, TrainConfig, sample_collocation, train

SCHEMA_VERSION = "apexpinn.s1_run/v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DECISIONS = "configs/decisions_v1.yaml"
GROUPS = ("rho", "k", "gamma", "h_opt", "kappa")  # profile order; ThetaLayout order is k, rho, gamma, h_opt
POSITIVE_GROUPS = ("rho", "k", "gamma", "kappa")  # profiled on a multiplicative grid in log coordinates
MODELS = ("M1", "M2")
MISSPEC_STUDY_VARIANTS = ("wrong_phi", "student_t", "piecewise_wear", "drop_early", "traffic")
RUN_STATUSES = ("ok", "nonfinite", "error")
DONE_STATUSES = ("ok", "nonfinite")
H_OPT_GRID_CLIP = (0.02, 0.98)
FULL_GRID_FACTORS = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)
SMOKE_GRID_FACTORS = (0.5, 0.75, 1.0, 1.33, 2.0)
H_OPT_OFFSETS = tuple(round(float(x), 6) for x in np.linspace(-0.35, 0.35, 13))
M1_WEIGHTS = LossWeights(data=1.0, thermal=1.0, wear=1.0, mono=0.1, reg=1e-6)
M2_WEIGHTS = LossWeights(data=1.0)  # M2 returns only the 'data' term
CONVERGENCE_TAIL = 0.2  # relative objective change over the last 20% of logged steps
# Key derivation (PLAN T5 method declarations): fold_in offsets of jax.random.key(seed).
PRIOR_FOLD, NET_FOLD, TRAIN_FOLD, COLLOC_FOLD, START_FOLD = 101, 102, 103, 104, 200
# Profile method version, part of every profile run's config_hash. 1 = unbounded L-BFGS-B, a failed ODE
# solve stops the run. 2 = box bounds S1C_BOUNDS and failed-solve handling, declared in PLAN T5.
PROFILE_METHOD_VERSION = 2
AT_BOUND_TOL = 1e-9  # a coordinate within this distance of a box edge counts as on the edge
O6_NAMES = {"rho_c": "rho", "k_c": "k"}

Log = Callable[[str], Any]
Runner = Callable[["StudyConfig", "RunSpec"], dict]


# --- configuration ---


def _int_tuple(name: str, values: Sequence[int], *, allow_empty: bool = False) -> tuple[int, ...]:
    out = tuple(values)
    ok = all(not isinstance(v, bool) and isinstance(v, int | np.integer) and v >= 0 for v in out)
    if not ok or len(set(out)) != len(out) or (not out and not allow_empty):
        raise ValueError(f"{name} must be distinct non-negative integers, got {values!r}")
    return tuple(int(v) for v in out)


def _increasing(name: str, values: Sequence[float], *, positive: bool) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)) or np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must hold at least 2 finite, strictly increasing values, got {values!r}")
    if positive and np.any(arr <= 0):
        raise ValueError(f"{name} must be positive, got {values!r}")
    return tuple(float(v) for v in arr)


@dataclass(frozen=True)
class StudyConfig:
    """Settings of one S1 study (smoke_config or full_config). Frozen and hashable.

    tolerance holds the O6 recovery tolerances as sorted (group, relative tolerance) pairs, e.g.
    (('k', 0.25), ('rho', 0.15)); groups with a tolerance are recovery-gated (S1b). grid_factors multiply
    the MLE value of positive parameters; h_opt_offsets are added to the MLE h_opt and clipped to
    [0.02, 0.98]. out_dir and report_path are relative to the repository root.
    """

    name: str
    n_stints: int
    compounds: tuple[str, ...]
    seeds_main: tuple[int, ...]
    noise_levels: tuple[float, ...]
    seeds_misspec: tuple[int, ...]
    misspec_variants: tuple[str, ...]
    misspec_noise: float
    tolerance: tuple[tuple[str, float], ...]
    m1_width: int
    m1_depth: int
    m1_train: TrainConfig
    m1_weights: LossWeights
    m2_train: TrainConfig
    profile_params: tuple[str, ...]
    grid_factors: tuple[float, ...]
    h_opt_offsets: tuple[float, ...]
    n_starts: int
    minimize_maxiter: int
    out_dir: str
    report_path: str
    models_main: tuple[str, ...] = MODELS
    m2_weights: LossWeights = M2_WEIGHTS
    m2_misspec_max_fit_s: float = 600.0
    ode_rtol: float = 1e-6
    ode_atol: float = 1e-6

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if isinstance(self.n_stints, bool) or not isinstance(self.n_stints, int) or self.n_stints < 1:
            raise ValueError(f"n_stints must be a positive integer, got {self.n_stints!r}")
        compounds = tuple(self.compounds)
        known = all(c in COMPOUNDS for c in compounds)
        if not compounds or len(set(compounds)) != len(compounds) or not known:
            raise ValueError(f"compounds must be distinct names from {COMPOUNDS}, got {self.compounds!r}")
        set_(self, "compounds", compounds)
        set_(self, "seeds_main", _int_tuple("seeds_main", self.seeds_main))
        set_(self, "seeds_misspec", _int_tuple("seeds_misspec", self.seeds_misspec, allow_empty=True))
        noise = tuple(float(v) for v in self.noise_levels)
        valid = all(np.isfinite(v) and v >= 0 for v in noise)
        if not noise or len(set(noise)) != len(noise) or not valid:
            raise ValueError(f"noise_levels must be distinct finite values >= 0, got {self.noise_levels!r}")
        set_(self, "noise_levels", noise)
        if not (np.isfinite(self.misspec_noise) and self.misspec_noise >= 0):
            raise ValueError(f"misspec_noise must be finite and >= 0, got {self.misspec_noise!r}")
        set_(self, "misspec_noise", float(self.misspec_noise))
        variants = tuple(self.misspec_variants)
        if len(set(variants)) != len(variants) or any(v not in MISSPEC_STUDY_VARIANTS for v in variants):
            raise ValueError(f"misspec_variants must be distinct names from {MISSPEC_STUDY_VARIANTS}")
        set_(self, "misspec_variants", variants)
        models = tuple(self.models_main)
        if not models or len(set(models)) != len(models) or any(m not in MODELS for m in models):
            raise ValueError(f"models_main must be distinct names from {MODELS}, got {self.models_main!r}")
        set_(self, "models_main", models)
        pairs = self.tolerance.items() if isinstance(self.tolerance, Mapping) else self.tolerance
        tol = tuple(sorted((str(g), float(t)) for g, t in pairs))
        if not tol or any(g not in ("rho", "k") or not (np.isfinite(t) and t > 0) for g, t in tol):
            raise ValueError(f"tolerance must be (group, > 0) pairs with groups 'rho' and 'k', got {tol!r}")
        set_(self, "tolerance", tol)
        for name in ("m1_width", "m1_depth", "n_starts", "minimize_maxiter"):
            value = getattr(self, name)
            minimum = 0 if name == "m1_depth" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        for name in ("m1_train", "m2_train"):
            if not isinstance(getattr(self, name), TrainConfig):
                raise ValueError(f"{name} must be a TrainConfig")
        for name in ("m1_weights", "m2_weights"):
            if not isinstance(getattr(self, name), LossWeights):
                raise ValueError(f"{name} must be a LossWeights")
        params = tuple(self.profile_params)
        if len(set(params)) != len(params) or any(p not in GROUPS for p in params):
            raise ValueError(f"profile_params must be distinct names from {GROUPS}, got {params!r}")
        set_(self, "profile_params", params)
        set_(self, "grid_factors", _increasing("grid_factors", self.grid_factors, positive=True))
        set_(self, "h_opt_offsets", _increasing("h_opt_offsets", self.h_opt_offsets, positive=False))
        for name in ("m2_misspec_max_fit_s", "ode_rtol", "ode_atol"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        for name in ("out_dir", "report_path"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty relative path string")

    @property
    def tolerance_map(self) -> dict[str, float]:
        return dict(self.tolerance)


def _decisions_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    return p


O6Values = tuple[tuple[int, ...], tuple[float, ...], tuple[tuple[str, float], ...]]


def _o6(decisions_path: str | Path) -> O6Values:
    """(seeds_main, noise_levels, tolerance) from the O6 block of the decisions file."""
    o6 = load_decisions(_decisions_path(decisions_path)).synthetic
    try:
        seeds = tuple(range(int(o6["n_seeds"])))
        noise = tuple(float(v) for v in o6["noise_levels_s"])
        raw_tol = dict(o6["tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"o6_synthetic is missing n_seeds, noise_levels_s or tolerance: {exc}") from exc
    unknown = [k for k in raw_tol if k not in O6_NAMES]
    if unknown:
        raise ValueError(f"o6_synthetic.tolerance has unknown parameters {unknown}")
    tolerance = tuple(sorted((O6_NAMES[k], float(v)) for k, v in raw_tol.items()))
    return seeds, noise, tolerance


def smoke_config(decisions_path: str | Path = DEFAULT_DECISIONS) -> StudyConfig:
    """Small pipeline check: one seed, one noise level, short training. Its numbers do not answer S1."""
    _, _, tolerance = _o6(decisions_path)
    return StudyConfig(
        name="smoke",
        n_stints=12,
        compounds=DEFAULT_COMPOUNDS,
        seeds_main=(0,),
        noise_levels=(0.05,),
        seeds_misspec=(0,),
        misspec_variants=MISSPEC_STUDY_VARIANTS,
        misspec_noise=0.05,
        tolerance=tolerance,
        m1_width=32,
        m1_depth=2,
        m1_train=TrainConfig(
            n_steps=300, learning_rate=3e-3, n_warm=50, n_ramp=100, n_colloc=16, log_every=50, float64=True
        ),
        m1_weights=M1_WEIGHTS,
        m2_train=TrainConfig(n_steps=100, learning_rate=1e-2, log_every=25, float64=True),
        profile_params=("rho", "k"),
        grid_factors=SMOKE_GRID_FACTORS,
        h_opt_offsets=H_OPT_OFFSETS,
        n_starts=1,
        minimize_maxiter=100,
        out_dir="experiments/s1_smoke",
        report_path="reports/s1_identifiability_smoke.md",
    )


def full_config(decisions_path: str | Path = DEFAULT_DECISIONS) -> StudyConfig:
    """The S1 study of PLAN T5: O6 seeds and noise levels, M1 and M2, all five profiled parameters."""
    seeds, noise, tolerance = _o6(decisions_path)
    return StudyConfig(
        name="full",
        n_stints=DEFAULT_N_STINTS,
        compounds=DEFAULT_COMPOUNDS,
        seeds_main=seeds,
        noise_levels=noise,
        seeds_misspec=(0, 1, 2),
        misspec_variants=MISSPEC_STUDY_VARIANTS,
        misspec_noise=0.05,
        tolerance=tolerance,
        m1_width=32,
        m1_depth=2,
        m1_train=TrainConfig(
            n_steps=5000, learning_rate=3e-3, n_warm=200, n_ramp=800, n_colloc=32, log_every=250, float64=True
        ),
        m1_weights=M1_WEIGHTS,
        m2_train=TrainConfig(n_steps=1000, learning_rate=1e-2, log_every=100, float64=True),
        profile_params=GROUPS,
        grid_factors=FULL_GRID_FACTORS,
        h_opt_offsets=H_OPT_OFFSETS,
        n_starts=3,
        minimize_maxiter=500,
        out_dir="experiments/s1",
        report_path="reports/s1_identifiability.md",
    )


# --- run plan ---


@dataclass(frozen=True)
class RunSpec:
    """One run: a recovery fit of one model, or the ODE-likelihood profiles ('profile') of one dataset."""

    part: str  # 'main' | 'misspec'
    kind: str  # 'fit' | 'profile'
    model: str | None  # 'M1' | 'M2' for fits, None for profiles
    variant: str  # 'well_specified' for main
    noise_s: float
    seed: int

    def __post_init__(self) -> None:
        if self.part not in ("main", "misspec"):
            raise ValueError(f"part must be 'main' or 'misspec', got {self.part!r}")
        if self.kind not in ("fit", "profile"):
            raise ValueError(f"kind must be 'fit' or 'profile', got {self.kind!r}")
        if (self.kind == "fit") != (self.model in MODELS):
            raise ValueError(f"model must be one of {MODELS} for fits, None for profiles; got {self.model!r}")
        allowed = ("well_specified",) if self.part == "main" else MISSPEC_STUDY_VARIANTS
        if self.variant not in allowed:
            raise ValueError(f"variant must be one of {allowed} for part {self.part!r}, got {self.variant!r}")
        object.__setattr__(self, "noise_s", float(self.noise_s))

    @property
    def run_id(self) -> str:
        model = self.model if self.model is not None else "ode"
        return f"{self.part}_{self.kind}_{model}_{self.variant}_noise{self.noise_s:g}_seed{self.seed}"


def plan_runs(cfg: StudyConfig, part: str, include_m2_misspec: bool = False) -> list[RunSpec]:
    """Planned runs of one part, sorted by run_id.

    main: per seed and noise level one profile run plus one fit per model in models_main.
    misspec: per seed and variant (noise misspec_noise) one profile run, one M1 fit, and an M2 fit if
    include_m2_misspec.
    """
    if part not in ("main", "misspec"):
        raise ValueError(f"part must be 'main' or 'misspec', got {part!r}")
    runs: list[RunSpec] = []
    if part == "main":
        for seed in cfg.seeds_main:
            for noise in cfg.noise_levels:
                runs.append(RunSpec("main", "profile", None, "well_specified", noise, seed))
                for model in cfg.models_main:
                    runs.append(RunSpec("main", "fit", model, "well_specified", noise, seed))
    else:
        models = ("M1", "M2") if include_m2_misspec else ("M1",)
        for seed in cfg.seeds_misspec:
            for variant in cfg.misspec_variants:
                runs.append(RunSpec("misspec", "profile", None, variant, cfg.misspec_noise, seed))
                runs.extend(RunSpec("misspec", "fit", m, variant, cfg.misspec_noise, seed) for m in models)
    return sorted(runs, key=lambda s: s.run_id)


def parse_shard(text: str) -> tuple[int, int]:
    """'i/n' -> (i, n) with 0 <= i < n."""
    try:
        i_text, n_text = text.split("/")
        shard = (int(i_text), int(n_text))
    except ValueError as exc:
        raise ValueError(f"shard must look like 'i/n', got {text!r}") from exc
    return _check_shard(shard)


def _check_shard(shard: tuple[int, int]) -> tuple[int, int]:
    i, n = shard
    if isinstance(i, bool) or isinstance(n, bool) or not (isinstance(i, int) and isinstance(n, int)):
        raise ValueError(f"shard must be two integers (i, n), got {shard!r}")
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"shard must satisfy 0 <= i < n, got {shard!r}")
    return i, n


def shard_runs(runs: Sequence[RunSpec], shard: tuple[int, int]) -> list[RunSpec]:
    """Runs of shard (i, n): runs sorted by run_id, then every n-th starting at i."""
    i, n = _check_shard(shard)
    return sorted(runs, key=lambda s: s.run_id)[i::n]


# --- keys, priors, data ---


def data_key(seed: int) -> jax.Array:
    """Data key of a seed, shared by every noise level and variant (paired design)."""
    return jax.random.key(seed)


def prior_key(seed: int) -> jax.Array:
    return jax.random.fold_in(data_key(seed), PRIOR_FOLD)


def net_key(seed: int) -> jax.Array:
    return jax.random.fold_in(data_key(seed), NET_FOLD)


def train_key(seed: int) -> jax.Array:
    return jax.random.fold_in(data_key(seed), TRAIN_FOLD)


def colloc_key(seed: int) -> jax.Array:
    return jax.random.fold_in(data_key(seed), COLLOC_FOLD)


def start_key(seed: int, i: int) -> jax.Array:
    """Key of profile MLE start i > 0; start 0 is the shared prior draw."""
    if i < 1:
        raise ValueError(f"i must be >= 1 (start 0 is the prior draw), got {i!r}")
    return jax.random.fold_in(data_key(seed), START_FOLD + i)


def prior_physics(seed: int) -> PhysicsParams:
    """Prior initialization draw of a seed (identifiability.sample_prior_physics at prior_key)."""
    return sample_prior_physics(prior_key(seed))


def init_physics(spec: RunSpec) -> PhysicsParams:
    """Initial Theta of a run: the seed's prior draw, the same for M1, M2 and the first profile start."""
    return prior_physics(spec.seed)


def dataset_kwargs(cfg: StudyConfig, spec: RunSpec) -> dict[str, Any]:
    """Arguments of make_synthetic_dataset for a run. The key depends on the seed only."""
    return {
        "key": data_key(spec.seed),
        "n_stints": cfg.n_stints,
        "compounds": cfg.compounds,
        "noise_s": spec.noise_s,
        "misspec": spec.variant,
    }


def dataset_for(cfg: StudyConfig, spec: RunSpec) -> SyntheticDataset:
    return make_synthetic_dataset(**dataset_kwargs(cfg, spec))


# --- parameter names and values ---


def param_names(group: str, compounds: Sequence[str]) -> list[str]:
    """ThetaLayout names of a group: 'kappa', or '<group>_c[<compound>]' per compound."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    return ["kappa"] if group == "kappa" else [f"{group}_c[{c}]" for c in compounds]


def group_of(name: str) -> str:
    return name.split("_c[", 1)[0]


def gated_names(cfg: StudyConfig) -> list[str]:
    """Recovery-gated parameters (groups with an O6 tolerance), in GROUPS order."""
    tol = cfg.tolerance_map
    return [n for g in GROUPS if g in tol for n in param_names(g, cfg.compounds)]


def profiled_names(cfg: StudyConfig) -> list[str]:
    return [n for g in GROUPS if g in cfg.profile_params for n in param_names(g, cfg.compounds)]


def theta_values(physics: PhysicsParams, compounds: Sequence[str], rho: Any = None) -> dict[str, float]:
    """Constrained values by ThetaLayout name for the used compounds. rho (shape (5,)) overrides rho_c."""
    layout = ThetaLayout.from_compounds(compounds)
    idx = list(layout.compound_idx)
    values: dict[str, float] = {}
    for group in ("k", "rho", "gamma", "h_opt"):
        arr = np.asarray(rho if group == "rho" and rho is not None else getattr(physics, group), np.float64)
        values.update({n: float(arr[i]) for n, i in zip(param_names(group, compounds), idx, strict=True)})
    values["kappa"] = float(np.asarray(physics.kappa, dtype=np.float64))
    return {n: values[n] for n in layout.names}


def variant_truth(ds: SyntheticDataset, compounds: Sequence[str]) -> dict[str, float]:
    """Truth each variant records: rho_c against ds.rho_target, the rest against theta_true."""
    return theta_values(ds.theta_true, compounds, rho=ds.rho_target)


# --- config hash and JSON ---


def _jsonable(value: Any) -> Any:
    """JSON-safe copy: numpy scalars and arrays to Python, non-finite floats to None, tuples to lists."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray | jax.Array):
        return _jsonable(np.asarray(value).tolist())
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _json_partial(value: Any) -> Any:
    """_jsonable(value) if json.dumps accepts it, else repr(value), so an error record always writes.

    A runner that raises ProfilePointError itself leaves a ProfileResult dataclass in .partial.
    """
    safe = _jsonable(value)
    try:
        json.dumps(safe, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return safe


def generator_sources() -> tuple[Path, ...]:
    """Source files whose text enters every run's config_hash: the synthetic generator and residuals.

    apexpinn.physics.synthetic holds make_synthetic_dataset, default_theta_true and MISSPEC_VARIANTS;
    apexpinn.physics.residuals holds the ODE right-hand sides and the observation equation that both the
    generator and the fitted models use.
    """
    return (Path(inspect.getfile(make_synthetic_dataset)), Path(inspect.getfile(thermal_residual)))


def source_digest(paths: Sequence[str | Path]) -> str:
    """sha256 over the files in order (file name and text), with line endings normalized."""
    digest = hashlib.sha256()
    for item in paths:
        path = Path(item)
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        digest.update(f"--- {path.name}\n".encode())
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def generator_digest() -> str:
    """source_digest of generator_sources(): the synthetic generator and the residual equations.

    An edit to the generator, the misspecification variants, the ODE right-hand sides or the observation
    equation changes every run's config_hash, so old records are never mixed with new ones. Line endings
    are normalized so Windows and Linux checkouts of the same commit give the same digest.
    """
    return source_digest(generator_sources())


def _ode_max_steps() -> int:
    """ODEModel.max_steps class default: the step budget of every ODE solve in the S1c likelihood."""
    return int(next(f.default for f in fields(ODEModel) if f.name == "max_steps"))


def _settings(cfg: StudyConfig, spec: RunSpec) -> dict[str, Any]:
    """Only the settings that the run uses, so unrelated config changes keep its hash.

    Every run also hashes the data and initialization code it depends on: the source digest of the
    generator and the residual equations, the variant's Misspecification fields, PRIOR_RANGES and the key
    fold offsets. Profile runs also hash the optimizer box S1C_BOUNDS, the ODE max_steps and
    PROFILE_METHOD_VERSION. The O6 tolerance is not hashed: the report recomputes recovery from
    rel_error, so a tolerance edit needs no rerun.
    """
    settings: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "spec": asdict(spec),
        "n_stints": cfg.n_stints,
        "compounds": list(cfg.compounds),
        "generator": {
            "source_sha256": generator_digest(),
            "misspec": asdict(MISSPEC_VARIANTS[spec.variant]),
        },
        "prior_ranges": {g: list(v) for g, v in sorted(PRIOR_RANGES.items())},
        "key_folds": {
            "prior": PRIOR_FOLD,
            "net": NET_FOLD,
            "train": TRAIN_FOLD,
            "colloc": COLLOC_FOLD,
            "start": START_FOLD,
        },
    }
    if spec.kind == "fit":
        if spec.model == "M1":
            settings["m1"] = {
                "width": cfg.m1_width,
                "depth": cfg.m1_depth,
                "train": asdict(cfg.m1_train),
                "weights": asdict(cfg.m1_weights),
            }
        else:
            settings["m2"] = {
                "train": asdict(cfg.m2_train),
                "weights": asdict(cfg.m2_weights),
                "rtol": cfg.ode_rtol,
                "atol": cfg.ode_atol,
            }
    else:
        settings["profile"] = {
            "params": list(cfg.profile_params),
            "grid_factors": (
                list(cfg.grid_factors) if any(g in POSITIVE_GROUPS for g in cfg.profile_params) else None
            ),
            "h_opt_offsets": list(cfg.h_opt_offsets) if "h_opt" in cfg.profile_params else None,
            "n_starts": cfg.n_starts,
            "maxiter": cfg.minimize_maxiter,
            "rtol": cfg.ode_rtol,
            "atol": cfg.ode_atol,
            "bounds": {g: list(v) for g, v in sorted(S1C_BOUNDS.items())},
            "max_steps": _ode_max_steps(),
            "method_version": PROFILE_METHOD_VERSION,
        }
    return settings


def config_hash(cfg: StudyConfig, spec: RunSpec) -> str:
    """sha256 of the canonical JSON (sorted keys) of the settings the run uses plus SCHEMA_VERSION."""
    text = json.dumps(_jsonable(_settings(cfg, spec)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """json.dumps(sort_keys=True, indent=2) to <path>.tmp, then os.replace (data/manifest.py pattern)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_records(out_dir: str | Path) -> dict[str, dict]:
    """run_id -> record for every readable <run_id>.json in out_dir ({} if it does not exist)."""
    folder = Path(out_dir)
    if not folder.is_dir():
        return {}
    records: dict[str, dict] = {}
    for path in sorted(folder.glob("*.json")):
        record = _read_json(path)
        if record is not None:
            records[str(record.get("run_id", path.stem))] = record
    return records


def usable(record: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The record if it finished (status ok or nonfinite), else None."""
    return record if record is not None and record.get("status") in DONE_STATUSES else None


# --- four-field records ---


def _device() -> str:
    return str(jax.devices()[0])


def _population(cfg: StudyConfig, spec: RunSpec, n_scored: int) -> str:
    return (
        f"synthetic stints from make_synthetic_dataset(key(seed={spec.seed})), n_stints={cfg.n_stints}, "
        f"n_scored={n_scored} laps, compounds={list(cfg.compounds)}, variant={spec.variant}"
    )


def _metric(quantity: str, population: str, conditions: str, instrument: str, value: Any) -> dict:
    v = None if value is None or not math.isfinite(float(value)) else float(value)
    return asdict(MetricRecord(quantity, population, conditions, instrument, v, "Observed"))


def objective_rel_change(objective: Sequence[float], tail: float = CONVERGENCE_TAIL) -> float | None:
    """(obj[-1] - obj[s]) / |obj[s]| with s the first logged step of the last `tail` share of the log."""
    obj = np.asarray(objective, dtype=np.float64)
    if obj.size < 2:
        return None
    s = min(int(math.floor((1.0 - tail) * (obj.size - 1))), obj.size - 2)
    ref, last = obj[s], obj[-1]
    if not (np.isfinite(ref) and np.isfinite(last)) or ref == 0:
        return None
    return float((last - ref) / abs(ref))


_DIAG_QUANTITY = {
    "scored_residual_rms_s": "sqrt of the final data MSE on scored laps with training offsets b_s, s",
    "residual_ratio": "scored_residual_rms_s / noise_s (criterion 3 compares it with 1.2), dimensionless",
    "thermal_residual_rms": "RMS of r_T at a fixed collocation sample (fold_in(key, 104)), per lap",
    "wear_residual_rms": "RMS of r_W at the same collocation sample, s per lap",
    "wear_violation_rate": "share of the same collocation points with dw/dtau < -1e-6, dimensionless",
}


def build_fit_record(
    cfg: StudyConfig,
    spec: RunSpec,
    *,
    estimates: Mapping[str, float],
    truth: Mapping[str, float],
    n_scored: int,
    diagnostics: Mapping[str, Any],
    convergence: Mapping[str, Any],
    wall_time_s: Mapping[str, float],
) -> dict:
    """Fit record: estimates, variant truth, relative errors, O6 recovery flags, diagnostics, records.

    rel_error = (estimate - truth) / truth (None if the estimate is not finite). recovered holds the
    recovery-gated parameters: |rel_error| <= O6 tolerance. status is 'nonfinite' if any estimate is not
    finite, else 'ok'.
    """
    if spec.kind != "fit":
        raise ValueError(f"spec must be a fit run, got kind {spec.kind!r}")
    tol = cfg.tolerance_map
    finite = {n: v is not None and math.isfinite(float(v)) for n, v in estimates.items()}
    rel_error = {
        n: (float(estimates[n]) - float(truth[n])) / float(truth[n]) if finite[n] else None for n in estimates
    }
    recovered = {
        n: rel_error[n] is not None and abs(rel_error[n]) <= tol[group_of(n)]
        for n in estimates
        if group_of(n) in tol
    }
    train_cfg = cfg.m1_train if spec.model == "M1" else cfg.m2_train
    population = _population(cfg, spec, n_scored)
    conditions = (
        f"noise_s={spec.noise_s:g} s, lap time only, init from the prior draw fold_in(key({spec.seed}), "
        f"{PRIOR_FOLD}), model={spec.model}, TrainConfig({asdict(train_cfg)}), "
        f"dtype={'float64' if train_cfg.float64 else 'float32'}, device={_device()}"
    )
    model_name = "apexpinn.models.pinn.PINN" if spec.model == "M1" else "apexpinn.models.ode_model.ODEModel"
    instrument = f"apexpinn.train.s1_study.run_fit, apexpinn.train.trainer.train, {model_name}"
    records = [
        _metric(
            f"relative error (estimate - truth) / truth of {n}, dimensionless",
            population,
            conditions,
            instrument,
            rel_error[n],
        )
        for n in estimates
    ]
    for key, quantity in _DIAG_QUANTITY.items():
        if key in diagnostics:
            records.append(_metric(quantity, population, conditions, instrument, diagnostics[key]))
    return {
        "status": "ok" if all(finite.values()) else "nonfinite",
        "estimates": dict(estimates),
        "truth": dict(truth),
        "rel_error": rel_error,
        "recovered": recovered,
        "diagnostics": dict(diagnostics),
        "convergence": dict(convergence),
        "n_scored": int(n_scored),
        "wall_time_s": dict(wall_time_s),
        "records": records,
    }


def build_profile_record(
    cfg: StudyConfig,
    spec: RunSpec,
    *,
    mle: Mapping[str, Any],
    spectrum: Mapping[str, Any] | None,
    profiles: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, float],
    sse_truth: float | None,
    n_obs: int,
    wall_time_s: Mapping[str, float],
) -> dict:
    """Profile record: MLE, Gauss-Newton spectrum, per-parameter profiles and intervals, records.

    status is 'nonfinite' if the MLE, the spectrum or any profile shape could not be computed.
    """
    if spec.kind != "profile":
        raise ValueError(f"spec must be a profile run, got kind {spec.kind!r}")
    values = mle.get("values", {})
    complete = (
        spectrum is not None
        and all(v is not None and math.isfinite(float(v)) for v in values.values())
        and all(p.get("shape") is not None for p in profiles.values())
    )
    population = _population(cfg, spec, n_obs)
    conditions = (
        f"noise_s={spec.noise_s:g} s, lap time only, exact ODE likelihood with b_s concentrated out, "
        f"log / logit coordinates, MLE from {cfg.n_starts} start(s) (start 0 = prior draw fold_in(key("
        f"{spec.seed}), {PRIOR_FOLD})), L-BFGS-B maxiter={cfg.minimize_maxiter}, free coordinates "
        f"inside S1C_BOUNDS, profile method v{PROFILE_METHOD_VERSION}, rtol={cfg.ode_rtol:g}, "
        f"atol={cfg.ode_atol:g}, max_steps={_ode_max_steps()}, device={_device()}"
    )
    instrument = (
        "apexpinn.train.s1_study.run_profile, apexpinn.physics.identifiability "
        "(ode_least_squares, ode_mle, profile_likelihood, likelihood_interval, gauss_newton_spectrum)"
    )
    records = []
    for name, prof in profiles.items():
        interval = prof.get("interval") or {}
        for side in ("lo", "hi"):
            records.append(
                _metric(
                    f"95% profile-likelihood interval {'lower' if side == 'lo' else 'upper'} endpoint of "
                    f"{name} (deviance n log(SSE / SSE_min) <= 3.841), natural units; an unbounded side "
                    "reports the grid edge",
                    population,
                    conditions,
                    instrument,
                    interval.get(side),
                )
            )
    if spectrum is not None:
        records.append(
            _metric(
                "smallest relative Gauss-Newton eigenvalue (eigenvalue / largest) of J^T J at the MLE, "
                "dimensionless",
                population,
                conditions,
                instrument,
                spectrum.get("smallest_relative"),
            )
        )
    return {
        "status": "ok" if complete else "nonfinite",
        "mle": dict(mle),
        "spectrum": None if spectrum is None else dict(spectrum),
        "profiles": {n: dict(p) for n, p in profiles.items()},
        "truth": dict(truth),
        "sse_truth": sse_truth,
        "n_obs": int(n_obs),
        "wall_time_s": dict(wall_time_s),
        "records": records,
    }


# --- runners ---


def m1_physics_diagnostics(model: PINN, batch: StintBatch, key: jax.Array, train_cfg: TrainConfig) -> dict:
    """RMS of r_T and r_W and the wear violation rate of a fitted PINN on one fixed collocation sample.

    q at a collocation point follows PINN.loss_terms: linear between knots and held at knot n_laps[s].
    """
    colloc = sample_collocation(key, batch.n_laps, train_cfg.n_colloc, train_cfg.colloc_extension)
    colloc = colloc.astype(batch.tau.dtype)
    knots = jnp.arange(batch.q.shape[1], dtype=batch.tau.dtype)
    physics = model.physics

    def per_stint(c, ctx, q_knots, t_col, n_s):
        h, _, dh, dw = model.latent_and_rates(t_col, ctx)
        q_c = jnp.interp(jnp.minimum(t_col, n_s.astype(t_col.dtype)), knots, q_knots)
        return thermal_residual(physics, c, h, dh, q_c), wear_residual(physics, c, h, dw), dw

    r_t, r_w, dw = jax.vmap(per_stint)(batch.compound_idx, batch.ctx, batch.q, colloc, batch.n_laps)
    return {
        "thermal_residual_rms": float(residual_rms(r_t)),
        "wear_residual_rms": float(residual_rms(r_w)),
        "wear_violation_rate": float(wear_violation_rate(dw)),
        "n_colloc_points": int(colloc.size),
    }


def run_fit(cfg: StudyConfig, spec: RunSpec) -> dict:
    """Train M1 or M2 on one dataset from the seed's prior draw and build the fit record."""
    start = time.perf_counter()
    ds = dataset_for(cfg, spec)
    batch = batch_from_synthetic(ds)
    prior = init_physics(spec)
    if spec.model == "M1":
        train_cfg, weights = cfg.m1_train, cfg.m1_weights
        model: Any = PINN.init(
            net_key(spec.seed),
            width=cfg.m1_width,
            depth=cfg.m1_depth,
            physics=prior,
            train_cfg=train_cfg,
            weights=weights,
        )
    else:
        train_cfg, weights = cfg.m2_train, cfg.m2_weights
        model = ODEModel.init(
            physics=prior, train_cfg=train_cfg, weights=weights, rtol=cfg.ode_rtol, atol=cfg.ode_atol
        )
    fitted, log = train(model, batch, weights, train_cfg, train_key(spec.seed))

    rms = float(fitted.noise_s)
    flagged = detect_collapse(fitted.physics).flagged
    diagnostics: dict[str, Any] = {
        "scored_residual_rms_s": rms,
        "residual_ratio": rms / spec.noise_s if spec.noise_s > 0 else None,
        "collapse_flags": [f for f in flagged if any(f.endswith(f"[{c}]") for c in cfg.compounds)],
    }
    if spec.model == "M1":
        diagnostics.update(m1_physics_diagnostics(fitted, batch, colloc_key(spec.seed), train_cfg))
    convergence = {
        "steps": log.steps,
        "objective": log.objective,
        "normalized": log.normalized,
        "objective_rel_change_last20": objective_rel_change(log.objective),
    }
    return build_fit_record(
        cfg,
        spec,
        estimates=theta_values(fitted.physics, cfg.compounds),
        truth=variant_truth(ds, cfg.compounds),
        n_scored=int(np.asarray(batch.mask).sum()),
        diagnostics=diagnostics,
        convergence=convergence,
        wall_time_s={"fit": float(log.wall_time_s), "total": time.perf_counter() - start},
    )


def profile_grid(cfg: StudyConfig, group: str, center: float) -> tuple[np.ndarray, np.ndarray]:
    """(grid in natural units, grid in layout coordinates) around an MLE value.

    Positive parameters: center * grid_factors, coordinate log. h_opt: center + h_opt_offsets clipped to
    [0.02, 0.98] and deduplicated, coordinate logit.
    """
    if not (math.isfinite(center) and center > 0):
        raise ValueError(f"center must be finite and > 0, got {center!r}")
    if group == "h_opt":
        natural = np.unique(np.clip(center + np.asarray(cfg.h_opt_offsets), *H_OPT_GRID_CLIP))
        if natural.size < 2:
            raise ValueError(f"h_opt grid around {center} has fewer than 2 distinct points")
        return natural, np.log(natural) - np.log1p(-natural)
    natural = center * np.asarray(cfg.grid_factors)
    return natural, np.log(natural)


def _to_natural(group: str, coord: float) -> float:
    return float(1.0 / (1.0 + np.exp(-coord))) if group == "h_opt" else float(np.exp(coord))


class ProfileRunError(RuntimeError):
    """run_profile failed. partial is a JSON-safe dict of what was computed before the failure.

    Keys: phase ('dataset', 'mle', 'sse_truth', 'spectrum' or 'profile'), param (the profiled layout name,
    or None), mle and spectrum (None until computed), profiles (the finished profile entries), and point
    (the failing profile's ProfilePointError.partial: grid, grid_coord, sse, success, failed,
    n_failed_evals; None if the exception carried none). The original exception is __cause__.
    """

    def __init__(self, message: str, partial: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.partial = dict(partial)


def _on_box_edge(
    theta: np.ndarray, bounds: Sequence[tuple[float, float]], names: Sequence[str], skip: int | None = None
) -> list[str]:
    """Names of the finite coordinates within AT_BOUND_TOL of a box edge (index skip excluded)."""
    out = []
    for j, (name, (lo, hi), x) in enumerate(zip(names, bounds, np.asarray(theta, float), strict=True)):
        if j != skip and math.isfinite(x) and (abs(x - lo) <= AT_BOUND_TOL or abs(x - hi) <= AT_BOUND_TOL):
            out.append(name)
    return out


def _profile_entry(
    prof: ProfileResult,
    *,
    group: str,
    index: int,
    natural: np.ndarray,
    coord: np.ndarray,
    mle_value: float,
    truth: float,
    sse_fit: float,
    n_obs: int,
    bounds: Sequence[tuple[float, float]],
    names: Sequence[str],
) -> dict[str, Any]:
    """Record entry of one profile. Failed points get no deviance (NaN); they enter likelihood_interval,
    which treats a failed point met before the threshold crossing as an unbounded side."""
    entry: dict[str, Any] = {
        "group": group,
        "grid": natural,
        "grid_coord": coord,
        "sse": prof.loss,
        "success": prof.success,
        "failed": prof.failed,
        "n_failed_evals": prof.n_failed_evals,
        "at_bound": [_on_box_edge(theta, bounds, names, skip=index) for theta in prof.thetas],
        "mle": mle_value,
        "truth": truth,
        "interval": None,
        "shape": None,
        "covers_truth": None,
    }
    usable_sse = np.isfinite(prof.loss) & (prof.loss > 0)
    if not np.any(usable_sse):
        return entry
    sse_min = min(float(np.min(prof.loss[usable_sse])), sse_fit)
    deviance = np.full(prof.loss.shape, np.nan)
    deviance[usable_sse] = sse_deviance(prof.loss[usable_sse], n_obs, sse_min=sse_min)
    iv = likelihood_interval(coord, deviance)
    lo, hi = _to_natural(group, iv.lo), _to_natural(group, iv.hi)
    natural_iv = Interval(
        lo=lo,
        hi=hi,
        lo_bounded=iv.lo_bounded,
        hi_bounded=iv.hi_bounded,
        threshold=iv.threshold,
        lo_failed=iv.lo_failed,
        hi_failed=iv.hi_failed,
    )
    entry.update(
        deviance=deviance,
        sse_min=sse_min,
        interval=asdict(natural_iv),
        shape=profile_shape_label(natural_iv, truth),
        covers_truth=(not iv.lo_bounded or lo <= truth) and (not iv.hi_bounded or truth <= hi),
    )
    return entry


def run_profile(cfg: StudyConfig, spec: RunSpec) -> dict:
    """S1c on one dataset: ODE-likelihood MLE, Gauss-Newton spectrum, and profiles of profile_params.

    One jitted value_and_grad of the SSE (eqx.filter_jit, so equinox runtime errors stay readable) is
    built per dataset and shared by the MLE and every profile, so the Diffrax gradient compiles once.
    The MLE and every profile solve keep the free coordinates inside S1C_BOUNDS; a failed ODE solve gives
    a non-finite SSE, which the optimizers back off from (PLAN T5 method declarations). Any exception is
    re-raised as ProfileRunError naming the phase (and the profiled parameter) with the partial results.
    """
    state: dict[str, Any] = {"phase": "dataset", "param": None, "grid": None}
    done: dict[str, Any] = {"mle": None, "spectrum": None, "profiles": {}}
    try:
        return _run_profile(cfg, spec, state, done)
    except Exception as exc:
        where = f" for {state['param']}" if state["param"] else ""
        point = getattr(exc, "partial", None)
        partial = {**state, **done, "point": None}
        del partial["grid"]
        if isinstance(point, ProfileResult):
            partial["point"] = {
                "grid": state["grid"],
                "grid_coord": point.grid,
                "sse": point.loss,
                "success": point.success,
                "failed": point.failed,
                "n_failed_evals": point.n_failed_evals,
            }
        message = f"run_profile failed in phase {state['phase']}{where}: {type(exc).__name__}: {exc}"
        raise ProfileRunError(message, _jsonable(partial)) from exc


def _run_profile(cfg: StudyConfig, spec: RunSpec, state: dict[str, Any], done: dict[str, Any]) -> dict:
    """Body of run_profile. state tracks the phase and parameter; done collects finished results."""
    t0 = time.perf_counter()
    ds = dataset_for(cfg, spec)
    batch = batch_from_synthetic(ds)
    layout = ThetaLayout.from_compounds(cfg.compounds)
    bounds = layout.coordinate_bounds(S1C_BOUNDS)
    base = init_physics(spec)
    sse_fn, residual_fn, n_obs = ode_least_squares(batch, layout, base, rtol=cfg.ode_rtol, atol=cfg.ode_atol)
    value_and_grad = eqx.filter_jit(jax.value_and_grad(sse_fn))
    starts = [layout.to_vector(base)]
    for i in range(1, cfg.n_starts):
        starts.append(layout.to_vector(sample_prior_physics(start_key(spec.seed, i))))

    state["phase"] = "mle"
    fit = ode_mle(sse_fn, starts, maxiter=cfg.minimize_maxiter, value_and_grad=value_and_grad, bounds=bounds)
    values = layout.constrained(fit.theta)
    mle = {
        "theta": fit.theta,
        "values": values,
        "sse": fit.sse,
        "start_sse": fit.start_sse,
        "best_index": fit.best_index,
        "success": fit.success,
        "start_failed": fit.start_failed,
        "n_failed_evals": fit.n_failed_evals,
        "at_bound": _on_box_edge(fit.theta, bounds, layout.names),
    }
    done["mle"] = mle

    state["phase"] = "sse_truth"
    truth = variant_truth(ds, cfg.compounds)
    v_truth = layout.to_vector(ds.theta_true)
    m = len(layout.compound_idx)
    v_truth[m : 2 * m] = np.log(np.asarray(ds.rho_target, dtype=np.float64)[list(layout.compound_idx)])
    sse_truth: float | None = float(sse_fn(jnp.asarray(v_truth)))
    if not math.isfinite(sse_truth):
        sse_truth = None
    t_mle = time.perf_counter()
    if not (np.all(np.isfinite(fit.theta)) and math.isfinite(fit.sse)):
        elapsed = t_mle - t0
        return build_profile_record(
            cfg,
            spec,
            mle=mle,
            spectrum=None,
            profiles={},
            truth=truth,
            sse_truth=sse_truth,
            n_obs=n_obs,
            wall_time_s={"mle": elapsed, "spectrum": 0.0, "profiles": 0.0, "total": elapsed},
        )

    state["phase"] = "spectrum"
    spec_result = gauss_newton_spectrum(residual_fn, fit.theta, mode="rev")
    spectrum = {
        "eigenvalues": spec_result.eigenvalues,
        "relative": spec_result.relative,
        "smallest_relative": float(spec_result.relative[-1]),
        "weakest_loadings": dict(zip(layout.names, spec_result.eigenvectors[:, -1].tolist(), strict=True)),
    }
    done["spectrum"] = spectrum
    t_spec = time.perf_counter()

    state["phase"] = "profile"
    profiles: dict[str, dict[str, Any]] = done["profiles"]
    for group in GROUPS:
        if group not in cfg.profile_params:
            continue
        for name in param_names(group, cfg.compounds):
            state["param"] = name
            index = layout.names.index(name)
            natural, coord = profile_grid(cfg, group, values[name])
            state["grid"] = natural
            prof = profile_likelihood(
                sse_fn,
                fit.theta,
                index,
                coord,
                maxiter=cfg.minimize_maxiter,
                value_and_grad=value_and_grad,
                bounds=bounds,
            )
            profiles[name] = _profile_entry(
                prof,
                group=group,
                index=index,
                natural=natural,
                coord=coord,
                mle_value=values[name],
                truth=truth[name],
                sse_fit=fit.sse,
                n_obs=n_obs,
                bounds=bounds,
                names=layout.names,
            )
    state.update(param=None, grid=None)
    t_end = time.perf_counter()
    wall_time_s = {"mle": t_mle - t0, "spectrum": t_spec - t_mle, "profiles": t_end - t_spec}
    wall_time_s["total"] = t_end - t0
    return build_profile_record(
        cfg,
        spec,
        mle=mle,
        spectrum=spectrum,
        profiles=profiles,
        truth=truth,
        sse_truth=sse_truth,
        n_obs=n_obs,
        wall_time_s=wall_time_s,
    )


def run_one(cfg: StudyConfig, spec: RunSpec) -> dict:
    """Default runner: run_fit for fits, run_profile for profiles."""
    return run_fit(cfg, spec) if spec.kind == "fit" else run_profile(cfg, spec)


# --- study ---


def m2_misspec_decision(cfg: StudyConfig, records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Whether M2 joins the misspecification study: slowest main M2 fit < m2_misspec_max_fit_s.

    include is None (undecided) while any planned main M2 fit record is missing or failed.
    """
    threshold = cfg.m2_misspec_max_fit_s
    decision: dict[str, Any] = {"max_fit_s": None, "max_run_id": None, "missing": []}
    decision["threshold_s"] = threshold
    if "M2" not in cfg.models_main:
        return {**decision, "include": False, "reason": "M2 is not in models_main"}
    specs = [s for s in plan_runs(cfg, "main") if s.kind == "fit" and s.model == "M2"]
    times: dict[str, float] = {}
    missing: list[str] = []
    for s in specs:
        record = usable(records.get(s.run_id))
        fit_s = None if record is None else (record.get("wall_time_s") or {}).get("fit")
        if fit_s is None or not math.isfinite(float(fit_s)):
            missing.append(s.run_id)
        else:
            times[s.run_id] = float(fit_s)
    max_run_id = max(times, key=times.__getitem__) if times else None
    max_fit_s = times[max_run_id] if max_run_id is not None else None
    if missing:
        include, reason = None, f"{len(missing)} of {len(specs)} main M2 fit records are missing or failed"
    else:
        include = max_fit_s is not None and max_fit_s < threshold
        reason = f"slowest main M2 fit {max_fit_s:.1f} s {'<' if include else '>='} {threshold:g} s"
    decision.update(include=include, max_fit_s=max_fit_s, max_run_id=max_run_id, missing=missing)
    decision["reason"] = reason
    return decision


def _check_runner_record(record: Any) -> dict:
    if not isinstance(record, dict):
        raise ValueError(f"runner must return a dict, got {type(record).__name__}")
    if record.get("status") not in RUN_STATUSES:
        raise ValueError(f"record status must be one of {RUN_STATUSES}, got {record.get('status')!r}")
    entries = record.get("records", [])
    if not isinstance(entries, list):
        raise ValueError("record 'records' must be a list of MetricRecord dicts")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each entry of 'records' must be a MetricRecord dict")
        MetricRecord(**entry)  # raises ValueError naming an empty field or a bad status
    return record


def run_study(
    cfg: StudyConfig,
    part: str = "all",
    shard: tuple[int, int] = (0, 1),
    force: bool = False,
    runner: Runner | None = None,
    root: str | Path = ".",
    log: Log = print,
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the planned runs of this shard that have no completed record, one JSON file per run.

    part 'all' runs main, then misspec. An existing record with the same config_hash and status ok or
    nonfinite is skipped; status error is rerun. A finished record whose config_hash differs raises
    ValueError naming the run, before anything runs, unless force=True, which reruns and overwrites it.
    An error record never blocks: it is rerun and overwritten whatever its config_hash. A runner
    exception is written as status 'error' with its text, its traceback and its partial attribute (the
    results computed before the failure, see ProfileRunError, or None), the log line shows the start of
    the error text, and the study continues.

    Misspec planning reads the main M2 fit records first (m2_misspec_decision). It raises ValueError if no
    main M2 fit record exists at all (part main has not run in this out_dir). While M2 is undecided (some
    main M2 fit missing or failed), only the M1 and profile runs are planned; the M2 runs are added by a
    later call once every main M2 fit has finished. only, if given, restricts the call to these run ids
    (ValueError naming an id that no part of this call plans).
    """
    if part not in ("main", "misspec", "all"):
        raise ValueError(f"part must be 'main', 'misspec' or 'all', got {part!r}")
    shard = _check_shard(tuple(shard))
    parts = ("main", "misspec") if part == "all" else (part,)
    selected: set[str] | None = None
    if only is not None:
        if isinstance(only, str):
            only = [only]
        selected = {str(rid) for rid in only}
        known = {s.run_id for p in parts for s in plan_runs(cfg, p, include_m2_misspec=True)}
        unknown = sorted(selected - known)
        if not selected or unknown:
            names = unknown or "none given"
            raise ValueError(f"only names run ids that part {part!r} does not plan: {names}")
    runner = run_one if runner is None else runner
    out = Path(root) / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "out_dir": str(out),
        "planned": 0,
        "run": 0,
        "skipped": 0,
        "errors": 0,
        "wall_time_s": 0.0,
        "m2_misspec": None,
    }
    start = time.perf_counter()
    for current in parts:
        include_m2 = False
        if current == "misspec":
            records = load_records(out)
            decision = m2_misspec_decision(cfg, records)
            summary["m2_misspec"] = decision
            if decision["include"] is None:
                m2_main = [s.run_id for s in plan_runs(cfg, "main") if s.kind == "fit" and s.model == "M2"]
                if not any(rid in records for rid in m2_main):
                    raise ValueError(
                        f"run part main first: no main-study M2 fit record in {out} (for example "
                        f"{m2_main[0]}); with shards, copy every shard's main records into one folder "
                        "before part misspec"
                    )
                log(
                    f"M2 in the misspecification study: undecided ({decision['reason']}); running the M1 "
                    "and profile runs only. Rerun part misspec after every main M2 fit has finished to "
                    "add the M2 runs."
                )
            else:
                include_m2 = bool(decision["include"])
                log(f"M2 in the misspecification study: {include_m2} ({decision['reason']})")
        specs = shard_runs(plan_runs(cfg, current, include_m2), shard)
        if selected is not None:
            specs = [s for s in specs if s.run_id in selected]
        summary["planned"] += len(specs)
        hashes = {s.run_id: config_hash(cfg, s) for s in specs}
        existing = {s.run_id: _read_json(out / f"{s.run_id}.json") for s in specs}
        mismatched = [
            rid
            for rid, rec in existing.items()
            if rec is not None and rec.get("status") != "error" and rec.get("config_hash") != hashes[rid]
        ]
        if mismatched and not force:
            raise ValueError(
                f"config changed for run {mismatched[0]} ({len(mismatched)} run(s) in {out} differ in "
                "config_hash); use force=True (--force) to rerun them, or choose another out_dir"
            )
        for spec in specs:
            rid = spec.run_id
            previous = existing[rid]
            if previous is not None and previous.get("config_hash") == hashes[rid]:
                if previous.get("status") in DONE_STATUSES:
                    summary["skipped"] += 1
                    log(f"{rid}  skipped ({previous.get('status')})")
                    continue
            t0 = time.perf_counter()
            try:
                record = _check_runner_record(runner(cfg, spec))
            except Exception as exc:  # one failed run must not stop the study
                record = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "partial": _json_partial(getattr(exc, "partial", None)),
                    "records": [],
                }
            elapsed = time.perf_counter() - t0
            record = dict(record)
            wall = dict(record.get("wall_time_s") or {})
            wall.setdefault("total", elapsed)
            fit_cfg = cfg.m1_train if spec.model == "M1" else cfg.m2_train
            float64 = fit_cfg.float64 if spec.kind == "fit" else bool(jax.config.x64_enabled)
            record.update(
                run_id=rid,
                spec=asdict(spec),
                config_hash=hashes[rid],
                config_name=cfg.name,
                schema=SCHEMA_VERSION,
                wall_time_s=wall,
                meta=run_metadata(seeds=[spec.seed], dtype="float64" if float64 else "float32"),
            )
            _write_json_atomic(out / f"{rid}.json", record)
            summary["run"] += 1
            summary["errors"] += record["status"] == "error"
            line = f"{rid}  {record['status']}  {elapsed:.1f} s"
            if record["status"] == "error":
                line += ": " + " ".join(str(record.get("error", "")).split())[:300]
            log(line)
    summary["wall_time_s"] = time.perf_counter() - start
    return summary
