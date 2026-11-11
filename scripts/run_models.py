"""T13 remote runner. --plan and --report-only never fit; execution requires a verified remote GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=Path, default=Path("configs/t13_cloud_v1.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/decisions_v1.yaml"))
    parser.add_argument("--gold-root", type=Path, default=Path("data/gold_development_frozen_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=("M1", "M2"))
    parser.add_argument("--fold", action="append")
    parser.add_argument("--phi", action="append", choices=("low", "mid", "high"))
    parser.add_argument(
        "--variant", action="append", choices=("main", "no_thermal", "no_wear", "constant_q", "no_window")
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--recovered-review", action="append", type=Path, default=[])
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    import jax

    jax.config.update("jax_enable_x64", True)
    from apexpinn.cloud.artifacts import digest, write_json
    from apexpinn.contracts import load_decisions
    from apexpinn.train.cloud_checkpoint import require_remote_gpu
    from apexpinn.train.physics_study import VARIANTS, read_method

    method, decisions = read_method(args.method), load_decisions(args.config)
    models, folds, labels, variants = (
        args.model or ["M1", "M2"],
        args.fold or list(decisions.folds),
        args.phi or ["low", "mid", "high"],
        args.variant or list(VARIANTS),
    )
    if any(f not in decisions.folds for f in folds) or any(
        len(set(x)) != len(x) for x in (models, folds, labels, variants)
    ):
        parser.error("models/folds/phi/variants must be unique configured values")
    jobs = [
        (m, f, p, v)
        for m in models
        for f in folds
        for p in labels
        for v in VARIANTS
        if v in variants and (m == "M1" or v == "main")
    ]
    if not jobs:
        parser.error("selection contains no applicable jobs")
    if args.plan:
        print(
            json.dumps(
                {
                    "status": "NOT RUN",
                    "jobs": jobs,
                    "tuning_trials": 20 * sum(v == "main" for _, _, _, v in jobs),
                    "calibration_fits": 5 * len(jobs),
                    "final_fits": 5 * len(jobs),
                    "execution": method["training_location"],
                    "resume": "same output path and unchanged identity",
                },
                indent=2,
            )
        )
        return 0
    if args.report_only:
        from apexpinn.eval.physics_report import collect_results

        return collect_results(
            args.out, recovered_reviews=tuple(args.recovered_review), output=args.report_out
        )
    require_remote_gpu()
    import os

    expected_runtime = {
        "apexpinn.t13_cloud/v1": "colab",
        "apexpinn.t13_sagemaker/v1": "sagemaker",
        "apexpinn.t13_studio/v1": "studio",
        "apexpinn.t13_ec2/v1": "ec2",
        "apexpinn.t13_kaggle/v1": "kaggle",
    }[method["method_id"]]
    if os.environ.get("APEXPINN_REMOTE_RUNTIME") != expected_runtime:
        raise ValueError("declared method and actual remote runtime differ")
    if args.self_test:
        from apexpinn.train.cloud_selftest import checkpoint_selftest

        checkpoint_selftest(args.out)
        return 0
    import importlib.metadata

    import pandas as pd

    from apexpinn.train.physics_study import run_job

    g0_path = Path("experiments/g0_development_20261103/acceptance.json")
    g1_path = Path("experiments/g1_real_20261103/acceptance.json")
    g0, g1 = json.loads(g0_path.read_text()), json.loads(g1_path.read_text())
    if decisions.status != "frozen" or g1["conditions"]["G1"] != "PASS":
        raise ValueError("frozen decisions and accepted G0/G1 evidence required")
    if digest(args.config) != g0["instrument"]["current_config_sha256"]:
        raise ValueError("frozen config differs from G0")
    if digest(Path("experiments/g1_real_20261103/review.json")) != g1["instrument"]["review_sha256"]:
        raise ValueError("G1 review identity mismatch")
    prerequisites = (Path("reports/s1_identifiability.md"), Path("experiments/pivots.md"))
    if any(not path.is_file() for path in prerequisites):
        raise ValueError("S1 report and user P1 decision are required before T13 training")
    if "## P1 (status: decided, R2)" not in prerequisites[1].read_text():
        raise ValueError("this method requires the recorded user P1=R2 decision")
    inputs = {}
    for label in ("low", "mid", "high"):
        manifest_path = args.gold_root / f"phi_{label}" / "manifest.json"
        if digest(manifest_path) != g0["phi"][label]["gold_manifest_sha256"]:
            raise ValueError("Gold manifest differs from accepted input")
        manifest = json.loads(manifest_path.read_text())
        for fold, meta in manifest["folds"].items():
            path = manifest_path.parent / fold / "laps.parquet"
            expected = meta["files"]["laps.parquet"]["sha256"]
            if digest(path) != expected:
                raise ValueError("Gold payload hash mismatch")
            inputs[f"{label}/{fold}"] = expected
    sources = {
        str(p).replace("\\", "/"): digest(p)
        for base in (Path("src/apexpinn"), Path("scripts"))
        for p in sorted(base.rglob("*.py"))
    }
    run_id = {
        "method_sha256": digest(args.method),
        "config_sha256": digest(args.config),
        "gold_sha256": inputs,
        "source_sha256": sources,
        "G0_sha256": digest(g0_path),
        "G1_sha256": digest(g1_path),
        "prerequisites_sha256": {path.as_posix(): digest(path) for path in prerequisites},
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("jax", "jaxlib", "equinox", "optax", "diffrax", "numpy", "pandas")
        },
        "devices": [str(d) + ":" + d.device_kind for d in jax.devices()],
    }
    declaration_path = args.out / "declaration.json"
    if declaration_path.exists():
        if json.loads(declaration_path.read_text())["identity"] != run_id:
            raise ValueError("run code/data/method/backend changed; choose a new output directory")
    else:
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("nonempty output has no compatible declaration")
        write_json(
            declaration_path,
            {
                "schema": "apexpinn.t13_declaration/v1",
                "identity": run_id,
                "method": method,
                "G2": "NOT RUN",
                "population": "development only",
                "S1": "preserved, no rerun",
            },
        )
        snapshot = args.out / "source"
        for filename in sources:
            path = snapshot / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(filename).read_bytes())
        (snapshot / "t13_method.yaml").write_bytes(args.method.read_bytes())
        (snapshot / "decisions_v1.yaml").write_bytes(args.config.read_bytes())
    # Frames are read only. Inner normalization is refitted inside each job, excluding both test groups.
    frames = {label: pd.read_parquet(args.gold_root / f"phi_{label}" / "F1/laps.parquet") for label in labels}
    for frame in frames.values():
        if not frame.dataset_split.eq("development").all():
            raise ValueError("temporal holdout forbidden")
    for name, fold, label, variant in jobs:
        result = run_job(frames[label], decisions, method, args.out, name, fold, label, variant, run_id)
        print(
            json.dumps(
                {"job": [name, fold, label, variant], "status": "finished", "metrics": len(result["metrics"])}
            ),
            flush=True,
        )
    from apexpinn.eval.physics_report import collect_results

    # A shard remains a valid completed shard; completeness is reported without claiming G2.
    collect_results(args.out, recovered_reviews=tuple(args.recovered_review), output=args.report_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
