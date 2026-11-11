"""Prepare a small source update and an unexecuted M1 notebook using existing Drive inputs."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import tempfile
from pathlib import Path

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import pack

BASE_SHA = "b787682f95e30b303bfe7a41a2739dd40051a200724f079ffcab84a7d5618a03"

RESTORE = """import importlib.util
from apexpinn.cloud.transfer import unpack

if file_sha(OVERLAY_BUNDLE) != OVERLAY_SHA256:
    raise ValueError('Update archive hash mismatch')
OVERLAY_ROOT = Path('/content') / ('apexpinn_update_' + OVERLAY_SHA256[:12])
if not OVERLAY_ROOT.exists():
    unpack(OVERLAY_BUNDLE, OVERLAY_ROOT, expected_sha256=OVERLAY_SHA256, kind='input')
else:
    import zipfile
    with zipfile.ZipFile(OVERLAY_BUNDLE) as update:
        expected_overlay = json.loads(update.read('bundle_manifest.json'))
    if json.loads((OVERLAY_ROOT / 'bundle_manifest.json').read_text()) != expected_overlay:
        raise ValueError('Update manifest changed')
    for name, item in expected_overlay['files'].items():
        if file_sha(OVERLAY_ROOT / name) != item['sha256']:
            raise ValueError('Update workspace changed')
spec = importlib.util.spec_from_file_location(
    'apexpinn_handoff', OVERLAY_ROOT / 'src/apexpinn/cloud/handoff.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)
WORKSPACE = Path('/content') / ('apexpinn_' + BUNDLE_SHA256[:8] + '_' + OVERLAY_SHA256[:12])
handoff.restore_overlay(INPUT_BUNDLE, OVERLAY_ROOT, WORKSPACE, base_sha256=BUNDLE_SHA256)
print('Verified updated workspace:', WORKSPACE)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="v2")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new handoff directory")
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.tag):
        parser.error("tag must contain lowercase letters, digits, underscore or hyphen")
    declaration = json.loads((args.original_run / "declaration.json").read_text())
    recovered = json.loads(args.recovery.read_text())
    if recovered["status"] != "RECOVERED_EVALUATION_COMPLETE":
        raise ValueError("completed recovered evaluation required")
    if digest(args.original_run / "declaration.json") != recovered["provenance"]["declaration_sha256"]:
        raise ValueError("original run declaration differs from recovered evaluation")
    forecast = args.original_run / "jobs" / recovered["job"] / "forecasts.jsonl"
    if digest(forecast) != recovered["provenance"]["forecast_sha256"]:
        raise ValueError("saved forecasts differ from recovered evaluation")
    recovered["provenance"]["config_sha256"] = declaration["identity"]["config_sha256"]
    recovered["provenance"]["numerical_source_sha256"] = {
        name: sha
        for name, sha in declaration["identity"]["source_sha256"].items()
        if name.startswith(("src/apexpinn/physics/", "src/apexpinn/models/", "src/apexpinn/data/"))
        or name
        in (
            "src/apexpinn/eval/metrics.py",
            "src/apexpinn/eval/protocol.py",
            "src/apexpinn/eval/bootstrap.py",
            "src/apexpinn/train/trainer.py",
            "src/apexpinn/train/physics_study.py",
        )
    }
    recovered["provenance"]["recovered_report_source_sha256"] = digest(args.recovery)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        stage = Path(temporary) / "handoff"
        stage.mkdir()
        payload = Path(temporary) / "payload"
        payload.mkdir(parents=True)
        for directory, pattern in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml")):
            for source in Path(directory).rglob(pattern):
                target = payload / source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        recovery_folder = payload / "recovered_evaluations" / recovered["job"]
        recovery_folder.mkdir(parents=True)
        write_json(recovery_folder / "review.json", recovered)
        shutil.copyfile(forecast, recovery_folder / "forecasts.jsonl")
        write_json(payload / "handoff.json", {"base_sha256": BASE_SHA, "compute_status": "NOT RUN"})
        archive = stage / f"apexpinn_t13_code_update_{args.tag}.zip"
        pack(payload, [p for p in payload.rglob("*") if p.is_file()], archive, kind="input")
        overlay_sha = digest(archive)
        notebook = json.loads(Path("notebooks/colab/train_folds_drive_parts.ipynb").read_text())
        settings = "".join(notebook["cells"][3]["source"])
        settings = settings.replace("RUN_ID = 't13_cloud_v1_run01'", "RUN_ID = 't13_m1_f1_mid_patch01'")
        settings = settings.replace("MODELS = ['M2']", "MODELS = ['M1']")
        settings += (
            f"\nOVERLAY_BUNDLE = DRIVE_ROOT / 'inputs/{archive.name}'\nOVERLAY_SHA256 = '{overlay_sha}'\n"
        )
        notebook["cells"][3]["source"] = settings.splitlines(keepends=True)
        original = "".join(notebook["cells"][7]["source"])
        bootstrap = original.split("WORKSPACE =", 1)[0]
        notebook["cells"][7]["source"] = (bootstrap + RESTORE).splitlines(keepends=True)
        command = "".join(notebook["cells"][11]["source"])
        command = command.replace(
            "subprocess.run(command + ['--plan']",
            (
                "RECOVERED_REVIEW = WORKSPACE / 'recovered_evaluations/M2_F1_mid_main/review.json'\n"
                "command += ['--recovered-review', str(RECOVERED_REVIEW)]\n"
                "subprocess.run(command + ['--plan']"
            ),
        )
        notebook["cells"][11]["source"] = command.splitlines(keepends=True)
        notebook["cells"][13][
            "source"
        ] = """SELFTEST_ROOT = DRIVE_ROOT / 'checks' / (RUN_ID + '_' + OVERLAY_SHA256[:12])
if (SELFTEST_ROOT / 'selftest.json').exists():
    print((SELFTEST_ROOT / 'selftest.json').read_text())
else:
    check_command = [str(PYTHON), 'scripts/run_models.py', '--out', str(SELFTEST_ROOT), '--self-test']
    subprocess.run([str(PYTHON), '-m', 'apexpinn.cloud.process', '--cwd', str(WORKSPACE),
                    '--logs', str(DRIVE_ROOT / 'logs' / RUN_ID / 'restart'), '--', *check_command],
                   check=True)
""".splitlines(keepends=True)
        notebook["cells"][15]["source"] = """LOG_ROOT = DRIVE_ROOT / 'logs' / RUN_ID / 'training'
subprocess.run([str(PYTHON), '-m', 'apexpinn.cloud.process', '--cwd', str(WORKSPACE),
                '--logs', str(LOG_ROOT), '--', *command], check=True)
""".splitlines(keepends=True)
        notebook["cells"][0]["source"] = (
            "# ApexPINN T13 prepared M1 comparison\n\n"
            "Select a hosted L4 GPU. Run the setup and restart check only after your execution order.\n"
            "The training cell selects M1/F1/mid/main. The recovered M2 result is included in collection.\n"
            "This handoff reuses the existing input ZIP parts plus a separate hash-verified code update.\n"
            "Full stdout/stderr and exit records are saved under MyDrive/apexpinn/logs/RUN_ID.\n"
        ).splitlines(keepends=True)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"], cell["execution_count"] = [], None
                ast.parse("".join(cell["source"]))
        target = stage / "train_next_m1.ipynb"
        target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        stage.rename(args.out)
    print(args.out / "train_next_m1.ipynb")
    print(f"Update SHA-256: {overlay_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
