"""Prepare a separate clean-calibration Colab handoff. No model fitting or remote execution."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import tempfile
from pathlib import Path

from prepare_colab_patch import BASE_SHA, RESTORE
from prepare_m1_diagnostic import cell

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import pack
from apexpinn.train.calibration_study import read_spec


def prepare(output: Path) -> dict:
    if output.exists():
        raise ValueError("choose a new handoff directory")
    method = read_spec(Path("configs/calibration_clean_v1.yaml"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        payload, stage = Path(tmp) / "payload", Path(tmp) / "handoff"
        payload.mkdir()
        stage.mkdir()
        for directory, pattern in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml")):
            for source in sorted(Path(directory).rglob(pattern)):
                target = payload / source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        write_json(payload / "handoff.json", {"base_sha256": BASE_SHA, "compute_status": "NOT RUN"})
        archive = stage / "apexpinn_calibration_clean_v1.zip"
        manifest = pack(payload, [p for p in payload.rglob("*") if p.is_file()], archive, kind="input")
        archive_sha = digest(archive)
        notebook = json.loads(Path("notebooks/colab/train_folds_drive_parts.ipynb").read_text())
        notebook["cells"] = notebook["cells"][:10]
        notebook["cells"][0] = cell(
            "markdown",
            """# ApexPINN clean M2 calibration pilot v1

Prepared, NOT RUN. Keep the existing F3 v1 training unchanged. Run this separate pilot after F3
has finished, on hosted Colab L4 with Python 3.12 and float64. This starts 25 new fits when launched:
20 tuning trials plus five final members, 1000 steps each. Duration and resource use are Unknown.

Post-hoc method declared 2026-11-06, prompted by v1 F1/F2 coverage differences. F3 v1 results were
not used to choose this method. Canonical declaration: configs/calibration_clean_v1.yaml.
For outer F3: tune on F4, reserve F5 for calibration, train tuning candidates on F1/F2/F6, then
fit the five final members on F1/F2/F4/F6. Never refit them after calibration. All four horizons
get separate event-balanced interval scales. No new fit is counted as a T13 v1 completion.

Raw/calibrated forecasts have identical means and targets and exact Gaussian-mixture variance.
Coverage, width and CRPS are reported per horizon and circuit. No coverage guarantee under circuit
shift or dependent laps is claimed. Comparing this pilot with v1 also changes its training data
and raw-noise estimator. No S1, v1 score, threshold, G2 or P2 decision is changed.
""",
        )
        notebook["cells"][2] = cell("markdown", "## 2. Select a new, separate pilot output\n")
        notebook["cells"][3] = cell(
            "code",
            f"""DRIVE_ROOT = Path('/content/drive/MyDrive/apexpinn')
INPUT_BUNDLE = DRIVE_ROOT / 'inputs/apexpinn_t13_input_v1.zip'
BUNDLE_SHA256 = '{BASE_SHA}'
OVERLAY_BUNDLE = DRIVE_ROOT / 'inputs/{archive.name}'
OVERLAY_SHA256 = '{archive_sha}'
PILOT_ID = 'calib_clean_m2_f3_mid_v1'
RUN_ROOT = DRIVE_ROOT / 'calibration_runs' / PILOT_ID
EXPORT_TAG = 'checkpoint_01'
FOLD = 'F3'
PHI = 'mid'
F3_JOB = DRIVE_ROOT / 'runs/t13_m2_f3_mid_run01/jobs/M2_F3_mid_main/job.json'
""",
        )
        notebook["cells"][4] = cell(
            "markdown",
            """## 3. Restore a separate verified workspace

Reuse the original input parts and a separately hashed code update. This workspace and its
checkpoints differ from the running F3 v1 workspace. No existing run directory is modified.
""",
        )
        bootstrap = "".join(notebook["cells"][7]["source"]).split("WORKSPACE =", 1)[0]
        notebook["cells"][7] = cell("code", bootstrap + RESTORE)
        notebook["cells"][8] = cell(
            "markdown",
            """## 4. Install locked Python and verify L4/float64

This cell performs setup and device checks, not training. Actual device details and any error are
printed and saved to the persistent preflight log, before training can start.
""",
        )
        install = "".join(notebook["cells"][9]["source"]).split("subprocess.run([str(PYTHON), '-c',", 1)[0]
        probe = """import json
import jax
jax.config.update('jax_enable_x64', True)
devices = [{'platform': d.platform, 'kind': d.device_kind} for d in jax.devices()]
print(json.dumps({'devices': devices, 'float64': jax.config.x64_enabled}), flush=True)
from apexpinn.train.cloud_checkpoint import require_colab_gpu
require_colab_gpu()
if not all('L4' in d['kind'].upper() for d in devices):
    raise RuntimeError('Select hosted L4. Detected: ' + str(devices))
"""
        install += f"\nGPU_PROBE = {probe!r}\n"
        install += """probe = subprocess.run(
    [str(PYTHON), '-m', 'apexpinn.cloud.process', '--cwd', str(WORKSPACE),
     '--logs', str(DRIVE_ROOT / 'logs' / PILOT_ID / 'preflight'), '--', str(PYTHON), '-c', GPU_PROBE],
    cwd=WORKSPACE, capture_output=True, text=True)
print(probe.stdout, end='')
print(probe.stderr, end='')
probe.check_returncode()
"""
        notebook["cells"][9] = cell("code", install)
        notebook["cells"] += [
            cell(
                "markdown",
                """## 5. Review the clean split and new fit budget

This prints the plan only. Calibration and test circuits must be absent from tuning and fitting.
Twenty search trials and five final members are a new experiment, not retries of completed v1 slots.
""",
            ),
            cell(
                "code",
                """command = [str(PYTHON), 'scripts/run_calibration.py', '--out', str(RUN_ROOT),
           '--fold', FOLD, '--phi', PHI]
subprocess.run(command + ['--plan'], cwd=WORKSPACE, check=True)
""",
            ),
            cell(
                "markdown",
                """## 6. Verify the existing trainer restart evidence

The unchanged numerical training engine already passed the v4 restart oracle. This checks those
engine hashes and does not run new optimizer updates. It does not claim the new orchestration
has been exercised end-to-end; the clean pilot itself remains NOT RUN until you launch it.
""",
            ),
            cell(
                "code",
                """oracle_path = DRIVE_ROOT / 'checks/t13_m1_f1_mid_patch01_0b6d9d66c3ba/selftest.json'
oracle = json.loads(oracle_path.read_text())
if not all(v.startswith('PASS:') for v in oracle['results'].values()):
    raise ValueError('Original trainer restart parity did not pass')
engine = {name: sha for name, sha in oracle['instrument']['source_sha256'].items()
          if name.startswith(('src/apexpinn/models/', 'src/apexpinn/physics/', 'src/apexpinn/data/'))
          or name in ('src/apexpinn/train/trainer.py', 'src/apexpinn/train/cloud_checkpoint.py',
                      'src/apexpinn/train/physics_study.py', 'src/apexpinn/eval/protocol.py',
                      'src/apexpinn/eval/prepared.py', 'src/apexpinn/eval/metrics.py')}
if not engine or any(file_sha(WORKSPACE / name) != sha for name, sha in engine.items()):
    raise ValueError('Numerical engine differs from the verified restart source')
print('Existing trainer restart evidence verified; no restart fits launched.')
""",
            ),
            cell(
                "markdown",
                """## 7. Launch the clean pilot after F3 finishes

This cell starts real training on Colab. It refuses to launch before the completed F3 v1 job record
exists. Checkpoints and trial slots persist on Drive; rerun with compatible unchanged identity to
resume. No post-calibration refit is performed. Completed forecast bundles are reused without replay.
""",
            ),
            cell(
                "code",
                """if not F3_JOB.is_file():
    raise RuntimeError('Keep F3 v1 running. Wait for its completed job record before this pilot.')
finished = json.loads(F3_JOB.read_text())
if (finished.get('schema') != 'apexpinn.t13_job/v1'
        or finished['conditions']['model'] != 'M2' or finished['population']['fold'] != 'F3'
        or len(finished['final_fits']) != 5):
    raise ValueError('The expected completed F3 v1 job was not found')
subprocess.run([str(PYTHON), '-m', 'apexpinn.cloud.process', '--cwd', str(WORKSPACE),
                '--logs', str(DRIVE_ROOT / 'logs' / PILOT_ID / 'training'), '--', *command], check=True)
""",
            ),
            cell(
                "markdown",
                """## 8. Review and export the separate result

The numerical evaluation is committed before rendering. If plotting failed after evaluation.json
was saved, this cell rebuilds the report without fitting or model inference. Original v1 results
stay unchanged. Compare interval coverage with width and CRPS, not pooled coverage alone.
""",
            ),
            cell(
                "code",
                """from IPython.display import Image, Markdown, display
if not (RUN_ROOT / 'evaluation.json').is_file():
    raise RuntimeError('No completed evaluation yet. Inspect the persistent process logs.')
if not (RUN_ROOT / 'report_receipt.json').is_file():
    subprocess.run(command + ['--report-only'], cwd=WORKSPACE, check=True)
display(Markdown((RUN_ROOT / 'calibration_report.md').read_text()))
display(Image(filename=str(RUN_ROOT / 'calibration_intervals.png')))
EXPORT = DRIVE_ROOT / 'exports' / (PILOT_ID + '_' + EXPORT_TAG + '.zip')
EXPORT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([str(PYTHON), '-m', 'apexpinn.cloud.transfer', 'pack-results',
                '--root', str(RUN_ROOT), '--out', str(EXPORT)], cwd=WORKSPACE, check=True)
print('Result archive:', EXPORT)
print('Receipt:', str(EXPORT) + '.receipt.json')
print('SHA-256:', file_sha(EXPORT))
""",
            ),
        ]
        notebook["metadata"].setdefault("colab", {})["name"] = "calibration_clean_v1.ipynb"
        for item in notebook["cells"]:
            if item["cell_type"] == "code":
                ast.parse("".join(item["source"]))
                item["execution_count"], item["outputs"] = None, []
        target = stage / "calibration_clean_v1.ipynb"
        target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        write_json(
            stage / "handoff_receipt.json",
            {
                "schema": "apexpinn.clean_calibration_handoff/v1",
                "compute_status": "NOT RUN",
                "method_id": method["method_id"],
                "method_sha256": digest(Path("configs/calibration_clean_v1.yaml")),
                "archive_sha256": archive_sha,
                "notebook_sha256": digest(target),
                "payloads": len(manifest["files"]),
                "base_sha256": BASE_SHA,
                "new_base_fits_when_launched": 25,
                "existing_F3_modified": False,
            },
        )
        stage.rename(output)
    return {"notebook": str(output / target.name), "archive_sha256": archive_sha, "compute_status": "NOT RUN"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
