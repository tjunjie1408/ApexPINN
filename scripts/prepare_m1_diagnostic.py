"""Build an unexecuted diagnostics-only Colab notebook and code update. No model execution."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import tempfile
from pathlib import Path

from prepare_colab_patch import BASE_SHA, RESTORE

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import pack


def cell(kind: str, text: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(text)
        value.update(execution_count=None, outputs=[])
    return value


def prepare(output: Path) -> dict:
    if output.exists():
        raise ValueError("choose a new handoff directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        root = Path(temporary)
        payload, stage = root / "payload", root / "handoff"
        payload.mkdir()
        stage.mkdir()
        for directory, pattern in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml")):
            for source in sorted(Path(directory).rglob(pattern)):
                target = payload / source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        write_json(payload / "handoff.json", {"base_sha256": BASE_SHA, "compute_status": "NOT RUN"})
        archive = stage / "apexpinn_m1_diagnostic_v1.zip"
        pack(payload, [p for p in payload.rglob("*") if p.is_file()], archive, kind="input")
        patch_sha = digest(archive)
        notebook = json.loads(Path("notebooks/colab/train_folds_drive_parts.ipynb").read_text())
        notebook["cells"] = notebook["cells"][:10]
        notebook.setdefault("metadata", {}).setdefault("colab", {})["name"] = "diagnose_m1_offsets.ipynb"
        notebook["cells"][0] = cell(
            "markdown",
            """# ApexPINN: diagnose saved M1 models

Prepared, NOT RUN. Select a hosted L4 GPU. This notebook performs inference on saved models only.
It does not train, tune, run a restart oracle, change checkpoints or replace forecasts.

Post-result method `m1_offset_diagnostic/v1`, declared 2026-11-05: inspect both calibration and final
M1/F1/mid/main fits. Compare trained offsets with the least-squares offsets on the same training
labels; split residual error into offset and shape components. Report all five members and plot
fixed batch rows. No decision threshold or pivot is introduced. Profiled training errors are not
held-out scores. Preserve the completed unfavorable result and use a separate output directory.
""",
        )
        notebook["cells"][2] = cell("markdown", "## 2. Select saved inputs and a new diagnostic ID\n")
        notebook["cells"][4] = cell(
            "markdown",
            """## 3. Verify and restore the separate code workspace

Reuse the original approved input ZIP parts and verify the diagnostic update independently.
Completed run files stay on Drive and are read only. New reports go under `apexpinn/diagnostics`.
""",
        )
        notebook["cells"][3] = cell(
            "code",
            f"""DRIVE_ROOT = Path('/content/drive/MyDrive/apexpinn')
INPUT_BUNDLE = DRIVE_ROOT / 'inputs/apexpinn_t13_input_v1.zip'
BUNDLE_SHA256 = '{BASE_SHA}'
OVERLAY_BUNDLE = DRIVE_ROOT / 'inputs/{archive.name}'
OVERLAY_SHA256 = '{patch_sha}'
SOURCE_RUN = DRIVE_ROOT / 'runs/t13_m1_f1_mid_patch01'
DIAGNOSTIC_ID = 'm1_offsets_f1_mid_v1'
OUTPUT = DRIVE_ROOT / 'diagnostics' / DIAGNOSTIC_ID
if not DIAGNOSTIC_ID or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in DIAGNOSTIC_ID):
    raise ValueError('Use a simple, unique diagnostic ID.')
if not (SOURCE_RUN / 'jobs/M1_F1_mid_main/job.json').is_file():
    raise FileNotFoundError('The completed M1 run is required on Drive.')
""",
        )
        bootstrap = "".join(notebook["cells"][7]["source"]).split("WORKSPACE =", 1)[0]
        notebook["cells"][7] = cell("code", bootstrap + RESTORE)
        notebook["cells"][8] = cell(
            "markdown",
            """## 4. Restore the locked Python 3.12 GPU environment

This verifies GPU and float64, without optimizer updates. The diagnostic also checks original
package versions, source bytes, Gold identity, model/checkpoint hashes and training offsets.
""",
        )
        notebook["cells"] += [
            cell(
                "markdown",
                """## 5. Run the saved-model diagnostic

This runs model inference on Colab. It deserializes optimizer state to recover the saved offsets,
but never computes gradients or updates the optimizer. Ten completed fits are inspected.
An existing output directory is refused. A partial diagnostic needs a new ID, not a training rerun.
Full process output and exit status persist under `apexpinn/logs/DIAGNOSTIC_ID`.
""",
            ),
            cell(
                "code",
                """LOG_ROOT = DRIVE_ROOT / 'logs' / DIAGNOSTIC_ID
command = [str(PYTHON), 'scripts/diagnose_m1.py', '--run', str(SOURCE_RUN), '--out', str(OUTPUT)]
subprocess.run([str(PYTHON), '-m', 'apexpinn.cloud.process', '--cwd', str(WORKSPACE),
                '--logs', str(LOG_ROOT), '--', *command], check=True)
""",
            ),
            cell(
                "markdown",
                """## 6. Read the separate report

A large offset contribution locates training pace error. Remaining profiled error locates shape
misfit. Neither establishes that a changed method will improve held-out forecasts. Original M1
scores remain unchanged. No output has been computed in this prepared notebook.
""",
            ),
            cell(
                "code",
                """from IPython.display import Image, Markdown, display
display(Markdown((OUTPUT / 'diagnostic.md').read_text()))
display(Image(filename=str(OUTPUT / 'offset_errors.png')))
display(Image(filename=str(OUTPUT / 'example_curve.png')))
""",
            ),
            cell("markdown", "## 7. Export diagnostics for review\nExport only the new diagnostic folder.\n"),
            cell(
                "code",
                """if not (OUTPUT / 'diagnostic.json').is_file():
    raise RuntimeError('No completed diagnostic receipt; inspect persistent logs.')
shutil.copytree(LOG_ROOT, OUTPUT / 'execution', dirs_exist_ok=True)
EXPORT = DRIVE_ROOT / 'exports' / (DIAGNOSTIC_ID + '_results_01.zip')
EXPORT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([str(PYTHON), '-m', 'apexpinn.cloud.transfer', 'pack-results',
                '--root', str(OUTPUT), '--out', str(EXPORT)], cwd=WORKSPACE, check=True)
print('Result archive:', EXPORT)
print('Receipt:', str(EXPORT) + '.receipt.json')
print('SHA-256:', file_sha(EXPORT))
""",
            ),
        ]
        for item in notebook["cells"]:
            if item["cell_type"] == "code":
                item["execution_count"], item["outputs"] = None, []
                ast.parse("".join(item["source"]))
        target = stage / "diagnose_m1_offsets.ipynb"
        target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        write_json(
            stage / "handoff_receipt.json",
            {
                "schema": "apexpinn.m1_diagnostic_handoff/v1",
                "compute_status": "NOT RUN",
                "archive_sha256": patch_sha,
                "notebook_sha256": digest(target),
                "base_sha256": BASE_SHA,
                "source_run": "t13_m1_f1_mid_patch01",
                "training": False,
                "optimizer_updates": 0,
            },
        )
        stage.rename(output)
    return {"notebook": str(output / target.name), "archive_sha256": patch_sha, "compute_status": "NOT RUN"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
