"""Prepare one-job-at-a-time F4-F6 notebooks, reusing the verified v4 code bundle. No fits."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from prepare_m1_diagnostic import cell

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.eval.review_figures import budget_plan


def prepare(out: Path, *, fold: str = "F4") -> dict:
    if out.exists() or fold not in ("F4", "F5", "F6"):
        raise ValueError("new output and one remaining mid fold F4/F5/F6 required")
    source = Path("notebooks/colab/train_m2_f3_mid.ipynb")
    notebook = json.loads(source.read_text())
    notebook["cells"][0] = cell(
        "markdown",
        """# ApexPINN: remaining M2/mid folds, one job at a time

Prepared, NOT RUN. The user launches this notebook on hosted Colab L4. Default: F4 only.
After exporting and reviewing F4, change NEXT_FOLD to F5, then F6 in separate executions.
Each fold has its own persistent run identity and unchanged T13 v1 budget: 20 tuning trials,
five inner calibration fits and five final members, 1000 updates each. No clean-pilot refit,
additional search, M1 ablation or restart-oracle fitting is launched here.

Review the figures before spending more compute. The full T13 matrix remains incomplete;
prioritizing these three folds does not remove its remaining requirements or establish G2/P2.
The completed clean-calibration pilot and original F1-F3 records stay unchanged.
""",
    )
    notebook["cells"][2] = cell(
        "markdown",
        """## 2. Choose one fold and refresh your budget estimate

Edit NEXT_FOLD to exactly one of F4/F5/F6. Re-enter your actual current compute-unit balance
before each new fold. Use only one runtime for a given run ID. Never run another copy because
the browser timer is stale. An existing completed job is reused instead of fitted again.

Your reported balance/rate is 100 units at 1.94 units/hour, or about 51.55 allocated L4 hours.
The editable 20-unit reserve leaves about 41.24 hours. Assumed per-fold durations below are
planning scenarios, not measured bounds or confidence intervals:

| Hours/fold | Three-fold hours | Units | Units left from 100 |
| --- | --- | --- | --- |
| 8 | 24 | 46.56 | 53.44 |
| 12 | 36 | 69.84 | 30.16 |
| 16 | 48 | 93.12 | 6.88 |

F2 v1 recorded 9.11 process hours. F3's 5.61-hour successful attempt excludes its earlier attempt;
the clean pilot's 10.58-hour successful attempt uses a different split and 25 fits. Neither is a
whole-study upper bound for a new fold. Setup, compilation, interruptions and idle allocated time
can add cost. This budget check cannot enforce a billing cap or guarantee credits are sufficient.
""",
    )
    settings = "".join(notebook["cells"][3]["source"])
    settings = settings.replace(
        "RUN_ID = 't13_m2_f3_mid_run01'",
        """NEXT_FOLD = '__FOLD__'
if NEXT_FOLD not in ('F4', 'F5', 'F6'):
    raise ValueError('Select one remaining fold: F4, F5 or F6')
RUN_ID = f't13_m2_{NEXT_FOLD.lower()}_mid_run01'
AVAILABLE_UNITS = 100.0  # Refresh from your account before EACH new fold.
UNITS_PER_HOUR = 1.94
RESERVE_UNITS = 20.0    # Editable operating preference; not a research parameter.
PLANNING_HOURS_PER_JOB = 12.0  # Assumed; not a measured upper bound.
if not (AVAILABLE_UNITS >= 0 and UNITS_PER_HOUR > 0 and RESERVE_UNITS >= 0
        and PLANNING_HOURS_PER_JOB > 0):
    raise ValueError('Check budget inputs')
print('One selected job:', RUN_ID)
print('Planning cost:', PLANNING_HOURS_PER_JOB * UNITS_PER_HOUR, 'units')
print('Above-reserve hours:', (AVAILABLE_UNITS - RESERVE_UNITS) / UNITS_PER_HOUR)""".replace(
            "__FOLD__", fold
        ),
    )
    settings = settings.replace("FOLDS = ['F3']", "FOLDS = [NEXT_FOLD]")
    notebook["cells"][3] = cell("code", settings)
    notebook["cells"][14] = cell(
        "markdown",
        """## 7. Launch exactly the selected fold

This starts real training. Check the plan above: M2, selected F4/F5/F6, mid, main only.
The numerical engine and v4 source stay unchanged. The existing source-matched restart receipt
is reused; no extra restart fits. A same-VM nonblocking process lock prevents another launch
for the same run ID while the first process still holds it. It does not coordinate different VMs.

Checkpoints and process records persist on Drive. Observe progress.json and checkpoint steps;
the Colab cell timer is not a reliable elapsed-time source. Open .log files can appear late through
the Drive API. Resume with identical source/data/environment/run identity after an interruption;
do not change trial budgets, treat completed slots as fresh, or overwrite old exports.
""",
    )
    notebook["cells"][15] = cell(
        "code",
        """JOB = RUN_ROOT / 'jobs' / f'M2_{NEXT_FOLD}_mid_main' / 'job.json'
if JOB.is_file():
    print('Completed selected job already exists. Continue to verification/export; no fitting launched.')
else:
    if AVAILABLE_UNITS - RESERVE_UNITS < PLANNING_HOURS_PER_JOB * UNITS_PER_HOUR:
        raise RuntimeError('Planning cost exceeds units above reserve. Review balance before starting.')
    FLOCK = shutil.which('flock')
    if not FLOCK:
        raise RuntimeError('Hosted Linux flock is required for same-VM duplicate-launch protection.')
    LOG_ROOT = DRIVE_ROOT / 'logs' / RUN_ID / 'training'
    lock = Path('/content') / (RUN_ID + '.launch.lock')
    subprocess.run([FLOCK, '--nonblock', str(lock), str(PYTHON), '-m', 'apexpinn.cloud.process',
                    '--cwd', str(WORKSPACE), '--logs', str(LOG_ROOT), '--', *command], check=True)
""",
    )
    export = "".join(notebook["cells"][17]["source"])
    prefix = """JOB = RUN_ROOT / 'jobs' / f'M2_{NEXT_FOLD}_mid_main' / 'job.json'
if not JOB.is_file():
    raise RuntimeError('Selected job incomplete. Inspect its persistent logs/checkpoints.')
job = json.loads(JOB.read_text())
declaration = json.loads((RUN_ROOT / 'declaration.json').read_text())
if (job['run_identity'] != declaration['identity']
        or job['conditions']['model'] != 'M2' or job['population']['fold'] != NEXT_FOLD
        or job['conditions']['phi'] != 0.06 or job['conditions']['variant'] != 'main'
        or len(job['trials']) != 20 or len(job['calibration_fits']) != 5 or len(job['final_fits']) != 5):
    raise ValueError('Completed selected job contract differs')
for name, expected in job['files'].items():
    if file_sha(JOB.parent / name) != expected:
        raise ValueError('Saved job artifact changed: ' + name)
"""
    notebook["cells"][17] = cell("code", prefix + export)
    notebook["cells"][-1] = cell(
        "markdown",
        """## 9. Export, review, then release the runtime

Return the versioned ZIP and receipt for verification. This per-shard report may include recovered
M2/F1 again; do not sum its collector count with earlier exports. Training completion is not G2.
After the export is safely on Drive, use Colab's Disconnect and delete runtime when finished to avoid
idle credit use. No automatic runtime allocation/release or next-fold launch is performed.

Only after reviewing the previous result and your actual remaining balance, select the next fold
in the settings. The persistent output is unique to each fold. Increment EXPORT_TAG for another
export of the same run; never overwrite old evidence. Full T13/H2/G2/P2 and later stages remain open.
""",
    )
    notebook["metadata"].setdefault("colab", {})["name"] = "train_m2_remaining_mid.ipynb"
    for i, c in enumerate(notebook["cells"]):
        c.setdefault("id", f"remaining-mid-{i:02d}")
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
            c["outputs"], c["execution_count"] = [], None
    out.mkdir(parents=True)
    target = out / "train_m2_remaining_mid.ipynb"
    target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    receipt = {
        "status": "PREPARED_NOT_RUN",
        "default_fold": fold,
        "selectable_folds": ["F4", "F5", "F6"],
        "jobs_per_launch": 1,
        "template_sha256": digest(source),
        "notebook_sha256": digest(target),
        "source_overlay": "Existing apexpinn_t13_code_update_v4.zip; no new training source",
        "training_launched": False,
        "extra_restart_fits": 0,
        "budget": budget_plan(),
    }
    write_json(out / "handoff_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out)))


if __name__ == "__main__":
    main()
