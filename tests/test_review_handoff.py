"""Budget arithmetic and mocked notebook launching. No model execution or optimizer updates."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from apexpinn.eval.review_figures import budget_plan


def builder():
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("prepare_remaining", script_dir / "prepare_remaining_m2.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_budget_scenarios_are_assumptions_with_explicit_reserve():
    plan = budget_plan()
    assert plan["total_hours"] == pytest.approx(100 / 1.94)
    assert plan["hours_above_reserve"] == pytest.approx(80 / 1.94)
    assert [s["units"] for s in plan["scenarios"]] == pytest.approx([46.56, 69.84, 93.12])
    assert [s["preserves_reserve"] for s in plan["scenarios"]] == [True, True, False]
    with pytest.raises(ValueError):
        budget_plan(balance=10, reserve=20)


def test_one_fold_settings_and_mocked_launch_budget_gate(tmp_path):
    prepare = builder().prepare
    for fold in ("F4", "F5", "F6"):
        out = tmp_path / fold
        receipt = prepare(out, fold=fold)
        notebook = json.loads((out / "train_m2_remaining_mid.ipynb").read_text())
        env = {"Path": Path}
        exec("".join(notebook["cells"][3]["source"]), env)
        assert env["FOLDS"] == [fold] and env["RUN_ID"] == f"t13_m2_{fold.lower()}_mid_run01"
        assert receipt["jobs_per_launch"] == 1 and not receipt["training_launched"]
        assert all(not c.get("outputs") for c in notebook["cells"])
        calls = []
        env.update(
            RUN_ROOT=tmp_path / "run" / fold,
            WORKSPACE=tmp_path,
            PYTHON="python",
            shutil=SimpleNamespace(which=lambda _: "/usr/bin/flock"),
            subprocess=SimpleNamespace(run=lambda *args, captured=calls, **kwargs: captured.append(args)),
            command=["mock"],
        )
        launch = "".join(notebook["cells"][15]["source"])
        env["AVAILABLE_UNITS"] = 25
        with pytest.raises(RuntimeError, match="reserve"):
            exec(launch, env)
        assert not calls
        env["AVAILABLE_UNITS"] = 100
        exec(launch, env)
        assert len(calls) == 1 and calls[0][0][:2] == ["/usr/bin/flock", "--nonblock"]
        calls.clear()
        job = env["JOB"]
        job.parent.mkdir(parents=True)
        job.write_text("{}")
        exec(launch, env)
        assert not calls
    with pytest.raises(ValueError):
        prepare(tmp_path / "bad", fold="F3")
