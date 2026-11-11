"""Small mocked execution and metadata tests. No model fits or diagnostic solves."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.handoff import restore_overlay
from apexpinn.cloud.process import run_logged
from apexpinn.cloud.transfer import pack, unpack


def test_logged_process_keeps_stderr_and_failed_exit(tmp_path, monkeypatch):
    from apexpinn.cloud import process as module

    monkeypatch.setenv("APEXPINN_REMOTE_RUNTIME", "colab")
    original = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: p == Path("/content") or original(p))

    class FakeProcess:
        pid = 123
        stdout = iter(["progress 100\n", "Traceback: original error\n"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 1

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: FakeProcess())
    with pytest.raises(module.subprocess.CalledProcessError):
        run_logged(["fake-training"], cwd=tmp_path, logs=tmp_path / "logs")
    record = json.loads(next((tmp_path / "logs").glob("*.json")).read_text())
    assert record["status"] == "failed" and record["returncode"] == 1
    assert "original error" in Path(record["log"]).read_text()


def test_logged_process_rejects_local_before_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("APEXPINN_REMOTE_RUNTIME", raising=False)
    with pytest.raises(RuntimeError, match="hosted Colab"):
        run_logged(["never-launched"], cwd=tmp_path, logs=tmp_path / "logs")
    assert not (tmp_path / "logs").exists()


def test_overlay_restores_and_rejects_modified_workspace(tmp_path):
    source = tmp_path / "base"
    source.mkdir()
    (source / "data.txt").write_text("retained data")
    archive = tmp_path / "base.zip"
    pack(source, [source / "data.txt"], archive, kind="input")
    patch = tmp_path / "patch"
    (patch / "src").mkdir(parents=True)
    (patch / "src/code.py").write_text("updated = True\n")
    write_json(patch / "handoff.json", {"base_sha256": digest(archive)})
    update = tmp_path / "update.zip"
    pack(patch, [patch / "src/code.py", patch / "handoff.json"], update, kind="input")
    restored_patch = tmp_path / "restored_patch"
    unpack(update, restored_patch, expected_sha256=digest(update), kind="input")
    destination = tmp_path / "workspace"
    restore_overlay(archive, restored_patch, destination, base_sha256=digest(archive))
    restore_overlay(archive, restored_patch, destination, base_sha256=digest(archive))
    assert (destination / "data.txt").read_text() == "retained data"
    (destination / "src/code.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="payload changed"):
        restore_overlay(archive, restored_patch, destination, base_sha256=digest(archive))


def test_s4_passes_recorded_context_and_history_requirement(monkeypatch):
    from apexpinn.eval import physics_report

    gold = pd.DataFrame(
        {
            "stint_id": ["s"] * 8,
            "stint_lap": range(1, 9),
            "circuit_id": ["c"] * 8,
            "compound_c": ["C3"] * 8,
            "obs_track_temp_c": [30.0] * 8,
            "is_scored_lap": [True] * 8,
            "proxy_y": list(range(8)),
            "ctx_circuit_q_mean": [0.4] * 8,
        }
    )
    calls = []

    def slice_stub(group, t0, horizon, *, circuit_q_mean, k_min):
        calls.append((circuit_q_mean, k_min))
        return object(), None, None

    monkeypatch.setattr(physics_report, "slice_at_origin", slice_stub)
    method = {
        "diagnostics": {
            "temperature_bands_c": [25, 35],
            "grid_points": 64,
            "dip_max_lap": 8,
            "dip_margin_s": 0.1,
        }
    }
    row = SimpleNamespace(
        stint_id="s", horizon=5, event_id="e", y_pred=1.0, y_true=1.0, t0=3, conditions={"k_min": 3}
    )
    physics_report.job_diagnostics(SimpleNamespace(members=()), gold, [row], method)
    assert calls == [(0.4, 3)]


def test_collector_counts_recovered_job_and_rejects_duplicate(tmp_path):
    from apexpinn.eval import physics_report

    root, packet = tmp_path / "run", tmp_path / "recovery"
    root.mkdir()
    packet.mkdir()
    identity = {
        "method_sha256": "method",
        "config_sha256": "config",
        "gold_sha256": {"mid/F1": "gold"},
        "source_sha256": {"src/apexpinn/physics/residuals.py": "frozen"},
    }
    write_json(root / "declaration.json", {"identity": identity})
    (packet / "forecasts.jsonl").write_text("saved forecasts")
    review = {
        "schema": "apexpinn.t13_recovery_report/v1",
        "status": "RECOVERED_EVALUATION_COMPLETE",
        "job": "M2_F1_mid_main",
        "scope": {"model": "M2", "fold": "F1", "phi": 0.06, "variant": "main"},
        "provenance": {
            "method_sha256": "method",
            "config_sha256": "config",
            "gold_sha256": "gold",
            "forecast_sha256": digest(packet / "forecasts.jsonl"),
            "numerical_source_sha256": identity["source_sha256"],
        },
        "execution": {
            "recorded_stage_timings": {
                stage: {"status_counts": {"finished": count}}
                for stage, count in (("tuning", 20), ("calibration", 5), ("final", 5))
            }
        },
        "s4_recovery": {"diagnostics": {"representative_stints": [{"stint": "saved"}]}},
        "metrics": [
            {
                "conditions": {"model": "M2", "phi": 0.06, "horizon_actual_laps": 5},
                "quantity": {"metric": "rmse"},
                "value": 1.0,
            }
        ],
    }
    path = packet / "review.json"
    write_json(path, review)
    assert physics_report.collect_results(root, recovered_reviews=(path,)) == 1
    result = json.loads((root / "review.json").read_text())
    assert result["population"]["completed_jobs"] == 1
    assert result["population"]["recovered_jobs"] == 1
    assert result["conditions"]["forecast_sweep"] == "INCOMPLETE"
    assert result["metrics"][0]["value"] == 1.0
    with pytest.raises(ValueError, match="duplicates"):
        physics_report.collect_results(root, recovered_reviews=(path, path))
    review["execution"]["recorded_stage_timings"]["tuning"]["status_counts"] = {"finished": 19, "failed": 1}
    write_json(path, review)
    assert physics_report.collect_results(root, recovered_reviews=(path,)) == 1
    review["provenance"]["config_sha256"] = "different"
    write_json(path, review)
    with pytest.raises(ValueError, match="method/config"):
        physics_report.collect_results(root, recovered_reviews=(path,))
