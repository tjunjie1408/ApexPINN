"""Pure array arithmetic and mocked guard checks. No models, fitting or optimizer updates."""

import numpy as np
import pytest

from apexpinn.eval import offset_diagnostic as diagnostic


def test_decomposition_separates_offset_and_shape_with_masked_padding():
    y = np.array([[10.0, 10.0, np.nan], [20.0, 22.0, 24.0]])
    shape = np.zeros_like(y)
    mask = np.array([[True, True, False], [True, True, True]])
    offsets = np.array([7.0, 21.0])
    before = [a.copy() for a in (y, shape, mask, offsets)]
    result = diagnostic.decompose_error(y, shape, mask, offsets)
    # First stint has only a constant 3 s offset error. Second has a 1 s offset error and shape error.
    assert result["trained_offset_rmse_s"] ** 2 == pytest.approx(29 / 5)
    assert result["profiled_offset_rmse_s"] ** 2 == pytest.approx(8 / 5)
    assert result["offset_component_mse_s2"] == pytest.approx(21 / 5)
    assert result["offset_share_of_training_sse"] == pytest.approx(21 / 29)
    assert [s["profiled_offset_s"] for s in result["stints"]] == [10, 22]
    for original, current in zip(before, (y, shape, mask, offsets), strict=True):
        np.testing.assert_array_equal(original, current)


def test_zero_error_is_finite_and_invalid_observations_stop():
    result = diagnostic.decompose_error([[2, 2]], [[0, 0]], [[True, True]], [2])
    assert result["offset_share_of_training_sse"] == 0
    with pytest.raises(ValueError, match="finite"):
        diagnostic.decompose_error([[np.nan]], [[0]], [[True]], [0])
    with pytest.raises(ValueError, match="observed"):
        diagnostic.decompose_error([[1]], [[0]], [[False]], [0])


def test_remote_guard_runs_before_reading_inputs_or_creating_output(tmp_path, monkeypatch):
    def blocked():
        raise RuntimeError("hosted Colab GPU required")

    monkeypatch.setattr(diagnostic, "_remote_guard", blocked)
    monkeypatch.setattr(diagnostic, "_restore_model", lambda *args: pytest.fail("model loader reached"))
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="hosted Colab"):
        diagnostic.run_diagnostic(tmp_path / "missing", tmp_path / "gold", output)
    assert not output.exists()


def test_plotting_overrides_colab_backend_before_matplotlib_import(tmp_path, monkeypatch):
    import builtins
    import os

    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "matplotlib":
            assert os.environ["MPLBACKEND"] == "Agg", "headless backend must be set before import"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    rows = [
        {
            "stage": "final",
            "member": i,
            "trained_offset_rmse_s": 2.0,
            "profiled_offset_rmse_s": 1.0,
            "curves": [{"batch_row": 0, "tau": [1, 2], "y_s": [90, 91], "shape_s": [0, 1]}],
            "stints": [{"trained_offset_s": 89, "profiled_offset_s": 90}],
        }
        for i in range(5)
    ]
    diagnostic._plot_results(rows, tmp_path)
    for name in ("offset_errors.png", "example_curve.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _saved_diagnostic_fixture(tmp_path):
    import json

    from apexpinn.cloud.artifacts import digest

    source, run = tmp_path / "saved", tmp_path / "run"
    source.mkdir()
    job = run / "jobs/M1_F1_mid_main"
    files = {}
    for stage in ("calibration", "final"):
        for member in range(5):
            rel = f"{stage}/member_{member}/fit.json"
            path = job / rel
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"identity": {"batch_sha256": stage}}))
            files[rel] = digest(path)
            row = {
                "stage": stage,
                "member": member,
                "fit_sha256": files[rel],
                "population": {"batch_sha256": stage},
                "n_stints": 1,
                "n_observed_laps": 2,
                "trained_offset_rmse_s": np.sqrt(2),
                "profiled_offset_rmse_s": 1.0,
                "offset_component_mse_s2": 1.0,
                "offset_share_of_training_sse": 0.5,
                "stints": [
                    {
                        "batch_row": 0,
                        "n": 2,
                        "shape_rmse_s": 1.0,
                        "trained_offset_s": 0.0,
                        "profiled_offset_s": 1.0,
                    }
                ],
            }
            (source / f"{stage}_member_{member}.json").write_text(json.dumps(row))
    (job / "job.json").write_text(json.dumps({"files": files}))
    status = tmp_path / "failed.json"
    status.write_text(json.dumps({"status": "failed", "returncode": 1}))
    return source, run, status


def test_recovery_uses_saved_rows_and_preserves_failed_exit_without_model_replay(tmp_path, monkeypatch):
    source, run, status = _saved_diagnostic_fixture(tmp_path)

    def fake_plot(rows, output):
        assert len(rows) == 10
        for name in ("offset_errors.png", "example_curve.png"):
            (output / name).write_bytes(b"PNG fixture")

    monkeypatch.setattr(diagnostic, "_plot_results", fake_plot)
    monkeypatch.setattr(diagnostic, "_remote_guard", lambda: pytest.fail("remote model guard reached"))
    monkeypatch.setattr(diagnostic, "_restore_model", lambda *args: pytest.fail("model replay reached"))
    result = diagnostic.recover_diagnostic(source, run, tmp_path / "recovered", process_status=status)
    assert result["status"] == "RECOVERED_DIAGNOSTIC_COMPLETE"
    assert result["conditions"]["model_replay"] is False
    assert result["original_process"]["returncode"] == 1
    assert result["population"]["fits"] == 10
    with pytest.raises(ValueError, match="new recovery"):
        diagnostic.recover_diagnostic(source, run, tmp_path / "recovered")


@pytest.mark.parametrize("corruption", ["fit_sha256", "n_observed_laps"])
def test_recovery_rejects_incompatible_identity_or_population_before_output(tmp_path, corruption):
    import json

    source, run, _ = _saved_diagnostic_fixture(tmp_path)
    path = source / "calibration_member_0.json"
    row = json.loads(path.read_text())
    row[corruption] = "changed" if corruption == "fit_sha256" else 99
    path.write_text(json.dumps(row))
    output = tmp_path / "recovered"
    with pytest.raises(ValueError, match="identity|population"):
        diagnostic.recover_diagnostic(source, run, output)
    assert not output.exists()
