"""G1 acceptance must not convert software fixtures or an incomplete sweep into real evidence."""

import json

import pytest

from apexpinn.eval.g1_report import (
    build_g1_report,
    read_study_for_review,
    review_complete,
    summarize_study,
)


@pytest.mark.parametrize("chunk_size", [1, 7, 31, 4096])
def test_incremental_review_reader_preserves_values_except_audit(tmp_path, chunk_size):
    source = {
        "conditions": {"phi": 0.03, "unicode": "Caf\u00e9 \u4e2d\u6587", "brace": '\\"}['},
        "forecasts": [{"target": -123456789.987654, "empty": []}, {}, [True, None, 123456]],
        "origin_audit": [{"arbitrary": [1, 2, {"reason": "history"}]}] * 100,
        "tail": [12345678901234567890, 1.2e-30],
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    expected = {**source, "origin_audit": []}
    assert read_study_for_review(path, chunk_size=chunk_size) == expected


@pytest.mark.parametrize("text", ['{"a": [1, 2', '{"a": []} trailing', '{"a": [], "a": []}'])
def test_incremental_review_reader_rejects_partial_or_ambiguous_json(tmp_path, text):
    path = tmp_path / "study.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_study_for_review(path, chunk_size=2)


def test_fixture_study_never_passes_real_gate(tmp_path):
    from test_eval import DECISIONS

    from apexpinn.train.baseline_study import fixture_silver, run_study

    receipt = run_study(
        fixture_silver(DECISIONS),
        DECISIONS,
        out=tmp_path / "fixture",
        report=tmp_path / "fixture.md",
        models=("B0",),
        folds=("F1",),
        validation_only=True,
        source_kind="crafted_fixture",
    )
    summary = summarize_study(receipt, source="explicit crafted fixture", source_hash="fixture")
    assert not summary["complete"] and "not real development evaluation" in summary["errors"]
    report = build_g1_report({"studies": {"mid": summary}}, source="test fixture review")
    assert "G1: **NOT ESTABLISHED**" in report
    assert "NOT ESTABLISHED" in report
    assert len(summary["origin_counts_by_fold"]) == 4
    assert "History rejection share" in report
    for row in receipt["origin_metrics"]:
        if row["quantity"]["metric"] == "history_rejection_share":
            row["value"] = 0.99
    receipt["selections"]["F1/B0"] = {"unselected": True}
    changed = summarize_study(receipt, source="tampered fixture", source_hash="fixture")
    assert "F1/h5: history rejection share differs" in changed["errors"]
    assert "F1/B0: selection differs from inner-only minimum" in changed["errors"]


def test_sweep_requires_all_declared_phi_and_identical_population_keys():
    studies = {
        label: {"phi": phi, "complete": True, "population_keys_sha256": "same"}
        for label, phi in [("low", 0.03), ("mid", 0.06), ("high", 0.09)]
    }
    assert review_complete(studies)
    studies["low"]["phi"], studies["high"]["phi"] = 0.09, 0.03
    assert not review_complete(studies)
    studies["low"]["phi"], studies["high"]["phi"] = 0.03, 0.09
    studies["high"]["phi"] = 0.06
    assert not review_complete(studies)
    studies["high"]["phi"] = 0.09
    studies["high"]["population_keys_sha256"] = "different"
    assert not review_complete(studies)
    del studies["high"]
    assert not review_complete(studies)
