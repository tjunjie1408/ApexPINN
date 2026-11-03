"""Export existing G1 forecasts and build the current code/data handoff. Never fits a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import input_paths, pack
from apexpinn.eval.g1_report import read_study_for_review


def records_digest(records: list[dict]) -> str:
    """Compare nested records across the lossless Parquet transport, including component scales."""

    def native(value):
        if isinstance(value, np.ndarray):
            return native(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [native(item) for item in value]
        return value

    checksum = hashlib.sha256()
    for row in records:
        checksum.update(json.dumps(native(row), sort_keys=True, allow_nan=False).encode())
        checksum.update(b"\n")
    return checksum.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/colab_exports/apexpinn_t13_input_v1.zip"))
    args = parser.parse_args()
    review = json.loads(Path("experiments/g1_real_20261103/review.json").read_text())
    if review["G1"] != "PASS":
        raise ValueError("accepted G1 required")
    for label in ("low", "mid", "high"):
        summary = review["studies"][label]
        source = Path(summary["source"])
        target = Path("data/cloud_baselines") / f"phi_{label}.parquet"
        metadata = target.with_suffix(".json")
        if metadata.exists():
            previous = json.loads(metadata.read_text())
            if previous["source_sha256"] != summary["source_sha256"] or digest(target) != previous["sha256"]:
                raise ValueError("existing compact baseline export differs")
        else:
            if target.exists():
                raise ValueError("uncommitted baseline export exists; inspect before making a new version")
            if digest(source) != summary["source_sha256"]:
                raise ValueError("G1 source hash mismatch")
            receipt = read_study_for_review(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(receipt["forecasts"]).to_parquet(target, index=False)
            record_sha = records_digest(receipt["forecasts"])
            if records_digest(pd.read_parquet(target).to_dict("records")) != record_sha:
                raise ValueError("compact baseline transport changed forecast records; preserve and inspect")
            write_json(
                metadata,
                {
                    "schema": "apexpinn.baseline_export/v1",
                    "evidence": "Observed",
                    "source": str(source),
                    "source_sha256": summary["source_sha256"],
                    "sha256": digest(target),
                    "forecasts": len(receipt["forecasts"]),
                    "records_sha256": record_sha,
                    "round_trip": "PASS: all nested forecast fields preserved",
                    "operation": "lossless record export; no fitting or scoring",
                },
            )
            del receipt
        print(json.dumps({"phi": label, "export": str(target)}), flush=True)
    root = Path.cwd()
    manifest = pack(root, input_paths(root), args.out, kind="input")
    print(json.dumps({"archive": str(args.out), "sha256": digest(args.out), "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
