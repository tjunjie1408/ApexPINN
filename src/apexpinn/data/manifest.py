"""Bronze Parquet writing and SHA-256 manifests (PLAN T8, PROJECT section 3.1).

Each Bronze event directory holds Parquet files plus ``manifest.json``. The manifest records the library
versions, the download timestamp, and per file the SHA-256 of the file bytes, the row count, and the
columns. The timestamp lives only in the manifest, never in the hashed Parquet files.

Parquet bytes are deterministic for a given frame, pandas version, and pyarrow version because the files
are written with fixed settings (``PARQUET_SETTINGS``, no index) and pyarrow writes no timestamps. A
different pyarrow version may produce different bytes for the same data, so the manifest records it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = "apexpinn.bronze_manifest/v1"
PARQUET_SETTINGS: dict[str, Any] = {
    "compression": "zstd",
    "compression_level": 3,
    "version": "2.6",
    "use_dictionary": True,
    "write_statistics": True,
    "data_page_version": "1.0",
}
_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of the file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_atomically(tmp: Path, path: Path) -> None:
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_parquet(frame: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Write ``frame`` (without its index) with fixed settings and return its manifest entry.

    The file is written to a temporary name and then renamed, so a crash never leaves a truncated file
    under the final name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, str(tmp), **PARQUET_SETTINGS)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _replace_atomically(tmp, path)
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": [str(c) for c in frame.columns],
    }


def write_manifest(event_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write ``manifest.json`` (sorted keys, so equal content gives equal bytes) atomically."""
    path = Path(event_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _replace_atomically(tmp, path)
    return path


def read_manifest(event_dir: str | Path) -> dict[str, Any] | None:
    """The parsed manifest, or None if it is missing or not valid JSON."""
    path = Path(event_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify_event(event_dir: str | Path, required_files: tuple[str, ...] = ()) -> list[str]:
    """Problems with a Bronze event directory; an empty list means complete and unmodified.

    Checks that the manifest exists with the expected schema, lists every ``required_files`` name, and
    that every listed file exists with the recorded SHA-256.
    """
    event_dir = Path(event_dir)
    manifest = read_manifest(event_dir)
    if manifest is None:
        return [f"{MANIFEST_NAME} missing or unreadable"]
    if manifest.get("schema") != MANIFEST_SCHEMA:
        return [f"manifest schema {manifest.get('schema')!r} is not {MANIFEST_SCHEMA!r}"]
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return ["manifest has no 'files' mapping"]
    problems = [f"{name} not listed in manifest" for name in required_files if name not in files]
    for name, entry in sorted(files.items()):
        path = event_dir / name
        if not path.is_file():
            problems.append(f"{name} missing")
        elif not isinstance(entry, Mapping) or sha256_file(path) != entry.get("sha256"):
            problems.append(f"{name} SHA-256 does not match the manifest")
    return problems
