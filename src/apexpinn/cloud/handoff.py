"""Restore an immutable input ZIP with a separately verified code-only update."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from apexpinn.cloud.artifacts import digest, write_json
from apexpinn.cloud.transfer import safe_name, unpack


def restore_overlay(base: Path, overlay: Path, destination: Path, *, base_sha256: str) -> None:
    """Publish a new workspace; reuse it only while every declared payload still matches."""
    patch = json.loads((overlay / "bundle_manifest.json").read_text())
    if patch.get("schema") != "apexpinn.cloud_bundle/v1" or patch.get("kind") != "input":
        raise ValueError("invalid update manifest")
    for name, item in patch["files"].items():
        safe_name(name)
        if name != "handoff.json" and name.split("/")[0] not in (
            "src",
            "scripts",
            "configs",
            "recovered_evaluations",
        ):
            raise ValueError("update contains an unsupported payload")
        if digest(overlay / name) != item["sha256"]:
            raise ValueError("update payload differs from manifest")
    handoff = json.loads((overlay / "handoff.json").read_text())
    if handoff["base_sha256"] != base_sha256:
        raise ValueError("update requires a different base archive")
    patch_sha = digest(overlay / "bundle_manifest.json")
    if destination.exists():
        state = json.loads((destination / "workspace_manifest.json").read_text())
        if state["base_sha256"] != base_sha256 or state["overlay_manifest_sha256"] != patch_sha:
            raise ValueError("workspace belongs to a different handoff")
        with zipfile.ZipFile(base) as archive:
            original = json.loads(archive.read("bundle_manifest.json"))
        if state["files"] != {**original["files"], **patch["files"]}:
            raise ValueError("workspace manifest changed")
        for name, item in state["files"].items():
            if digest(destination / safe_name(name)) != item["sha256"]:
                raise ValueError("workspace payload changed; restore into a new destination")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "workspace"
        manifest = unpack(base, stage, expected_sha256=base_sha256, kind="input")
        for name in patch["files"]:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(overlay / name, target)
        write_json(
            stage / "workspace_manifest.json",
            {
                "base_sha256": base_sha256,
                "overlay_manifest_sha256": patch_sha,
                "files": {**manifest["files"], **patch["files"]},
            },
        )
        stage.rename(destination)
