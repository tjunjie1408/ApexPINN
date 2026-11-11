"""Hash-verified ZIP handoff. No model, Google credentials or remote execution dependency."""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from apexpinn.cloud.artifacts import digest, write_json

# Preserve legacy input reads; apply the stricter local-document boundary to new exports.
PRIVATE_DOCS = {"PROJECT.md", "PRD.md", "PLAN.md", "GLOSSARY.md"}
PRIVATE_EXPORT_DOCS = {
    "agents.md",
    "record.md",
    "project.md",
    "prd.md",
    "plan.md",
    "glossary.md",
    "experimental_blueprint.md",
    "experiment_blueprint.md",
    "decisions_review.md",
    "decision_review.md",
}


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(p in ("", ".", "..") for p in name.split("/"))
        or any(p.endswith((".", " ")) for p in path.parts)
    ):
        raise ValueError(f"unsafe archive path: {name!r}")
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if any(p.split(".")[0].lower() in reserved for p in path.parts):
        raise ValueError("reserved Windows archive path")
    if path.name in PRIVATE_DOCS or any(p.startswith(".") for p in path.parts):
        raise ValueError("private/hidden files are not cloud artifacts")
    return name


def pack(root: Path, paths: list[Path], output: Path, *, kind: str) -> dict:
    """Package an explicit allowlist, never an entire workstation/repository tree."""
    if output.exists():
        raise FileExistsError(output)
    root = root.resolve()
    members = {}
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"not an ordinary artifact file: {path}")
        name = safe_name(path.resolve().relative_to(root).as_posix())
        if Path(name).name.casefold() in PRIVATE_EXPORT_DOCS:
            raise ValueError(f"private local document cannot be exported: {name}")
        if name.casefold() in {p.casefold() for p in members}:
            raise ValueError("duplicate archive member")
        members[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    if not members:
        raise ValueError("empty handoff")
    manifest = {"schema": "apexpinn.cloud_bundle/v1", "kind": kind, "files": members}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(root / name, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    write_json(output.with_suffix(output.suffix + ".receipt.json"), {"sha256": digest(output), **manifest})
    return manifest


def unpack(archive_path: Path, destination: Path, *, expected_sha256: str, kind: str) -> dict:
    """Verify every byte/path before publishing a new destination; never overwrite existing results."""
    if destination.exists():
        raise FileExistsError(destination)
    if digest(archive_path) != expected_sha256:
        raise ValueError("archive SHA-256 mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [safe_name(info.filename) for info in infos]
        if len({p.casefold() for p in names}) != len(names):
            raise ValueError("duplicate/case-colliding archive members")
        if sum(i.file_size for i in infos) > 64 * (1 << 30):
            raise ValueError("archive exceeds 64 GiB expansion limit")
        if any(stat.S_ISLNK(i.external_attr >> 16) or i.is_dir() for i in infos):
            raise ValueError("links/directory entries are not supported")
        if archive.getinfo("bundle_manifest.json").file_size > 64 * (1 << 20):
            raise ValueError("manifest exceeds 64 MiB limit")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if manifest.get("schema") != "apexpinn.cloud_bundle/v1" or manifest.get("kind") != kind:
            raise ValueError("bundle schema/kind mismatch")
        if set(names) != {*manifest["files"], "bundle_manifest.json"}:
            raise ValueError("unmanifested/missing archive files")
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "verified"
            staging.mkdir()
            for name, expected in manifest["files"].items():
                target = staging / safe_name(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                if target.stat().st_size != expected["bytes"] or digest(target) != expected["sha256"]:
                    raise ValueError(f"payload hash/size mismatch: {name}")
            write_json(staging / "bundle_manifest.json", manifest)
            staging.rename(destination)
    return manifest


def input_paths(root: Path) -> list[Path]:
    paths = [root / n for n in ("pyproject.toml", "uv.lock", "README.md")]
    paths += [root / "notebooks/colab/README.md"]
    for directory, pattern in (
        ("src", "*.py"),
        ("scripts", "*.py"),
        ("configs", "*.yaml"),
        ("notebooks", "*.ipynb"),
    ):
        paths += list((root / directory).rglob(pattern))
    paths += list((root / "data/gold_development_frozen_v1").glob("phi_*/manifest.json"))
    paths += list((root / "data/gold_development_frozen_v1").glob("phi_*/F*/laps.parquet"))
    paths += list((root / "data/cloud_baselines").glob("*.parquet"))
    paths += list((root / "data/cloud_baselines").glob("*.json"))
    paths += [
        root / n
        for n in (
            "experiments/g0_development_20261103/acceptance.json",
            "experiments/g1_real_20261103/acceptance.json",
            "experiments/g1_real_20261103/review.json",
            "experiments/pivots.md",
            "reports/s1_identifiability.md",
        )
    ]
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("pack-input")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--out", type=Path, required=True)
    results = sub.add_parser("pack-results")
    results.add_argument("--root", type=Path, required=True)
    results.add_argument("--out", type=Path, required=True)
    receive = sub.add_parser("unpack")
    receive.add_argument("archive", type=Path)
    receive.add_argument("--dest", type=Path, required=True)
    receive.add_argument("--sha256", required=True)
    receive.add_argument("--kind", choices=("input", "results"), required=True)
    args = parser.parse_args()
    if args.action == "unpack":
        result = unpack(args.archive, args.dest, expected_sha256=args.sha256, kind=args.kind)
    elif args.action == "pack-input":
        result = pack(args.root, input_paths(args.root), args.out, kind="input")
    else:
        if args.out.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("write result archive outside its source tree")
        paths = [
            p
            for p in args.root.rglob("*")
            if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(args.root).parts)
            and not p.name.endswith((".tmp", ".pending"))
        ]
        result = pack(args.root, paths, args.out, kind="results")
    print(json.dumps({"kind": result["kind"], "files": len(result["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
