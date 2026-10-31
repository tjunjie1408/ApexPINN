"""Run metadata recorded with every training run: versions, devices, dtype, seeds, git commit, data hash.

See PROJECT.md section 5.6 (reproducibility) and PLAN.md T4. The git commit is read from the .git
directory directly, without starting a subprocess.
"""

from __future__ import annotations

import importlib.metadata
import platform
import string
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import jax

PACKAGES = ("jax", "jaxlib", "equinox", "diffrax", "optax")
DTYPES = ("float32", "float64")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_sha(text: str) -> bool:
    return len(text) in (40, 64) and all(ch in string.hexdigits for ch in text)


def _git_dir(start: Path) -> Path | None:
    """The .git directory of the nearest parent of start (inclusive) that has one."""
    for folder in (start, *start.parents):
        dot_git = folder / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():  # worktree or submodule: the file holds "gitdir: <path>"
            content = dot_git.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                return (folder / content.removeprefix("gitdir:").strip()).resolve()
            return None
    return None


def _common_dir(git_dir: Path) -> Path:
    """The directory with shared refs: git_dir itself, or the target of a worktree's commondir file."""
    commondir = git_dir / "commondir"
    if commondir.is_file():
        return (git_dir / commondir.read_text(encoding="utf-8").strip()).resolve()
    return git_dir


def _git_commit(start: Path) -> str | None:
    """HEAD commit of the repository containing start, or None on any failure.

    HEAD is read from the (worktree) git directory; branch refs and packed-refs are looked up there first and
    then in the common directory named by commondir, where a worktree's branches live.
    """
    try:
        git_dir = _git_dir(start)
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head if _is_sha(head) else None  # detached HEAD
        ref = head.removeprefix("ref:").strip()
        common = _common_dir(git_dir)
        search = (git_dir,) if common == git_dir else (git_dir, common)
        for folder in search:
            ref_file = folder / ref
            if ref_file.is_file():
                sha = ref_file.read_text(encoding="utf-8").strip()
                return sha if _is_sha(sha) else None
        for folder in search:
            packed = folder / "packed-refs"
            if not packed.is_file():
                continue
            for line in packed.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                sha, _, name = line.partition(" ")
                if name.strip() == ref and _is_sha(sha):
                    return sha
        return None
    except (OSError, UnicodeDecodeError):
        return None


def run_metadata(
    seeds: Sequence[int] | None = None, data_hash: str | None = None, dtype: str | None = None
) -> dict:
    """JSON-serializable run metadata (PROJECT section 5.6).

    dtype defaults to 'float64' if jax_enable_x64 is on, else 'float32'. git_commit is the HEAD of the
    repository that contains this package (found from the module's location, not the working directory, so
    a notebook started elsewhere still records it), or None if the package is not inside a git checkout.
    """
    if dtype is None:
        dtype = "float64" if jax.config.x64_enabled else "float32"
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    seed_list = []
    for seed in seeds or ():
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seeds must be integers, got {seed!r}")
        seed_list.append(int(seed))
    if data_hash is not None and not isinstance(data_hash, str):
        raise ValueError(f"data_hash must be a string or None, got {type(data_hash).__name__}")
    return {
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "devices": [str(device) for device in jax.devices()],
        "dtype": dtype,
        "seeds": seed_list,
        "git_commit": _git_commit(Path(__file__).resolve().parent),
        "data_hash": data_hash,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
