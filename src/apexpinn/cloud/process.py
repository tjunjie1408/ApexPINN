"""Persist each user-launched remote command's output and process outcome."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from apexpinn.cloud.artifacts import write_json
from apexpinn.cloud.runtime import require_remote_environment


def run_logged(command: list[str], *, cwd: Path, logs: Path) -> None:
    """Stream merged output to stdout and an immutable attempt log in persistent storage."""
    require_remote_environment()
    logs.mkdir(parents=True, exist_ok=True)
    attempt = logs / f"attempt_{uuid.uuid4().hex}"
    log_path = attempt.with_suffix(".log")
    status_path = attempt.with_suffix(".json")
    record = {
        "command": command,
        "cwd": str(cwd),
        "log": str(log_path),
        "status": "starting",
        "started_at": datetime.now(UTC).isoformat(),
    }
    write_json(status_path, record)
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        try:
            with subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            ) as process:
                record.update(status="running", pid=process.pid)
                write_json(status_path, record)
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
        except BaseException as exc:
            record.update(
                status="interrupted",
                error_type=type(exc).__name__,
                elapsed_s=time.perf_counter() - started,
                ended_at=datetime.now(UTC).isoformat(),
            )
            write_json(status_path, record)
            raise
    paused = returncode == 75 and os.environ.get("APEXPINN_REMOTE_RUNTIME") == "kaggle"
    record.update(
        status="finished" if returncode == 0 else "paused" if paused else "failed",
        returncode=returncode,
        elapsed_s=time.perf_counter() - started,
        ended_at=datetime.now(UTC).isoformat(),
    )
    write_json(status_path, record)
    print(f"Saved process record: {status_path}", flush=True)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    run_logged(command, cwd=args.cwd, logs=args.logs)


if __name__ == "__main__":
    main()
