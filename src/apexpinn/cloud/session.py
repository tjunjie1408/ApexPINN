"""Cooperative Kaggle segment deadlines, separate from the scientific trial budget."""

from __future__ import annotations

import os
import time


class SessionPaused(SystemExit):
    """Exit without marking the current fit or trial as failed."""

    def __init__(self) -> None:
        super().__init__(75)


def pause_due() -> bool:
    if os.environ.get("APEXPINN_REMOTE_RUNTIME") != "kaggle":
        return False
    deadline = os.environ.get("APEXPINN_SESSION_DEADLINE")
    return deadline is not None and time.monotonic() >= float(deadline)


def check_session() -> None:
    if pause_due():
        print("Planned segment pause. Preserve this run and resume its compatible checkpoints.", flush=True)
        raise SessionPaused()
