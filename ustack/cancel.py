"""A stop request for long computations run in a worker thread (Flake layout › Auto design: Abort, user 2026-11-21).

The thread sets a check (:func:`set_check`) that raises when the user asked to stop; the generators call :func:`poll`
in their inner loops, so the run ends within a fraction of a second. The check is per thread: the same generators
called from the window's own thread are never stopped."""

from __future__ import annotations

import threading

_local = threading.local()


def set_check(fn) -> None:
    """``fn()`` raises to stop the calling thread's computation (None clears it)."""
    _local.fn = fn


def poll() -> None:
    fn = getattr(_local, "fn", None)
    if fn is not None:
        fn()
