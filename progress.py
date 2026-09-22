"""
progress.py - dependency-free progress reporting for long runs.

Writes to stderr so piped stdout (result tables, CSV) stays clean and
parseable. On a terminal it rewrites a single line; when redirected to a log
file it emits a timestamped line at a fixed interval instead of thousands of
carriage returns.

    from progress import Progress, iterate

    p = Progress(90, "no mutation")
    for ...:
        p.update()
    p.close()

    for batch in iterate(loader, label="train"):
        ...

Set PROGRESS=0 in the environment to silence it everywhere.
"""
from __future__ import annotations

import os
import sys
import time

__all__ = ["Progress", "iterate", "enabled", "fmt_time"]

_BAR_WIDTH = 24


def enabled() -> bool:
    """Progress is on unless PROGRESS is set to 0/false/no/off."""
    return os.getenv("PROGRESS", "1").strip().lower() not in ("0", "false", "no", "off")


def fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Progress:
    """Count-based progress with elapsed time and a completion-rate ETA."""

    def __init__(self, total: int, label: str = "", stream=None,
                 tty_interval: float = 0.25, log_interval: float = 30.0):
        self.total = max(int(total), 0)
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.n = 0
        self.start = time.time()
        self._last_emit = 0.0
        self._dirty = False
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interval = tty_interval if self.is_tty else log_interval
        self.enabled = enabled()
        if self.enabled:
            self._emit(force=True)

    # -- public ------------------------------------------------------------
    def update(self, n: int = 1, note: str = "") -> None:
        """Advance by n completed items."""
        self.n += n
        self._emit(note=note)

    def tick(self, note: str = "") -> None:
        """Refresh the elapsed clock without advancing the count."""
        self._emit(note=note)

    def close(self, note: str = "") -> None:
        """Finish the line. Safe to call twice."""
        if not self.enabled or not self._dirty:
            return
        # The final update already printed a complete line in log mode; only
        # re-emit if there is something new to say or the count never landed.
        if self.is_tty or note or not (self.total and self.n >= self.total):
            self._emit(force=True, note=note)
        if self.is_tty:
            self.stream.write("\n")
        self.stream.flush()
        self._dirty = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- internal ----------------------------------------------------------
    def _emit(self, force: bool = False, note: str = "") -> None:
        if not self.enabled:
            return
        now = time.time()
        done = self.total and self.n >= self.total
        if not force and not done and (now - self._last_emit) < self.interval:
            return
        self._last_emit = now

        elapsed = now - self.start
        parts = []
        if self.label:
            parts.append(self.label)

        if self.total:
            frac = min(self.n / self.total, 1.0)
            if self.is_tty:
                filled = int(_BAR_WIDTH * frac)
                parts.append("[" + "#" * filled + "-" * (_BAR_WIDTH - filled) + "]")
            parts.append(f"{self.n}/{self.total}")
            parts.append(f"{frac * 100:3.0f}%")
        else:
            parts.append(str(self.n))

        parts.append(f"elapsed {fmt_time(elapsed)}")
        if self.total and self.n:
            remaining = (elapsed / self.n) * (self.total - self.n)
            parts.append(f"eta {fmt_time(remaining)}")
        if note:
            parts.append(note)

        line = "  ".join(parts)
        if self.is_tty:
            self.stream.write("\r\033[2K" + line)
        else:
            self.stream.write(time.strftime("[%H:%M:%S] ") + line + "\n")
        self.stream.flush()
        self._dirty = True


def iterate(iterable, total: int | None = None, label: str = "", **kw):
    """Wrap an iterable (e.g. a DataLoader) in a Progress bar."""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = 0
    p = Progress(total, label, **kw)
    try:
        for item in iterable:
            yield item
            p.update()
    finally:
        p.close()
