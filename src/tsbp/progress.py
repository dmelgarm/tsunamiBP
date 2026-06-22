"""A tiny dependency-free progress bar for the ray-tracing loops.

The expensive loops back-project one wavefront point at a time (each does a ray
fan + gridding); this prints a single self-updating line to stderr so a long run
shows how far along it is.  Disabled by default (pass a label to enable), so
library and test use stay silent.
"""
from __future__ import annotations

import sys
import time


def _fmt(secs):
    secs = int(secs)
    if secs >= 60:
        return f"{secs // 60:d}m{secs % 60:02d}s"
    return f"{secs:d}s"


def track(iterable, total, label="", stream=None, width=24):
    """Yield from ``iterable`` while printing a ``label |####----| 45% 9/20 eta``
    line that updates after each item.  ``total`` is the item count."""
    stream = stream or sys.stderr
    start = time.time()
    last_pct = -1
    for i, item in enumerate(iterable):
        yield item
        done = i + 1
        pct = int(100 * done / total) if total else 100
        if pct != last_pct or done == total:
            last_pct = pct
            filled = int(width * done / total) if total else width
            bar = "#" * filled + "-" * (width - filled)
            elapsed = time.time() - start
            eta = elapsed / done * (total - done) if done else 0.0
            stream.write(f"\r  {label} |{bar}| {pct:3d}%  {done}/{total}  "
                         f"eta {_fmt(eta):>6}")
            stream.flush()
    stream.write("\n")
    stream.flush()


def maybe_track(iterable, total, label):
    """``track`` if ``label`` is truthy, else the plain iterable (silent)."""
    return track(iterable, total, label) if label else iterable
