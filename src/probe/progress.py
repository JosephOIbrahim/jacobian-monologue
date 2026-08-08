"""Zero-dependency progress bar. stdlib only.

tqdm is present transitively (huggingface-hub pulls it) but is NOT a declared
dependency of this project. Use this instead so the probe does not silently
break if that transitive drops.

Two entry points:

    from probe.progress import bar, Bar

    for pair in bar(fact_pairs, label="axis-2 sweep"):
        ...

    b = Bar(total=n_pairs * 5, label="axis-2 sweep")
    for pair in pairs:
        for pos in range(5):
            run(pair, pos)
            b.step()
    b.close()

Falls back to periodic plain lines when stdout is not a TTY, so Claude Code
logs and redirected output stay readable instead of filling with \\r noise.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Optional, TextIO, TypeVar

T = TypeVar("T")

_BLOCKS = " ▏▎▍▌▋▊▉█"


def _fmt(seconds: float) -> str:
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Bar:
    """Manual-advance progress bar."""

    def __init__(
        self,
        total: int,
        label: str = "",
        width: int = 32,
        stream: Optional[TextIO] = None,
        min_interval: float = 0.1,
    ) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.width = width
        self.stream = stream or sys.stderr
        self.min_interval = min_interval
        self.n = 0
        self._start = time.monotonic()
        self._last_draw = 0.0
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._closed = False
        self._drew_final = False
        self._draw(force=True)

    def step(self, k: int = 1) -> None:
        self.n = min(self.n + k, self.total)
        self._draw()

    def _draw(self, force: bool = False) -> None:
        if self._closed:
            return
        now = time.monotonic()
        interval = self.min_interval if self._tty else 5.0
        if not force and (now - self._last_draw) < interval and self.n < self.total:
            return
        if self.n >= self.total:
            if self._drew_final:
                return
            self._drew_final = True
        self._last_draw = now

        frac = self.n / self.total
        elapsed = now - self._start
        rate = self.n / elapsed if elapsed > 0 and self.n else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else float("inf")

        if self._tty:
            filled = frac * self.width
            whole = int(filled)
            part = int((filled - whole) * (len(_BLOCKS) - 1))
            glyphs = "█" * whole
            if whole < self.width:
                glyphs += _BLOCKS[part] + " " * (self.width - whole - 1)
            line = (
                f"\r{self.label:<20.20} |{glyphs}| "
                f"{self.n:>5}/{self.total} {frac * 100:5.1f}% "
                f"[{_fmt(elapsed)}<{_fmt(eta)}, {rate:.2f}/s]"
            )
            self.stream.write(line)
        else:
            self.stream.write(
                f"[{self.label}] {self.n}/{self.total} ({frac * 100:.1f}%) "
                f"elapsed={_fmt(elapsed)} eta={_fmt(eta)} rate={rate:.2f}/s\n"
            )
        self.stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._draw(force=True)
        if self._tty:
            self.stream.write("\n")
        self.stream.flush()
        self._closed = True

    def __enter__(self) -> "Bar":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def bar(
    iterable: Iterable[T],
    total: Optional[int] = None,
    label: str = "",
    width: int = 32,
    stream: Optional[TextIO] = None,
) -> Iterator[T]:
    """Wrap an iterable. Falls back gracefully if it has no __len__."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = 0
    if not total:
        yield from iterable
        return
    b = Bar(total=total, label=label, width=width, stream=stream)
    try:
        for item in iterable:
            yield item
            b.step()
    finally:
        b.close()
