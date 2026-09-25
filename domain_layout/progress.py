"""Consistent progress reporting across every long-running stage.

A stage that takes hours and prints nothing is indistinguishable from a stage that has
hung - a distinction this project has had to make in a cluster log more than once. It also
matters for anyone reproducing the work: a visible counter is the difference between
"this is running" and "did I configure something wrong?".

Bars go to stderr so they never contaminate piped output, and disable themselves when
stderr is not a terminal, which is what a SLURM log is. Under SLURM the periodic INFO lines
that each stage already emits remain the record; the bar is for interactive runs.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def progress[T](
    items: Iterable[T],
    *,
    description: str,
    total: int | None = None,
    unit: str = "protein",
) -> Iterator[T]:
    """Wrap an iterable in a progress bar that behaves in a log file.

    Args:
        items: What to iterate.
        description: Shown to the left of the bar; name the stage, not the loop.
        total: Item count when it is not derivable from ``items``.
        unit: What is being counted, singular.
    """
    return tqdm(
        items,
        desc=description,
        total=total,
        unit=f" {unit}",
        file=sys.stderr,
        # A bar redrawing thousands of times fills a cluster log with control characters,
        # so it is silent unless someone is watching.
        disable=not sys.stderr.isatty(),
        dynamic_ncols=True,
        leave=False,
    )
