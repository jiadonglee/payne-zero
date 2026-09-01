"""Every figure this repository produces, in one package.

Two audiences, kept apart because they have different rules:

``payne_zero_figures.paper``    The manuscript figures.  Vector PDF at A&A
                                column widths, Paper I house style, and every
                                number traceable to ``results/``.  These are
                                the ones that ship.

``payne_zero_figures.reports``  The lab reports and evidence figures produced
                                while the work was being done.  Screen-first
                                PNG and multi-page PDF, looser styling, and
                                free to reference superseded model checkpoints
                                -- which is exactly why they must never be a
                                source for a manuscript number.

Shared machinery lives in :mod:`payne_zero_figures.style` (palettes and
rcParams presets) and :mod:`payne_zero_figures.data` (artifact loading).
Build the manuscript figures with::

    MPLCONFIGDIR=/tmp/mpl python3 -m payne_zero_figures.paper.figures
"""

from __future__ import annotations

from . import data, style

__all__ = ["data", "style"]
