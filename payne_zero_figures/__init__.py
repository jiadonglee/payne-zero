"""Lab-report and evidence visualizations for initializer research.

``payne_zero_figures.reports`` contains screen-first PNG and multi-page PDF
reports.  Shared machinery lives in :mod:`payne_zero_figures.style` (palettes
and rcParams presets) and :mod:`payne_zero_figures.data` (artifact loading).

The private manuscript and its dedicated figure-generation package are not
part of this public branch.
"""

from __future__ import annotations

from . import data, style

__all__ = ["data", "style"]
