"""Figure styles and palettes, in one place.

Before this module every plotting script in the repository carried its own
``_configure_style()`` and its own colour constants.  Eleven copies had drifted
into four genuinely different looks, and -- the reason this matters rather than
merely being untidy -- two of them use the *same variable names for different
colours*.  In the paper style ``LEARNED`` is burnt orange and ``PRODUCTION`` is
grey; in the evidence style ``LEARNED`` is blue and ``PRODUCTION`` is orange.
Collapsing them into one palette would have silently recoloured every figure in
the repository, so they are kept as separate, named palettes and the callers say
which one they mean.

Four presets, each reproducing exactly what its callers had inline:

``PAPER``      Paper I (arXiv:2607.24141) house style for the manuscript:
              serif body, no gridlines, four thin spines with inward ticks.
``REPORT``     Okabe-Ito on sans-serif, for the multi-page PDF lab reports.
``COOL_STAR``  The same idea one notch smaller, for the dense cool-star reports.
``EVIDENCE``   Off-white surface with a recessive grid, for the screen-first
              evidence PNGs.

Use::

    from payne_zero_figures import style
    style.configure("REPORT")
    style.configure("REPORT", **{"savefig.facecolor": "white"})
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

# --------------------------------------------------------------------------
# Palettes.  Each is self-contained; none of them share names by accident.
# --------------------------------------------------------------------------


class PaperPalette:
    """Paper I semantic roles.

    Near-black for the released/reference product, burnt orange for this
    paper's learned candidate, steel blue for exact-truth physics, brick red
    reserved for the acceptance bar and genuine failures.
    """

    INK = "#1a1a1a"
    INK_SECONDARY = "#404040"
    INK_MUTED = "#8a8a8a"
    LEARNED = "#b8621b"      # burnt orange: the learned (m,T) candidate
    PRODUCTION = "#4d4d4d"   # dark grey: the released six-field initialiser
    ORACLE = "#3a7ca5"       # steel blue: exact (m,T) + physics
    CRITICAL = "#a0342c"     # brick red: the acceptance bar, nothing else


class EvidencePalette:
    """Screen-first categorical slots for the evidence PNGs.

    The first three slots of the validated categorical palette, which is the
    set that clears the all-pairs CVD and normal-vision floors (worst pair
    dE 9.2 CVD / 24.0 normal).  Aqua sits below 3:1 on the light surface, so
    every series in these figures is directly labelled -- identity is never
    carried by hue alone.

    Note the clash with :class:`PaperPalette`: here ``LEARNED`` is blue and
    ``PRODUCTION`` is orange, the reverse of the manuscript's assignment.
    """

    LEARNED = "#2a78d6"      # categorical slot 1
    PRODUCTION = "#eb6834"   # slot 2
    ORACLE = "#1baf7a"       # slot 3
    CRITICAL = "#d03b3b"     # status: critical, the bar and its exceedances
    INK = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#8a8880"
    SURFACE = "#fcfcfb"


class OkabeIto:
    """The Okabe-Ito colourblind-safe set used by the PDF lab reports."""

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    PURPLE = "#CC79A7"
    SKY = "#56B4E9"
    RED = "#CC3311"
    BLACK = "#222222"
    GREY = "#777777"
    LIGHT_GREY = "#E7E7E7"


# The manuscript's spectral acceptance bar, in normalized flux.  Shared because
# both the paper figures and the evidence figures draw the same line.
BAR = 5.0e-3

# A&A column widths, in inches: 88 mm single column, 180 mm across both.
SINGLE = 88.0 / 25.4
DOUBLE = 180.0 / 25.4


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------


def register_fonts() -> None:
    """Make TeX Gyre Termes available to matplotlib.

    It ships with TeX Live but outside matplotlib's search path; registering it
    is what lets the manuscript figures match the txfonts body text.  A no-op
    where TeX Live is not installed, in which case the serif stack in ``PAPER``
    falls through to Times New Roman and then DejaVu Serif.
    """

    texlive = Path("/usr/local/texlive/2020/texmf-dist/fonts/opentype/public/tex-gyre")
    for name in ("texgyretermes-regular.otf", "texgyretermes-bold.otf",
                 "texgyretermes-italic.otf", "texgyretermes-bolditalic.otf"):
        path = texlive / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))


# --------------------------------------------------------------------------
# Presets.  Each dict is verbatim what its callers used to set inline.
# --------------------------------------------------------------------------

_P = PaperPalette

PAPER: dict = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "STIX Two Text",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "normal",
    "axes.titlecolor": _P.INK,
    "axes.labelcolor": _P.INK,
    "axes.edgecolor": _P.INK,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.color": _P.INK,
    "ytick.color": _P.INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.labelcolor": _P.INK_SECONDARY,
    "ytick.labelcolor": _P.INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "lines.linewidth": 1.1,
}

REPORT: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COOL_STAR: dict = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

EVIDENCE: dict = {
    "figure.facecolor": EvidencePalette.SURFACE,
    "axes.facecolor": EvidencePalette.SURFACE,
    "savefig.facecolor": EvidencePalette.SURFACE,
    "font.size": 9,
    "axes.labelcolor": EvidencePalette.INK_SECONDARY,
    "axes.edgecolor": EvidencePalette.INK_MUTED,
    "axes.linewidth": 0.8,
    "xtick.color": EvidencePalette.INK_SECONDARY,
    "ytick.color": EvidencePalette.INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": EvidencePalette.INK,
    "legend.frameon": False,
    "legend.fontsize": 8,
}

PRESETS: dict[str, dict] = {
    "PAPER": PAPER,
    "REPORT": REPORT,
    "COOL_STAR": COOL_STAR,
    "EVIDENCE": EVIDENCE,
}


def configure(preset: str, **overrides) -> None:
    """Apply a named preset, plus any per-caller rcParams on top.

    The overrides exist because a couple of callers differ from their preset by
    exactly one key -- one report wants a white ``savefig.facecolor``, another a
    smaller title -- and reproducing that difference is cheaper and safer than
    minting a near-duplicate preset for each.
    """

    if preset not in PRESETS:
        raise KeyError(f"unknown style preset {preset!r}; "
                       f"expected one of {sorted(PRESETS)}")
    if preset == "PAPER":
        register_fonts()
    matplotlib.rcParams.update(PRESETS[preset])
    if overrides:
        matplotlib.rcParams.update(overrides)


# --------------------------------------------------------------------------
# Frame helpers
# --------------------------------------------------------------------------


def inward(ax) -> None:
    """Paper I frame: no grid, four thin spines, inward ticks on all sides."""

    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def recess(ax, *, grid_axis="y") -> None:
    """Recessive grid and axes: the data is the figure, the frame is not."""

    ax.grid(axis=grid_axis, color=EvidencePalette.INK_MUTED, alpha=0.22,
            linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def top_legend(fig, handles, labels, *, ncol, y=1.0, fontsize=7):
    """One horizontal, frameless legend row above the figure, as in Paper I."""

    return fig.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                      fontsize=fontsize, handlelength=1.6, columnspacing=1.4)


def panel_label(ax, label: str, *, x=-0.12, y=1.06, fontsize=11) -> None:
    """Bold panel letter outside the axes, as the PDF reports place it."""

    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold",
            fontsize=fontsize, va="top")
