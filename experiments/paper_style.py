"""Shared figure style of Paper II.

One Times face for text and mathematics, matching the A&A body text. Closed
dark axes with inward ticks on all four sides and minor ticks, no grid.
Widths follow A&A: one column 88 mm, full page 180 mm, so a figure is
included at its own size and 9 pt in the figure is 9 pt on the page.

Colours mark roles: the observation is charcoal, the primary model (the
two-profile start, the Payne Zero fit) orange, the alternative (the
six-profile start, the ASPCAP model) blue, further comparisons green and
purple. A start keeps its colour in every figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMN = 88 / 25.4
PAGE = 180 / 25.4

INK = "#1a1a1a"
SECONDARY = "#3d3d3d"
MUTED = "#7a7a7a"
LIGHT = "#bdbdbd"
ORANGE = "#d95f02"
BLUE = "#1f5fa8"
GREEN = "#1b8a5a"
PURPLE = "#7b3294"
PRIMARY = ORANGE
ALTERNATIVE = BLUE

# Starts and models, with the line style that the text names.
ROLES = {
    "two_profile": dict(color=PRIMARY, ls="-", label="Two-profile network"),
    "six_profile": dict(color=ALTERNATIVE, ls="--", label="Six-profile network"),
    "nearest_0": dict(color=GREEN, ls="-", label="Nearest node, lower"),
    "nearest_1": dict(color=GREEN, ls=":", label="Nearest node, upper"),
    "marcs_node": dict(color=GREEN, ls="-.", label="MARCS grid node"),
    "grey_convective": dict(color=PURPLE, ls="-", label="Grey–convective"),
    "exact_rebuild": dict(color=INK, ls=":", label="Exact rebuild"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX Two Text", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.labelpad": 3,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.borderaxespad": 0.4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel(ax, letter, x=0.0, y=1.02):
    """Panel letter above the top-left corner of the axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK)


def end_label(ax, x, y, text, dx=4, dy=0, ha="left", va="center", color=INK):
    """Direct label next to a mark, in place of a legend entry."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=8, color=color)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
