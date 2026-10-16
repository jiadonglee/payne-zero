"""Shared figure style of Paper II: one typeface, one palette, thin marks, recessive axes.

Widths follow A&A: one column 88 mm, full page 180 mm. Colors are assigned by
role and never by order, so a start keeps its color in every figure. The
categorical set passes the colour-vision and contrast checks on white.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMN = 88 / 25.4
PAGE = 180 / 25.4

INK = "#1d1d1f"
SECONDARY = "#515154"
MUTED = "#86868b"
GRID = "#e8e8ed"
BLUE = "#0071e3"
ORANGE = "#e8590c"
GREEN = "#1d9a4a"
PURPLE = "#8944ab"

# Starts and models, with the line style that the text names.
ROLES = {
    "two_profile": dict(color=BLUE, ls="-", label="Two-profile network"),
    "six_profile": dict(color=ORANGE, ls="--", label="Six-profile network"),
    "nearest_0": dict(color=GREEN, ls="-", label="Nearest node, lower"),
    "nearest_1": dict(color=GREEN, ls=":", label="Nearest node, upper"),
    "marcs_node": dict(color=GREEN, ls="-", label="MARCS grid node"),
    "grey_convective": dict(color=PURPLE, ls="-", label="Grey–convective"),
    "exact_rebuild": dict(color=INK, ls=":", label="Exact rebuild"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica Neue",
    "mathtext.it": "Helvetica Neue:italic",
    "mathtext.bf": "Helvetica Neue:bold",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY,
    "ytick.labelcolor": SECONDARY,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel(ax, letter, x=0.0, y=1.02):
    """Panel letter above the top-left corner of the axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=INK)


def end_label(ax, x, y, text, dx=4, dy=0, ha="left", va="center"):
    """Direct label at a curve end; text stays in the secondary ink, the curve carries the colour."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=7, color=SECONDARY)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
