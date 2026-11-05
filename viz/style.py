"""matplotlib rcParams for every figure in this kit: real LaTeX text (not mathtext -- confirmed
working on this machine after installing cm-super + dvipng, 2026-11-05) and consistent
sizing/spacing. Call apply_style() once, before building any figure.
"""
import matplotlib

from . import palette

# Poster: large, bold, legible from a few feet away. Journal: single-column width, smaller text.
FIGSIZE_POSTER = (9, 9)
FIGSIZE_POSTER_WIDE = (11, 6)
FIGSIZE_JOURNAL_COLUMN = (3.4, 2.8)     # ~single IEEE/APA column width in inches
FIGSIZE_JOURNAL_FULL_WIDTH = (7.0, 4.0)  # full page width


def apply_style(context="poster"):
    """context: "poster" (large text, thicker lines) or "journal" (compact, thinner lines)."""
    matplotlib.use("Agg")
    base_fontsize = 14 if context == "poster" else 9
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 2,
        "ytick.labelsize": base_fontsize - 2,
        "legend.fontsize": base_fontsize - 1,
        "axes.edgecolor": palette.AXIS,
        "axes.labelcolor": palette.INK_SECONDARY,
        "xtick.color": palette.MUTED,
        "ytick.color": palette.MUTED,
        "text.color": palette.INK,
        "axes.facecolor": palette.SURFACE,
        "figure.facecolor": palette.SURFACE,
        "savefig.facecolor": palette.SURFACE,
        "axes.grid": True,
        "grid.color": palette.GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.5 if context == "poster" else 1.5,
        "lines.markersize": 8 if context == "poster" else 5,
        "legend.frameon": False,
    })


def tex_escape(s: str) -> str:
    """Escape the handful of characters LaTeX treats specially, for labels built from runtime
    data (mammal IDs, filenames) rather than hand-written math strings."""
    replacements = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s
