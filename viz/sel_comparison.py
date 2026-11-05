"""SEL comparison: horizontal grouped bars, one pair (echo_ais vs optimized) per mammal, with
dB-reduction + %-less-energy direct labels. Same layout validated earlier this session
(poster_results/make_poster_comparison.py), rebuilt here with the skill's fixed echo/optimized
roles and real LaTeX text instead of mathtext.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import palette
from .export import save_figure
from .style import tex_escape


def plot_sel_comparison(mammal_ids, echo_sel, optimized_sel, out_dir, name="sel_comparison",
                         context="poster", formats=("pdf", "eps"), title=None):
    """mammal_ids, echo_sel, optimized_sel: equal-length sequences, one entry per mammal."""
    echo_sel = np.asarray(echo_sel, dtype=float)
    optimized_sel = np.asarray(optimized_sel, dtype=float)
    n = len(mammal_ids)
    y = np.arange(n)
    w = 0.38

    figsize = (9, 1.2 * n + 2.5) if context == "poster" else (3.4, 0.8 * n + 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y + w / 2, echo_sel, height=w, color=palette.ECHO_AIS_COLOR, zorder=3, label="AIS (echo) voyage")
    ax.barh(y - w / 2, optimized_sel, height=w, color=palette.OPTIMIZED_COLOR, zorder=3, label="Optimized voyage")

    label_fontsize = 12 if context == "poster" else 7
    value_fontsize = 11 if context == "poster" else 6.5
    offset = max(echo_sel.max(), optimized_sel.max()) * 0.015
    for yi, v in zip(y + w / 2, echo_sel):
        ax.text(v + offset, yi, f"{v:.1f}", va="center", fontsize=value_fontsize, color=palette.INK_SECONDARY)
    for yi, v in zip(y - w / 2, optimized_sel):
        ax.text(v + offset, yi, f"{v:.1f}", va="center", fontsize=value_fontsize, color=palette.INK_SECONDARY)

    for i, (e, o) in enumerate(zip(echo_sel, optimized_sel)):
        reduction_db = e - o
        pct = 100 * (1 - 10 ** (-reduction_db / 10))
        sign = "-" if reduction_db >= 0 else "+"
        ax.annotate(f"{sign}{abs(reduction_db):.1f} dB\n({pct:.0f}\\% less energy)",
                    (max(e, o) + offset * 15, i), va="center", ha="center",
                    fontsize=value_fontsize, color=palette.INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"Mammal {i + 1}" for i in range(n)], fontsize=label_fontsize, color=palette.INK)
    ax.set_xlabel(r"Sound Exposure Level (dB re $1\,\mu\text{Pa}^2\text{s}$)")
    ax.set_xlim(0, max(echo_sel.max(), optimized_sel.max()) * 1.4)
    ax.grid(axis="x", zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    if title:
        ax.set_title(title)

    return save_figure(fig, name, out_dir, formats=formats)
