"""Received level and cumulative SEL vs. voyage time, one panel per mammal (small multiples --
categorical identity, not magnitude, so each mammal gets its own fixed slot from
palette.CATEGORICAL rather than a sequential ramp). Reads a full-order CSV
(reconstruct_full_order's output): needs time_min, received_level_db_<id>, cumulative_sel_db_<id>
per mammal_id.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import palette
from .export import save_figure


def plot_received_level_timeline(full_order_rows, mammal_ids, out_dir, name="received_level_timeline",
                                   context="poster", formats=("pdf", "eps")):
    time_min = np.array([r["time_min"] for r in full_order_rows])
    n = len(mammal_ids)
    figsize = (9, 2.2 * n) if context == "poster" else (3.4, 1.6 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]

    for i, mid in enumerate(mammal_ids):
        ax = axes[i]
        color = palette.CATEGORICAL[i % len(palette.CATEGORICAL)]
        nl = np.array([r[f"received_level_db_{mid}"] for r in full_order_rows])
        ax.plot(time_min, nl, color=color, lw=2.0)
        ax.set_ylabel(f"Mammal {i + 1}\nRL (dB)")
        ax.grid(True)
    axes[-1].set_xlabel("Voyage time (min)")
    fig.suptitle("Received level vs. voyage time")
    fig.tight_layout()

    return save_figure(fig, name, out_dir, formats=formats)


def plot_cumulative_sel_timeline(full_order_rows, mammal_ids, out_dir, name="cumulative_sel_timeline",
                                  context="poster", formats=("pdf", "eps")):
    """All mammals on ONE axis (not small multiples) -- cumulative SEL trajectories are
    comparable across mammals on the same scale, unlike instantaneous received level, so one
    shared axis with a legend is the right form here."""
    time_min = np.array([r["time_min"] for r in full_order_rows])
    figsize = (9, 5.5) if context == "poster" else (3.4, 2.6)
    fig, ax = plt.subplots(figsize=figsize)

    for i, mid in enumerate(mammal_ids):
        color = palette.CATEGORICAL[i % len(palette.CATEGORICAL)]
        cum = np.array([r[f"cumulative_sel_db_{mid}"] for r in full_order_rows])
        ax.plot(time_min, cum, color=color, lw=2.5, label=f"Mammal {i + 1}")

    ax.set_xlabel("Voyage time (min)")
    ax.set_ylabel(r"Cumulative SEL (dB re $1\,\mu\text{Pa}^2\text{s}$)")
    ax.legend(loc="lower right")
    ax.grid(True)

    return save_figure(fig, name, out_dir, formats=formats)
