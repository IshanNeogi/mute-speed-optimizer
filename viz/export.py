"""One place for save-format policy: vector output (PDF/EPS) for print, consistent bbox/padding,
never a raster round-trip. EPS doesn't support alpha transparency -- figures in this kit avoid
relying on it, but export.py flags it rather than silently producing a broken EPS if one creeps in.
"""
from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig, name: str, out_dir, formats=("pdf", "eps")):
    """Writes <out_dir>/<name>.<fmt> for each format. Returns the list of paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        if fmt == "eps":
            _warn_if_alpha_used(fig, name)
        fig.savefig(path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        written.append(path)
    return written


def _warn_if_alpha_used(fig, name):
    for ax in fig.get_axes():
        for artist in ax.get_children():
            alpha = getattr(artist, "get_alpha", lambda: None)()
            if alpha is not None and alpha < 1.0:
                print(f"WARNING: figure '{name}' has an artist with alpha={alpha} -- "
                      f"EPS has no transparency support, this element may render solid/wrong "
                      f"in the .eps output. Check it, or export that figure as PDF only.")
                return
