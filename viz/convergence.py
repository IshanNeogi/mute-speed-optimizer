"""GA convergence: best-feasible cost vs. generation, one line per seed. Reads `cost_history`
from seed_results.json entries (run.py captures this via minimize(..., save_history=True)).
"""
import matplotlib.pyplot as plt

from . import palette
from .export import save_figure


def plot_convergence(seed_results, out_dir, name="convergence", context="poster", formats=("pdf", "eps")):
    """seed_results: list of run.py's per-seed result dicts (each with 'seed' and 'cost_history')."""
    figsize = (9, 5.5) if context == "poster" else (3.4, 2.6)
    fig, ax = plt.subplots(figsize=figsize)

    for i, result in enumerate(seed_results):
        history = result.get("cost_history", [])
        if not history:
            continue
        color = palette.CATEGORICAL[i % len(palette.CATEGORICAL)]
        ax.plot(range(len(history)), history, color=color, lw=2.0, label=f"Seed {result['seed']}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best feasible cost")
    ax.legend(loc="upper right")
    ax.grid(True)

    return save_figure(fig, name, out_dir, formats=formats)
