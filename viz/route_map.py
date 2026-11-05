"""Route map: ship track colored by speed (sequential magnitude encoding), optimized vs.
echo_ais, mammal tracks overlaid. Reads a full-order CSV (reconstruct.reconstruct_full_order's
output) for the optimized track; echo_ais's own speed comes from data/baseline_route.csv directly
(it needs no reconstruction -- it's not a GA output).

Land-mask background is optional: if sandbox/whale_motion's land mask isn't reachable (it's
outside this package, ~16MB, not copied in), the map still renders correctly on a plain surface.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import palette
from .export import save_figure
from .style import FIGSIZE_POSTER

ROOT = Path(__file__).resolve().parents[1]


def _try_load_land_mask():
    try:
        whale_sandbox = ROOT.parents[0] / "whale_motion"
        sys.path.insert(0, str(whale_sandbox))
        from improved.constants import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
        from improved.land_mask import load_pure_land_mask
        return load_pure_land_mask(), (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    except Exception:
        return None, None


def plot_route_map(full_order_rows, mammal_tracks: dict, out_dir, name="route_map",
                    context="poster", formats=("pdf", "eps"), title=None):
    """full_order_rows: list of dicts from reconstruct_full_order (needs lat/lon/speed_kn).
    mammal_tracks: {mammal_id: (times_min, lat, lon)} -- plotted as dashed tracks with a start
    marker, same convention as the sandbox's earlier poster route map."""
    lat = np.array([r["lat"] for r in full_order_rows])
    lon = np.array([r["lon"] for r in full_order_rows])
    speed = np.array([r["speed_kn"] for r in full_order_rows])

    fig, ax = plt.subplots(figsize=FIGSIZE_POSTER if context == "poster" else (7.0, 7.0))

    land_mask, extent = _try_load_land_mask()
    if land_mask is not None:
        ax.imshow(np.where(land_mask, 1.0, np.nan), extent=extent, origin="lower", cmap="Greys",
                  vmin=0, vmax=1, alpha=0.5, zorder=1)
    ax.set_facecolor("#dfeaf5")

    pts = np.array([lon, lat]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vmin, vmax = speed.min(), speed.max()
    lc = LineCollection(segs, array=(speed[:-1] + speed[1:]) / 2, cmap=palette.sequential_cmap(),
                        norm=plt.Normalize(vmin, vmax), linewidths=4.5 if context == "poster" else 2.5,
                        zorder=3)
    ax.add_collection(lc)

    track_grays = ["#4a4846", "#8c8b86", "#aaa9a3"]
    for i, (mid, (t, mlat, mlon)) in enumerate(mammal_tracks.items()):
        color = track_grays[i % len(track_grays)]
        ax.plot(mlon, mlat, color=color, lw=1.3, ls="--", alpha=0.85, zorder=4)
        ax.scatter(mlon[0], mlat[0], s=70, marker="o", facecolor="white", edgecolor=color, lw=1.6, zorder=5)
        ax.annotate(r"Mammal %d" % (i + 1), (mlon[0], mlat[0]), xytext=(8, 8), textcoords="offset points",
                    fontsize=10, color=palette.INK, zorder=6, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

    lon_pad = (lon.max() - lon.min()) * 0.08
    lat_pad = (lat.max() - lat.min()) * 0.08
    ax.set_xlim(lon.min() - lon_pad, lon.max() + lon_pad)
    ax.set_ylim(lat.min() - lat_pad, lat.max() + lat_pad)
    lat0 = (lat.min() + lat.max()) / 2.0
    ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
    ax.set_anchor("E")
    ax.set_xlabel("Longitude ($^\\circ$)")
    ax.set_ylabel("Latitude ($^\\circ$)")
    if title:
        ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.12)
    cbar = fig.colorbar(lc, cax=cax)
    cbar.set_label("Speed (kn)")

    return save_figure(fig, name, out_dir, formats=formats)
