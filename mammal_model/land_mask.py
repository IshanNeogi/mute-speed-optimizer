"""Water mask and coordinate helpers for the Salish Sea study area.

Ported verbatim from backend/app/mammal/land_mask.py (logic unchanged; only the data path points
at this package's own ../data/ instead of backend/app/mammal/data/).

Masks are bool (N, N) arrays stored in ../data/*.npz (key "mask"), True = whales may not be there.
Row i <-> latitude LAT_MIN + i*dlat, column j <-> longitude LON_MIN + j*dlon with
dlat=(LAT_MAX-LAT_MIN)/(N-1), dlon=(LON_MAX-LON_MIN)/(N-1). N comes from the array's shape
(currently 4000: ~81 x 59 m cells, built from the real shoreline polygons).

  land_mask.npz    the shoreline itself (for display)
  blocked_mask.npz land, plus water shallower than 2m (GEBCO 2023), grown by a 1-cell safety
                   buffer -- the mask the whale model keeps whales out of

Coordinates: the simulation runs in UTM zone 10N (EPSG:32610) offsets [km] from the start point.
FROZEN scientific core -- see AGENTS.md.
"""
from pathlib import Path

import numpy as np
from pyproj import Transformer

from .constants import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, UTM_EPSG

_DATA = Path(__file__).resolve().parents[1] / "data"
_TO_UTM = Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)
_TO_LL = Transformer.from_crs(f"EPSG:{UTM_EPSG}", "EPSG:4326", always_xy=True)
_CACHE: dict = {}


def _load(name: str) -> np.ndarray:
    if name not in _CACHE:
        path = _DATA / name
        if not path.exists():
            raise FileNotFoundError(f"Mask {path} not found -- see data/README.md")
        mask = np.load(path)["mask"]
        mask.flags.writeable = False
        _CACHE[name] = mask
    return _CACHE[name]


def load_land_mask() -> np.ndarray:
    """The mask the whale model keeps whales out of: land + shallow water + safety buffer."""
    return _load("blocked_mask.npz")


def load_pure_land_mask() -> np.ndarray:
    """The shoreline only (no shallow-water rule, no buffer) -- for display."""
    return _load("land_mask.npz")


def ll_to_km(lat, lon, lat0: float, lon0: float):
    """(lat, lon) degrees -> (x_km, y_km) UTM offsets from (lat0, lon0)."""
    e0, n0 = _TO_UTM.transform(lon0, lat0)
    e, n = _TO_UTM.transform(np.asarray(lon, float), np.asarray(lat, float))
    return (e - e0) / 1000.0, (n - n0) / 1000.0


def km_to_ll(x_km, y_km, lat0: float, lon0: float):
    """(x_km, y_km) UTM offsets from (lat0, lon0) -> (lat, lon) in degrees."""
    e0, n0 = _TO_UTM.transform(lon0, lat0)
    lon, lat = _TO_LL.transform(e0 + np.asarray(x_km, float) * 1000.0, n0 + np.asarray(y_km, float) * 1000.0)
    return lat, lon


def on_land_vec(lats: np.ndarray, lons: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Vectorised mask lookup. Positions outside the study area take the value of the nearest edge cell."""
    if mask is None:
        return np.zeros(len(lats), dtype=bool)
    n_lat, n_lon = mask.shape
    i = np.clip(((lats - LAT_MIN) / (LAT_MAX - LAT_MIN) * (n_lat - 1)).astype(int), 0, n_lat - 1)
    j = np.clip(((lons - LON_MIN) / (LON_MAX - LON_MIN) * (n_lon - 1)).astype(int), 0, n_lon - 1)
    return mask[i, j]
