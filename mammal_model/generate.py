"""Public entry point: a starting location/heading/speed -> a projected mammal track.

Ported from backend/app/mammal/{generate_path.py, predict.py}, trimmed to the explicit-start-state
path only (production's "watson_prior" random-spawn source -- a different, KDE-based sampling of
a START location from historical sighting density -- is NOT ported here; this package only covers
"I already know where/how fast/which way a mammal is moving, project it forward," which is what
production's predict.py itself is for too). The underlying physics
(mammal_model.motion.simulate_paths) is unchanged -- FROZEN scientific core, see AGENTS.md.
"""
from typing import Literal, Optional, TypedDict

import numpy as np

from .constants import ALPHA, SIGMA
from .land_mask import load_land_mask
from .motion import find_most_probable_path, simulate_paths

Selection = Literal["random", "medoid"]


class PathPoint(TypedDict):
    minute: int
    lat: float
    lon: float


def heading_speed_to_vx_vy(heading_deg: float, speed_kmh: float):
    """heading_deg: compass bearing, clockwise from north (the convention production's predict.py
    uses -- treated as UTM-grid north; the grid/true-north offset is <~1.5deg in the Salish Sea
    and is ignored, same as production). speed_kmh: >=0. Returns (vx, vy) km/h, UTM (east, north)."""
    if speed_kmh < 0:
        raise ValueError(f"speed_kmh must be >= 0, got {speed_kmh}")
    h = np.deg2rad(heading_deg)
    return float(speed_kmh * np.sin(h)), float(speed_kmh * np.cos(h))


def generate_mammal_track(
    lat_deg: float,
    lon_deg: float,
    heading_deg: float,
    speed_kmh: float,
    horizon_min: int,
    seed: Optional[int] = None,
    selection: Selection = "random",
    n_realizations: int = 1000,
    history_latlon: Optional[np.ndarray] = None,
    check_on_land: bool = True,
) -> list[PathPoint]:
    """One projected mammal track, one point per minute, starting from an explicit state.

    lat_deg, lon_deg: start position [deg, WGS84]. Checked against the land/shallow-water mask
      first (see check_on_land) -- a start on land makes every downstream number meaningless.
    heading_deg: direction of travel [deg, clockwise from north, compass convention].
    speed_kmh: speed over ground [km/h]. (Project convention elsewhere is knots for ship speed --
      this model's own units are km/h per the source paper; convert yourself, e.g. kn*1.852, or
      use the generate_mammal.py CLI's --speed-kn which does it for you.)
    horizon_min: projection horizon [min], 1-minute steps; returns horizon_min+1 points (minute 0
      = the input position).
    seed: RNG seed; None draws fresh randomness each call (not reproducible).
    selection: "random" (default) -- one simulated realization of the model. "medoid" -- simulate
      n_realizations and return the one closest to their mean path (production's legacy ../MUTE
      behavior; biased toward low-movement paths, see production's generate_path.py docstring).
    history_latlon: optional (k, 2) [lat, lon deg] positions just before the start, oldest first,
      to continue a directional tendency across a restart (only relevant if this mammal already
      had a track before this call -- leave None for a brand-new mammal).
    check_on_land: raise ValueError if (lat_deg, lon_deg) is on land/shallow water, before running
      anything -- on by default; production's own spawn-input validation does the same check.

    Never stalls or falls back to a default track: errors propagate to the caller.
    """
    if horizon_min < 1:
        raise ValueError(f"horizon_min must be >= 1, got {horizon_min}")
    land_mask = load_land_mask()
    if check_on_land:
        from .land_mask import on_land_vec
        if bool(on_land_vec(np.array([lat_deg]), np.array([lon_deg]), land_mask)[0]):
            raise ValueError(f"start position ({lat_deg}, {lon_deg}) is on land/shallow water per "
                              f"the blocked mask -- pick a water point, or pass check_on_land=False "
                              f"to bypass (not recommended).")

    vx0, vy0 = heading_speed_to_vx_vy(heading_deg, speed_kmh)
    rng = np.random.default_rng(seed)
    n = 1 if selection == "random" else n_realizations
    paths = simulate_paths(lat_deg, lon_deg, vx0, vy0, n, land_mask, rng, horizon_min, history_latlon)
    best = paths[0] if selection == "random" else find_most_probable_path(paths)[1]
    return [
        {"minute": minute, "lat": round(float(lat), 6), "lon": round(float(lon), 6)}
        for minute, (lat, lon) in enumerate(best)
    ]
