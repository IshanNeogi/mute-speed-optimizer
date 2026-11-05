"""Route geometry/kinematics helpers: downsample/upsample between full resolution and the GA's
coarse working resolution, segment distances/times, and half-segment waypoint duration weighting.

Ported verbatim from sandbox/optimizer/GA_test/resample.py + objective.py's segment_distances_nm/
segment_times_hr/waypoint_duration_weights -- these mirror backend/app/optimizer/orchestration.py's
own `_downsample_indices_by_distance` + `np.interp(full_s, control_dist, res.X)` pattern. No logic
changes.
"""
import numpy as np
from pyproj import Geod

GEOD = Geod(ellps="WGS84")


def cumulative_distance_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if len(lat) <= 1:
        return np.zeros(len(lat))
    _, _, seg = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])


def downsample_indices_by_distance(cum_dist: np.ndarray, n_target: int) -> list:
    n_rows = len(cum_dist)
    if n_target <= 1 or n_rows <= n_target or cum_dist[-1] <= 0:
        return list(range(n_rows))
    target_dist = np.linspace(0.0, cum_dist[-1], n_target)
    return sorted({int(np.argmin(np.abs(cum_dist - td))) for td in target_dist})


def downsample_route(lat: np.ndarray, lon: np.ndarray, speed: np.ndarray, n_target: int):
    """Full-res (lat, lon, speed) -> n_target control points, picked evenly by distance.
    Returns (idx, cum_dist_full, lat_cp, lon_cp, speed_cp, cum_dist_cp)."""
    cum_dist = cumulative_distance_m(lat, lon)
    idx = downsample_indices_by_distance(cum_dist, n_target)
    return idx, cum_dist, lat[idx], lon[idx], speed[idx], cum_dist[idx]


def upsample_speed(cum_dist_full: np.ndarray, cum_dist_cp: np.ndarray, speed_cp: np.ndarray) -> np.ndarray:
    """Control-point speeds -> full resolution, linear-interpolated by cumulative distance."""
    return np.interp(cum_dist_full, cum_dist_cp, speed_cp)


def segment_distances_nm(wp_lat: np.ndarray, wp_lon: np.ndarray) -> np.ndarray:
    _, _, dist_m = GEOD.inv(wp_lon[:-1], wp_lat[:-1], wp_lon[1:], wp_lat[1:])
    return dist_m / 1852.0


def route_distance_nm(wp_lat: np.ndarray, wp_lon: np.ndarray) -> float:
    """Total route length, straight-line-between-waypoints sum (same convention
    segment_distances_nm/T_hr kinematics use throughout this package -- NOT the full-resolution
    curved-coastline length, which is slightly longer, see reconstruct/'s discretization note)."""
    return float(segment_distances_nm(wp_lat, wp_lon).sum())


def constant_speed_for_voyage_time(wp_lat: np.ndarray, wp_lon: np.ndarray, time_hr: float) -> float:
    """The constant speed (kn) that covers this route in exactly `time_hr` hours."""
    if time_hr <= 0:
        raise ValueError(f"time_hr must be positive, got {time_hr}")
    return route_distance_nm(wp_lat, wp_lon) / time_hr


def segment_times_hr(x: np.ndarray, seg_dist_nm: np.ndarray) -> np.ndarray:
    """x: (pop, n_wp) candidate speeds (kn). Returns (pop, n_wp-1) segment times (hr)."""
    avg_speed = np.clip((x[:, :-1] + x[:, 1:]) / 2.0, 1e-3, None)
    return seg_dist_nm[np.newaxis, :] / avg_speed


def waypoint_duration_weights(seg_time_hr: np.ndarray) -> np.ndarray:
    """Half-segment trapezoidal weighting. (pop, n_wp-1) -> (pop, n_wp)."""
    pop, n_seg = seg_time_hr.shape
    n_wp = n_seg + 1
    w = np.zeros((pop, n_wp))
    w[:, 0] = seg_time_hr[:, 0] / 2.0
    w[:, -1] = seg_time_hr[:, -1] / 2.0
    if n_seg > 1:
        w[:, 1:-1] = (seg_time_hr[:, :-1] + seg_time_hr[:, 1:]) / 2.0
    return w
