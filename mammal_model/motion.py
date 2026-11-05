"""Project a mammal's forward path from a starting point (P_init).

Ported verbatim from backend/app/mammal/path_projection.py (logic unchanged). FROZEN scientific
core -- see AGENTS.md.

Ornstein-Uhlenbeck velocity process (Lin 2023 / Lin, Dowd & Joy 2025) with the paper's direction
rule, fallback branch (Algorithm 1): each step the O-U direction is replaced by the circular
average of itself and the *directional tendency* -- the direction of net travel over the last
TENDENCY_WINDOW_MIN minutes of the simulated track (the earliest known position is used while less
history exists). The rule is written in physical time, so it is the paper's rule exactly at a
3-min step and the production app's rule at 1 min (validated in production's VALIDATION.md, Stage
2b). Order per step follows the paper: move with the previous velocity, update the velocity, apply
the direction rule, then the coastal step for the next move.

Coordinates: UTM zone 10N offsets [km] from the start point (as in the paper). Coastal step: paper
appendix -- a move that would touch land or shallow water (mask, see land_mask.py) is replaced by
the best of COAST_N_CANDIDATES headings, each travelling the intended distance or stopping at the
boundary; the top two-thirds by distance are kept and the one closest to the directional tendency
is taken. A move is tested along its whole length, not only at its end.
"""
import numpy as np

from .constants import (
    ALPHA, COAST_KEEP_FRACTION, COAST_N_CANDIDATES, COAST_SAMPLE_SPACING_KM, DT_PROJECTION_HOURS,
    PULL_HALF_MIN, SIGMA, TENDENCY_WINDOW_MIN,
)
from .land_mask import km_to_ll, ll_to_km, on_land_vec


def _move_blocked(x, y, dx, dy, lat0, lon0, mask):
    """True where the straight move (x, y) -> (x + dx, y + dy) [km] touches a blocked cell (sampled along its length)."""
    r = np.hypot(dx, dy)
    k = max(1, int(np.ceil(r.max() / COAST_SAMPLE_SPACING_KM))) if len(r) else 1
    f = (np.arange(1, k + 1) / k)[None, :]
    lat, lon = km_to_ll((x[:, None] + dx[:, None] * f).ravel(), (y[:, None] + dy[:, None] * f).ravel(), lat0, lon0)
    return on_land_vec(lat, lon, mask).reshape(len(x), k).any(axis=1)


def _coastal_step(x, y, vx, vy, theta_ref, lat0, lon0, mask):
    """Paper appendix coastal step for every whale whose next move would be blocked.

    Returns (vx, vy, scale): re-aimed velocities (speed kept) and the fraction of the intended
    distance the next move covers (1 = unchanged; < 1 = stops at the boundary; 0 = no water
    candidate, hold in place)."""
    dt = DT_PROJECTION_HOURS
    scale = np.ones(len(x))
    blocked = _move_blocked(x, y, vx * dt, vy * dt, lat0, lon0, mask)
    ang = np.linspace(0, 2 * np.pi, COAST_N_CANDIDATES, endpoint=False)
    for i in np.where(blocked)[0]:
        spd = np.hypot(vx[i], vy[i])
        r = spd * dt
        if r < 1e-9:
            continue
        k = max(1, int(np.ceil(r / COAST_SAMPLE_SPACING_KM)))
        f = np.arange(1, k + 1) / k
        px = x[i] + r * np.cos(ang)[:, None] * f[None, :]
        py = y[i] + r * np.sin(ang)[:, None] * f[None, :]
        lat, lon = km_to_ll(px.ravel(), py.ravel(), lat0, lon0)
        blk = on_land_vec(lat, lon, mask).reshape(COAST_N_CANDIDATES, k)
        free = np.where(blk.any(axis=1), blk.argmax(axis=1), k)      # free samples before the boundary
        viable = np.where(free >= 1)[0]
        if viable.size == 0:
            scale[i] = 0.0
            continue
        order = viable[np.argsort(-free[viable], kind="stable")]
        keep = order[: int(np.ceil(COAST_KEEP_FRACTION * viable.size))]
        diff = np.abs(np.angle(np.exp(1j * (ang[keep] - theta_ref[i]))))
        chosen = keep[int(np.argmin(diff))]
        vx[i], vy[i] = spd * np.cos(ang[chosen]), spd * np.sin(ang[chosen])
        scale[i] = free[chosen] / k
    return vx, vy, scale


def simulate_paths(
    lat0: float,
    lon0: float,
    vx0,
    vy0,
    n_realizations: int,
    land_mask,
    rng: np.random.Generator,
    n_steps: int,
    history_latlon=None,
) -> np.ndarray:
    """Run n_realizations O-U trajectories for n_steps steps of DT_PROJECTION_HOURS from (lat0, lon0).

    vx0, vy0: initial velocity [km/h] (UTM east, north), scalar or (n_realizations,) arrays.
    history_latlon: optional (k, 2) [deg] positions of the whale just BEFORE the start (oldest
    first), used only to continue the directional tendency across a restart.
    Returns paths: (n_realizations, n_steps+1, 2) array, axis 2 = (lat, lon) degrees."""
    dt_min = DT_PROJECTION_HOURS * 60.0
    lag = int(round(TENDENCY_WINDOW_MIN / dt_min))
    w = 1.0 - 2.0 ** (-dt_min / PULL_HALF_MIN)          # weight on the tendency per step
    phi = 1.0 - ALPHA * DT_PROJECTION_HOURS
    eta = SIGMA * np.sqrt(DT_PROJECTION_HOURS)

    hist = np.zeros((0, 2)) if history_latlon is None else np.asarray(history_latlon, float)[-lag:]
    k = len(hist)
    hx, hy = ll_to_km(hist[:, 0], hist[:, 1], lat0, lon0) if k else (np.zeros(0), np.zeros(0))
    x = np.zeros((n_realizations, k + n_steps + 1))     # UTM km east of the start; columns [0, k) = history
    y = np.zeros((n_realizations, k + n_steps + 1))
    x[:, :k], y[:, :k] = hx, hy

    vx = np.broadcast_to(np.asarray(vx0, float), (n_realizations,)).copy()
    vy = np.broadcast_to(np.asarray(vy0, float), (n_realizations,)).copy()
    paths = np.zeros((n_realizations, n_steps + 1, 2))
    paths[:, 0, 0], paths[:, 0, 1] = lat0, lon0

    scale = np.ones(n_realizations)
    if land_mask is not None:  # the first move is protected like every later one
        vx, vy, scale = _coastal_step(x[:, k], y[:, k], vx, vy, np.arctan2(vy, vx), lat0, lon0, land_mask)

    for i in range(1, n_steps + 1):
        x[:, k + i] = x[:, k + i - 1] + vx * DT_PROJECTION_HOURS * scale
        y[:, k + i] = y[:, k + i - 1] + vy * DT_PROJECTION_HOURS * scale

        noise = rng.standard_normal((n_realizations, 2)) * eta
        vx = phi * vx + noise[:, 0]
        vy = phi * vy + noise[:, 1]

        j = k + max(i - lag, -k)                        # earliest known position while history is short
        dx, dy = x[:, k + i] - x[:, j], y[:, k + i] - y[:, j]
        theta_ou = np.arctan2(vy, vx)
        theta_tend = np.arctan2(dy, dx)
        speed = np.hypot(vx, vy)
        theta = np.arctan2(
            (1 - w) * np.sin(theta_ou) + w * np.sin(theta_tend),
            (1 - w) * np.cos(theta_ou) + w * np.cos(theta_tend),
        )
        undefined = (dx == 0.0) & (dy == 0.0)              # no net travel yet: tendency undefined, keep O-U direction
        theta = np.where(undefined, theta_ou, theta)
        vx, vy = speed * np.cos(theta), speed * np.sin(theta)

        if land_mask is not None:
            vx, vy, scale = _coastal_step(x[:, k + i], y[:, k + i], vx, vy, np.where(undefined, theta, theta_tend),
                                          lat0, lon0, land_mask)

        step_lat, step_lon = km_to_ll(x[:, k + i], y[:, k + i], lat0, lon0)
        paths[:, i, 0], paths[:, i, 1] = step_lat, step_lon

    return paths


def estimate_velocity_kmh(lat0: float, lon0: float, lat1: float, lon1: float):
    """(vx, vy) km/h [UTM east, north] implied by a one-minute lat/lon step, for seeding the next
    run's vx0/vy0 when extending a mammal's horizon mid-voyage. With the paper's order of
    operations the step from minute i-1 to i is exactly the velocity used for that move, so this
    recovers it."""
    dx_km, dy_km = ll_to_km(lat1, lon1, lat0, lon0)
    return float(dx_km / DT_PROJECTION_HOURS), float(dy_km / DT_PROJECTION_HOURS)


def find_most_probable_path(paths: np.ndarray):
    """Return (index, path) of the trajectory closest to the running centroid. paths: (n_realizations, n_steps+1, 2)"""
    centroid = paths.mean(axis=0)
    diff = paths - centroid[np.newaxis, :, :]
    total_sq = (diff ** 2).sum(axis=(1, 2))
    idx = int(np.argmin(total_sq))
    return idx, paths[idx]
