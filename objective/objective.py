"""Multi-mammal speed objective -- every candidate is scored against ALL mammals simultaneously,
every generation (no per-mammal GA runs, no superimposition, no journey-position-dependent
"currently worst" cycling -- that approach was tried and abandoned, see sandbox/optimizer/LOG.md
2026-10-01: combining separately-optimized per-mammal profiles via pointwise-min broke
self-consistency, with one mammal's SEL regressing +8.2dB after re-scoring against its real track).

    noise_cost(x) = sel_max_weight * max_m(SEL_m(x)) + (1 - sel_max_weight) * mean_m(SEL_m(x))
    cost(x) = noise_cost(x) + w_time * T(x)

Source level is computed once per candidate (ship-speed-only, independent of mammal identity) and
reused across all mammals; all mammals' TL queries are batched into one model call.

Received-level mode (`received_level_mode`, default "peak_band"): how the per-waypoint received
level is formed from the 33 JOMOPANS decidecade bands, after the same single scalar TL value is
subtracted from each band (the light TL model has no frequency resolution of its own -- see
tl_model/tl_model.py):
  - "peak_band" (default): NL = max over bands. Matches production's speed_objective.py exactly
    (confirmed, sandbox/optimizer/LOG.md 2026-10-01: "NL_peak = max over bands -- the actual
    optimizer cost quantity") and everything in SCIENTIFIC_FINDINGS.md was validated against this.
  - "broadband": NL = 10*log10(sum(10**(band/10))), i.e. energy-summed across all 33 bands instead
    of taking the loudest one. A real change to the optimizer's own cost function, not just a
    reporting difference -- the GA searches against broadband SEL end to end, so results under
    this mode are NOT the same validated numbers as "peak_band" and have not themselves been
    separately validated the way peak_band has. Chosen deliberately (not production's convention)
    -- see the paper's own methods text for the physical justification.

TL model choice (light vs. full) is config-driven via tl_model.TLModelConfig. Only "light" is
actually validated for this objective (the GA runs behind every poster/sandbox result used it) --
"full" is accepted but reduced to mean-across-29-bands per mammal/waypoint, since JOMOPANS' 33
decidecade bands don't line up with the full model's 29 bands (see tl_model/tl_model.py docstring);
this is a coarser approximation than "light" was ever benchmarked at, use it knowingly.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route_utils.route_utils import segment_distances_nm, segment_times_hr, waypoint_duration_weights
from source_model.jomopans_model import jomopans_model_matrix
from tl_model.tl_model import TLModelConfig, load_tl_model

F_FINE = np.geomspace(10.0, 20000.0, 4000)


def mammal_position_at_multi(abs_time_min, mammal_tracks):
    """mammal_tracks: list of (times_min, lat, lon) per mammal. Returns (M, pop, n_wp) lat/lon."""
    M = len(mammal_tracks)
    pop, n_wp = abs_time_min.shape
    lat_out = np.zeros((M, pop, n_wp))
    lon_out = np.zeros((M, pop, n_wp))
    flat = abs_time_min.ravel()
    for m, (t, lat, lon) in enumerate(mammal_tracks):
        lat_out[m] = np.interp(flat, t, lat).reshape(pop, n_wp)
        lon_out[m] = np.interp(flat, t, lon).reshape(pop, n_wp)
    return lat_out, lon_out


def evaluate_population_multi(x, wp_lat, wp_lon, mammal_tracks, ship_class, ship_length_ft,
                               w_time, sel_max_weight, tl_config: TLModelConfig,
                               simulation_time_min=0.0, received_level_mode="peak_band"):
    """x: (pop, n_wp) candidate speeds (kn). mammal_tracks: list of (times_min, lat, lon).
    received_level_mode: "peak_band" (default, validated) or "broadband" -- see module docstring.
    Returns dict: cost, T_hr, SEL_per_mammal (pop-indexed per mammal, shape (M, pop)), SEL_max,
    SEL_mean, NL_per_mammal (per-waypoint received level under the chosen mode)."""
    if received_level_mode not in ("peak_band", "broadband"):
        raise ValueError(f"received_level_mode must be 'peak_band' or 'broadband', got {received_level_mode!r}")
    pop, n_wp = x.shape
    M = len(mammal_tracks)
    seg_dist = segment_distances_nm(wp_lat, wp_lon)
    seg_time = segment_times_hr(x, seg_dist)
    cum_time_hr = np.concatenate([np.zeros((pop, 1)), np.cumsum(seg_time, axis=1)], axis=1)
    T_hr = cum_time_hr[:, -1]
    abs_time_min = simulation_time_min + cum_time_hr * 60.0

    mam_lat, mam_lon = mammal_position_at_multi(abs_time_min, mammal_tracks)   # (M, pop, n_wp)

    ship_lat_t = np.broadcast_to(wp_lat, (pop, n_wp))
    ship_lon_t = np.broadcast_to(wp_lon, (pop, n_wp))
    ship_lat_rep = np.broadcast_to(ship_lat_t, (M, pop, n_wp))
    ship_lon_rep = np.broadcast_to(ship_lon_t, (M, pop, n_wp))

    query = np.column_stack([ship_lat_rep.ravel(), ship_lon_rep.ravel(), mam_lat.ravel(), mam_lon.ravel()])
    model = load_tl_model(tl_config)
    tl_raw = model(query)
    if tl_config.kind == "light":
        tl = tl_raw.reshape(M, pop, n_wp)
    else:
        tl = tl_raw.mean(axis=1).reshape(M, pop, n_wp)   # full model: mean-across-bands reduction

    ls_bands, _ = jomopans_model_matrix(F_FINE, ship_class, x, ship_length_ft)   # (pop, n_wp, 33)
    nl_bands = np.maximum(0.0, ls_bands[np.newaxis, :, :, :] - tl[:, :, :, np.newaxis])   # (M, pop, n_wp, 33)
    if received_level_mode == "peak_band":
        nl = nl_bands.max(axis=3)   # (M, pop, n_wp)
    else:
        nl = 10.0 * np.log10(np.clip((10.0 ** (nl_bands / 10.0)).sum(axis=3), 1e-30, None))   # broadband energy-sum

    weight_hr = waypoint_duration_weights(seg_time)   # (pop, n_wp), shared across mammals
    E = weight_hr[np.newaxis, :, :] * 10.0 ** (nl / 10.0)
    SEL_per_mammal = 10.0 * np.log10(np.clip(E.sum(axis=2), 1e-30, None))   # (M, pop)

    SEL_max = SEL_per_mammal.max(axis=0)    # (pop,)
    SEL_mean = SEL_per_mammal.mean(axis=0)  # (pop,)
    noise_cost = sel_max_weight * SEL_max + (1.0 - sel_max_weight) * SEL_mean
    cost = noise_cost + w_time * T_hr

    return {"cost": cost, "T_hr": T_hr, "SEL_per_mammal": SEL_per_mammal, "SEL_max": SEL_max,
            "SEL_mean": SEL_mean, "NL_per_mammal": nl}
