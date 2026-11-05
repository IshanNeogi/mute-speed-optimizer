"""Deterministically rebuild the full-order (full route resolution, e.g. 583 waypoints) result
from a run's saved outputs -- NO re-optimization. The GA search is the only non-reproducible step
in a run; everything downstream of its output (the optimized_speed control-point array) is a pure
function of (route geometry, mammal tracks, ship/weight config, TL model) and can always be
regenerated exactly, as many times as you want, from:

  - run_manifest.json (the exact RunConfig used)
  - seed_results.json (or summary.json) -- just needs one seed's `optimized_speed` array
  - data/baseline_route.csv, data/mammal_paths.csv (already fixed, versioned inputs)

This is the "don't store the full-order CSV every run" design: materialize it on demand for the
one case you're about to plot/publish, from the small amount of data that's always saved.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Geod

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from route_utils.route_utils import cumulative_distance_m, downsample_route, upsample_speed
from source_model.jomopans_model import jomopans_model
from tl_model.tl_model import TLModelConfig, query_tl

GEOD = Geod(ellps="WGS84")
F_DECIDECADE = np.geomspace(10.0, 20000.0, 2000)   # same convention as objective.py's F_FINE


def load_full_resolution_route(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    lat = np.array([float(r["lat"]) for r in rows])
    lon = np.array([float(r["lon"]) for r in rows])
    speed = np.array([float(r["speed_kn_estimated"]) for r in rows])
    return lat, lon, speed


def load_mammal(csv_path, mammal_id):
    rows = list(csv.DictReader(open(csv_path)))
    track = sorted((int(r["minute"]), float(r["lat"]), float(r["lon"]))
                   for r in rows if int(r["mammal_id"]) == mammal_id)
    return np.array([t[0] for t in track], float), np.array([t[1] for t in track]), np.array([t[2] for t in track])


def reconstruct_full_order(manifest: dict, optimized_speed, data_root: Path, simulation_time_min: float = 0.0):
    """Returns a list of row dicts, one per full-resolution waypoint: waypoint_idx, cum_dist_nm,
    time_min, lat, lon, speed_kn, then per mammal: distance_km_<id>, received_level_db_<id>,
    cumulative_SEL_db_<id> (running total up to and including this waypoint)."""
    ship = manifest["ship"]
    opt = manifest["optimizer"]
    tl_raw = manifest["tl_model"]
    tl_config = TLModelConfig(kind=tl_raw["kind"], light_model_dir=tl_raw["light_model_dir"],
                               full_model_dir=tl_raw["full_model_dir"],
                               full_n_components=tl_raw.get("full_n_components", 10))
    mammal_ids = manifest["mammal_ids"]
    received_level_mode = opt.get("received_level_mode", "peak_band")   # match the run this reconstructs

    lat_full, lon_full, echo_speed_full = load_full_resolution_route(data_root / manifest["data"]["baseline_route_csv"])
    cum_dist_full_m = cumulative_distance_m(lat_full, lon_full)
    cum_dist_full_nm = cum_dist_full_m / 1852.0

    # same deterministic control-point selection the GA ran against (function of lat/lon + n_target only)
    _, cum_dist_m, _, _, _, cum_dist_cp_m = downsample_route(lat_full, lon_full, echo_speed_full, opt["n_waypoints"])
    speed_full = upsample_speed(cum_dist_full_m, cum_dist_cp_m, np.asarray(optimized_speed))

    seg_dist_nm = np.diff(cum_dist_full_nm)
    avg_speed = np.clip((speed_full[:-1] + speed_full[1:]) / 2.0, 1e-3, None)
    seg_time_hr = seg_dist_nm / avg_speed
    cum_time_hr = np.concatenate([[0.0], np.cumsum(seg_time_hr)])
    half_seg_weight_hr = np.zeros_like(cum_time_hr)
    half_seg_weight_hr[0] = seg_time_hr[0] / 2.0
    half_seg_weight_hr[-1] = seg_time_hr[-1] / 2.0
    if len(seg_time_hr) > 1:
        half_seg_weight_hr[1:-1] = (seg_time_hr[:-1] + seg_time_hr[1:]) / 2.0
    abs_time_min = simulation_time_min + cum_time_hr * 60.0

    mammal_tracks = {mid: load_mammal(data_root / manifest["data"]["mammal_paths_csv"], mid) for mid in mammal_ids}
    mammal_lat = {mid: np.interp(abs_time_min, t[0], t[1]) for mid, t in mammal_tracks.items()}
    mammal_lon = {mid: np.interp(abs_time_min, t[0], t[2]) for mid, t in mammal_tracks.items()}

    n = len(lat_full)
    cumulative_energy = {mid: 0.0 for mid in mammal_ids}
    rows = []
    for i in range(n):
        ls_bands, _ = jomopans_model(F_DECIDECADE, ship["ship_class"], speed_full[i], ship["ship_length_ft"])
        row = {
            "waypoint_idx": i, "cum_dist_nm": round(float(cum_dist_full_nm[i]), 4),
            "time_min": round(float(abs_time_min[i]), 3), "lat": float(lat_full[i]), "lon": float(lon_full[i]),
            "speed_kn": round(float(speed_full[i]), 4),
        }
        for mid in mammal_ids:
            m_lat, m_lon = mammal_lat[mid][i], mammal_lon[mid][i]
            _, _, dist_m = GEOD.inv(lon_full[i], lat_full[i], m_lon, m_lat)
            tl = query_tl(tl_config, lat_full[i], lon_full[i], np.array([m_lat]), np.array([m_lon]))[0]
            nl_bands = np.clip(ls_bands - tl, 0.0, None)
            if received_level_mode == "peak_band":
                nl = float(np.max(nl_bands))
            else:
                nl = float(10 * np.log10(np.clip(np.sum(10 ** (nl_bands / 10.0)), 1e-30, None)))
            cumulative_energy[mid] += half_seg_weight_hr[i] * 10 ** (nl / 10.0)
            sel_cum = 10 * np.log10(max(cumulative_energy[mid], 1e-30))
            row[f"distance_km_{mid}"] = round(float(dist_m) / 1000.0, 4)
            row[f"received_level_db_{mid}"] = round(nl, 3)
            row[f"cumulative_sel_db_{mid}"] = round(float(sel_cum), 3)
        rows.append(row)
    return rows


def write_full_order_csv(rows, out_path):
    if not rows:
        raise ValueError("no rows to write")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def reconstruct_from_results_dir(results_dir: Path, data_root: Path, seed: int = None, out_path: Path = None):
    """Convenience wrapper: reads run_manifest.json + seed_results.json (or summary.json's
    selected seed if `seed` is None) from `results_dir`, writes the full-order CSV."""
    results_dir = Path(results_dir)
    manifest = json.load(open(results_dir / "run_manifest.json"))

    if seed is not None:
        seed_results = json.load(open(results_dir / "seed_results.json"))
        match = [r for r in seed_results if r.get("seed") == seed]
        if not match:
            raise ValueError(f"seed {seed} not found in {results_dir / 'seed_results.json'}")
        chosen = match[0]
    else:
        summary = json.load(open(results_dir / "summary.json"))
        chosen = summary["selected_seed"]
        if chosen is None:
            raise ValueError(f"no feasible selected seed in {results_dir / 'summary.json'}")

    rows = reconstruct_full_order(manifest, chosen["optimized_speed"], data_root)
    out_path = out_path or (results_dir / f"full_order_seed{chosen['seed']}.csv")
    write_full_order_csv(rows, out_path)
    return out_path
