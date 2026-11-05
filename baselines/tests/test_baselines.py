"""Run: python -m baselines.tests.test_baselines  (from handover/)"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from baselines.baselines import compute_heuristic_baselines, select_best_seed
from route_utils.route_utils import downsample_route
from tl_model.tl_model import TLModelConfig

DATA = ROOT / "data"
N_WAYPOINTS = 30
SHIP_CLASS, SHIP_LENGTH_FT = "Other", 697.18
MIN_V, MAX_V = 8.0, 14.5   # compute_heuristic_baselines' defaults

def _resolve_light_dir():
    """MUTE_LIGHT_MODEL_DIR (same env var run.py/config.py use) wins if set. Falls back to
    MUTE_ROOT + the known subpath, or a dev-machine-only directory guess -- which must NOT crash
    on a machine with a different directory layout (this used to raise IndexError on a fresh
    clone outside the original dev machine's exact nesting -- found and fixed 2026-11-05)."""
    direct = os.environ.get("MUTE_LIGHT_MODEL_DIR")
    if direct:
        return direct
    mute_root_env = os.environ.get("MUTE_ROOT")
    if mute_root_env:
        return str(Path(mute_root_env) / "v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq")
    try:
        guess = ROOT.parents[2] / "MUTE"
    except IndexError:
        return None
    return str(guess / "v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq")


LIGHT_DIR = _resolve_light_dir()


def _light_available():
    return LIGHT_DIR is not None and Path(LIGHT_DIR).exists()


def load_route():
    rows = list(csv.DictReader(open(DATA / "baseline_route.csv")))
    lat = np.array([float(r["lat"]) for r in rows])
    lon = np.array([float(r["lon"]) for r in rows])
    speed = np.array([float(r["speed_kn_estimated"]) for r in rows])
    _, _, wp_lat, wp_lon, speed_cp, _ = downsample_route(lat, lon, speed, N_WAYPOINTS)
    return wp_lat, wp_lon, speed_cp


def load_mammal(mammal_id):
    rows = list(csv.DictReader(open(DATA / "mammal_paths.csv")))
    track = sorted((int(r["minute"]), float(r["lat"]), float(r["lon"]))
                   for r in rows if int(r["mammal_id"]) == mammal_id)
    return np.array([t[0] for t in track], float), np.array([t[1] for t in track]), np.array([t[2] for t in track])


def test_heuristic_baselines_monotonic_group_sel_max():
    """Cross-check from LOG.md (constant_speed_profiles, mammals 3/12/8):
    min=104.09, mid=110.74, max=116.74dB SEL_max, T_hr 15.33/10.90/8.46 -- monotonic both ways."""
    if not _light_available():
        print("SKIP test_heuristic_baselines_monotonic_group_sel_max (no local TL model)")
        return
    wp_lat, wp_lon, echo_speed = load_route()
    tracks = [load_mammal(mid) for mid in (3, 12, 8)]
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)
    baselines = compute_heuristic_baselines(wp_lat, wp_lon, echo_speed, tracks, SHIP_CLASS,
                                             SHIP_LENGTH_FT, sel_max_weight=0.5, tl_config=tl_cfg)
    sel_max_min = max(baselines["min_speed"]["SEL_per_mammal"])
    sel_max_mid = max(baselines["mid_speed"]["SEL_per_mammal"])
    sel_max_max = max(baselines["max_speed"]["SEL_per_mammal"])
    assert sel_max_min < sel_max_mid < sel_max_max
    assert baselines["min_speed"]["T_hr"] > baselines["mid_speed"]["T_hr"] > baselines["max_speed"]["T_hr"]
    assert abs(sel_max_min - 104.09) < 1.0
    assert abs(baselines["min_speed"]["T_hr"] - 15.33) < 0.5


def test_custom_mid_v_overrides_the_default_midpoint():
    """mid_v is also the constraint floor (run.py resolves optimizer.constraint_voyage_time_hr
    into an equivalent speed and passes it through here) -- confirm an explicit value actually
    propagates instead of being silently ignored in favor of the (min_v+max_v)/2 default."""
    if not _light_available():
        print("SKIP test_custom_mid_v_overrides_the_default_midpoint (no local TL model)")
        return
    wp_lat, wp_lon, echo_speed = load_route()
    tracks = [load_mammal(12)]
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)

    default_mid = compute_heuristic_baselines(wp_lat, wp_lon, echo_speed, tracks, SHIP_CLASS,
                                               SHIP_LENGTH_FT, sel_max_weight=0.5, tl_config=tl_cfg)
    custom_mid = compute_heuristic_baselines(wp_lat, wp_lon, echo_speed, tracks, SHIP_CLASS,
                                              SHIP_LENGTH_FT, sel_max_weight=0.5, tl_config=tl_cfg,
                                              mid_v=9.5)
    assert default_mid["mid_speed"]["speed"][0] == (MIN_V + MAX_V) / 2.0
    assert custom_mid["mid_speed"]["speed"][0] == 9.5
    assert default_mid["mid_speed"]["speed"][0] != custom_mid["mid_speed"]["speed"][0]
    # min/max/echo_ais baselines (unrelated to mid_v) must be unaffected
    assert default_mid["min_speed"]["speed"] == custom_mid["min_speed"]["speed"]
    assert default_mid["echo_ais"]["SEL_per_mammal"] == custom_mid["echo_ais"]["SEL_per_mammal"]


def test_mid_v_out_of_bounds_raises():
    wp_lat, wp_lon, echo_speed = load_route()
    tracks = [load_mammal(12)]
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)
    try:
        compute_heuristic_baselines(wp_lat, wp_lon, echo_speed, tracks, SHIP_CLASS, SHIP_LENGTH_FT,
                                     sel_max_weight=0.5, tl_config=tl_cfg, mid_v=20.0)
        assert False, "expected ValueError for mid_v outside [min_v, max_v]"
    except ValueError:
        pass


def test_select_best_seed_prefers_echo_qualified_fastest():
    seeds = [
        {"seed": 1, "feasible": True, "beats_echo_ais": False, "T_hr": 8.0},
        {"seed": 2, "feasible": True, "beats_echo_ais": True, "T_hr": 9.5},
        {"seed": 3, "feasible": True, "beats_echo_ais": True, "T_hr": 9.0},  # fastest echo-qualified
        {"seed": 4, "feasible": False, "beats_echo_ais": True, "T_hr": 7.0},  # infeasible, ignored
    ]
    best, branch = select_best_seed(seeds)
    assert best["seed"] == 3
    assert branch == "beats_echo_ais"


def test_select_best_seed_falls_back_to_fastest_feasible():
    seeds = [
        {"seed": 1, "feasible": True, "beats_echo_ais": False, "T_hr": 8.0},
        {"seed": 2, "feasible": True, "beats_echo_ais": False, "T_hr": 9.5},
        {"seed": 3, "feasible": False, "beats_echo_ais": True, "T_hr": 1.0},
    ]
    best, branch = select_best_seed(seeds)
    assert best["seed"] == 1
    assert branch == "fastest_feasible_fallback"


def test_select_best_seed_none_feasible():
    seeds = [{"seed": 1, "feasible": False, "beats_echo_ais": False, "T_hr": 8.0}]
    best, branch = select_best_seed(seeds)
    assert best is None
    assert branch == "NONE_FEASIBLE"


if __name__ == "__main__":
    test_heuristic_baselines_monotonic_group_sel_max()
    test_custom_mid_v_overrides_the_default_midpoint()
    test_mid_v_out_of_bounds_raises()
    test_select_best_seed_prefers_echo_qualified_fastest()
    test_select_best_seed_falls_back_to_fastest_feasible()
    test_select_best_seed_none_feasible()
    print("baselines: all tests passed (or skipped where the TL model is absent)")
