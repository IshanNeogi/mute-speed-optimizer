"""Run: python -m objective.tests.test_objective  (from handover/)

Uses the real data/baseline_route.csv + data/mammal_paths.csv shipped with this repo, and the
external light TL model (skips cleanly if not found locally -- see data/README.md).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from objective.objective import evaluate_population_multi
from route_utils.route_utils import downsample_route
from tl_model.tl_model import TLModelConfig

DATA = ROOT / "data"
N_WAYPOINTS = 30
SHIP_CLASS, SHIP_LENGTH_FT = "Other", 697.18
MIN_V, MAX_V = 8.0, 14.5

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


def test_baseline_sel_matches_known_value_mammal_12():
    """Cross-check from sandbox/optimizer/LOG.md (2026-10-01): mammal 12's baseline (echo_ais
    speed) SEL on this exact route/data was independently computed as 108.03dB by the 30-waypoint
    objective pipeline, itself cross-validated to within 0.08dB of a fully independent
    583-waypoint/1-min pipeline. This test guards against a silent regression during the port."""
    if not _light_available():
        print("SKIP test_baseline_sel_matches_known_value_mammal_12 (no local TL model)")
        return
    wp_lat, wp_lon, speed_cp = load_route()
    track = load_mammal(12)
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)

    x = speed_cp[np.newaxis, :]  # pop=1, the baseline profile itself
    result = evaluate_population_multi(x, wp_lat, wp_lon, [track], SHIP_CLASS, SHIP_LENGTH_FT,
                                        w_time=0.0, sel_max_weight=0.5, tl_config=tl_cfg)
    sel = result["SEL_per_mammal"][0, 0]
    assert abs(sel - 108.03) < 0.5, f"expected ~108.03dB, got {sel:.2f}dB"


def test_slower_profile_is_quieter_and_slower_group_sel_max():
    """Monotonicity holds for SEL_max across a multi-mammal group (cross-check from LOG.md's
    constant_speed_profiles work, mammals 3/12/8: min/mid/max SEL_max = 104.09/110.74/116.74dB).
    NOT true for every single mammal in isolation -- a slower ship arrives at a later time, which
    can shift which part of a MOVING mammal's track it's actually closest to (the "no free
    waypoint" / timing-alignment finding, LOG.md 2026-10-01) -- so single-mammal SEL can be
    non-monotonic in speed. Group SEL_max/T_hr, averaged over several mammals, is the level this
    property was actually validated at."""
    if not _light_available():
        print("SKIP test_slower_profile_is_quieter_and_slower_group_sel_max (no local TL model)")
        return
    wp_lat, wp_lon, speed_cp = load_route()
    tracks = [load_mammal(mid) for mid in (3, 12, 8)]
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)

    x = np.stack([
        np.full(N_WAYPOINTS, MIN_V),
        np.full(N_WAYPOINTS, (MIN_V + MAX_V) / 2),
        np.full(N_WAYPOINTS, MAX_V),
    ])
    result = evaluate_population_multi(x, wp_lat, wp_lon, tracks, SHIP_CLASS, SHIP_LENGTH_FT,
                                        w_time=0.0, sel_max_weight=0.5, tl_config=tl_cfg)
    sel_max = result["SEL_max"]  # (3,) one per candidate
    T = result["T_hr"]
    assert sel_max[0] < sel_max[1] < sel_max[2], "slower should be quieter (group SEL_max), monotonically"
    assert T[0] > T[1] > T[2], "slower should take longer, monotonically"


def test_multi_mammal_sel_max_and_mean_consistent():
    if not _light_available():
        print("SKIP test_multi_mammal_sel_max_and_mean_consistent (no local TL model)")
        return
    wp_lat, wp_lon, speed_cp = load_route()
    tracks = [load_mammal(mid) for mid in (0, 4, 8, 12, 16)]
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)

    x = speed_cp[np.newaxis, :]
    result = evaluate_population_multi(x, wp_lat, wp_lon, tracks, SHIP_CLASS, SHIP_LENGTH_FT,
                                        w_time=0.0, sel_max_weight=0.5, tl_config=tl_cfg)
    sel_per_mammal = result["SEL_per_mammal"][:, 0]
    assert sel_per_mammal.shape == (5,)
    assert np.isclose(result["SEL_max"][0], sel_per_mammal.max())
    assert np.isclose(result["SEL_mean"][0], sel_per_mammal.mean())


if __name__ == "__main__":
    test_baseline_sel_matches_known_value_mammal_12()
    test_slower_profile_is_quieter_and_slower_group_sel_max()
    test_multi_mammal_sel_max_and_mean_consistent()
    print("objective: all tests passed (or skipped where the TL model is absent)")
