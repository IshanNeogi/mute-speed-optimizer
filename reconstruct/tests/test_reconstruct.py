"""Run: python -m reconstruct.tests.test_reconstruct  (from handover/)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from config import DataPaths, OptimizerConfig, RunConfig, ShipConfig, config_to_dict
from reconstruct.reconstruct import reconstruct_full_order, write_full_order_csv
from tl_model.tl_model import TLModelConfig

_MUTE_ROOT = Path(os.environ.get("MUTE_ROOT", ROOT.parents[2] / "MUTE"))
LIGHT_DIR = str(_MUTE_ROOT / "v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq")


def _light_available():
    return Path(LIGHT_DIR).exists()


def test_reconstruction_matches_control_point_count_and_is_monotonic_in_time():
    if not _light_available():
        print("SKIP test_reconstruction_matches_control_point_count_and_is_monotonic_in_time (no local TL model)")
        return
    cfg = RunConfig(
        ship=ShipConfig(), optimizer=OptimizerConfig(n_waypoints=30),
        tl_model=TLModelConfig(kind="light", light_model_dir=LIGHT_DIR),
        data=DataPaths(baseline_route_csv="data/baseline_route.csv", mammal_paths_csv="data/mammal_paths.csv"),
        mammal_ids=[12],
    )
    manifest = config_to_dict(cfg)
    optimized_speed = np.full(30, 11.25)   # stand-in "GA output" -- mid_speed everywhere

    rows = reconstruct_full_order(manifest, optimized_speed, ROOT)
    assert len(rows) == 583   # full baseline_route.csv resolution
    times = [r["time_min"] for r in rows]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:])), "time must be non-decreasing"
    speeds = [r["speed_kn"] for r in rows]
    assert all(abs(s - 11.25) < 0.01 for s in speeds), "flat control-point profile should upsample flat"
    assert "received_level_db_12" in rows[0]
    assert "cumulative_sel_db_12" in rows[0]
    # cumulative SEL must be non-decreasing (energy only accumulates)
    cum = [r["cumulative_sel_db_12"] for r in rows]
    cum_linear = [10 ** (c / 10.0) for c in cum]
    assert all(c2 >= c1 - 1e-6 for c1, c2 in zip(cum_linear, cum_linear[1:]))


def test_reconstruction_total_sel_is_in_the_right_ballpark_vs_control_point_objective():
    """NOT a tight match, and that's expected, not a bug: the 30-waypoint objective's half-segment
    duration weighting can assign a disproportionately long "dwell time" to whichever single
    coarse waypoint happens to land nearest a close encounter, inflating its SEL contribution
    versus the full-resolution reconstruction's much finer time sampling of the same real
    encounter. Checked directly for this case: closest ship-mammal distance agrees well between
    the two (3.36km full-order vs. 3.14km at the nearest control point) while total SEL differs
    by ~3.3dB -- a genuine resolution/discretization effect, not a position or TL error. LOG.md's
    "0.08dB agreement" cross-validation used real (not artificial flat-speed) profiles where this
    effect happened to be small; it is NOT a general <1dB guarantee. Loose bound here on purpose."""
    if not _light_available():
        print("SKIP test_reconstruction_total_sel_is_in_the_right_ballpark_vs_control_point_objective (no local TL model)")
        return
    import csv as csv_mod
    from objective.objective import evaluate_population_multi
    from route_utils.route_utils import downsample_route

    cfg = RunConfig(
        ship=ShipConfig(), optimizer=OptimizerConfig(n_waypoints=30),
        tl_model=TLModelConfig(kind="light", light_model_dir=LIGHT_DIR),
        data=DataPaths(baseline_route_csv="data/baseline_route.csv", mammal_paths_csv="data/mammal_paths.csv"),
        mammal_ids=[12],
    )
    manifest = config_to_dict(cfg)
    optimized_speed = np.full(30, 11.25)

    rows = reconstruct_full_order(manifest, optimized_speed, ROOT)
    sel_full_order = rows[-1]["cumulative_sel_db_12"]

    rows_route = list(csv_mod.DictReader(open(ROOT / "data" / "baseline_route.csv")))
    lat = np.array([float(r["lat"]) for r in rows_route])
    lon = np.array([float(r["lon"]) for r in rows_route])
    speed = np.array([float(r["speed_kn_estimated"]) for r in rows_route])
    _, _, wp_lat, wp_lon, _, _ = downsample_route(lat, lon, speed, 30)
    from reconstruct.reconstruct import load_mammal
    mtrack = load_mammal(ROOT / "data" / "mammal_paths.csv", 12)

    result = evaluate_population_multi(optimized_speed[np.newaxis, :], wp_lat, wp_lon, [mtrack],
                                        cfg.ship.ship_class, cfg.ship.ship_length_ft, 0.0, 0.5, cfg.tl_model)
    sel_control_point = result["SEL_per_mammal"][0, 0]
    assert abs(sel_full_order - sel_control_point) < 5.0, (
        f"full-order ({sel_full_order:.2f}dB) vs control-point ({sel_control_point:.2f}dB) "
        f"SEL differ by more than the expected discretization-effect range")


if __name__ == "__main__":
    test_reconstruction_matches_control_point_count_and_is_monotonic_in_time()
    test_reconstruction_total_sel_is_in_the_right_ballpark_vs_control_point_objective()
    print("reconstruct: all tests passed (or skipped where the TL model is absent)")
