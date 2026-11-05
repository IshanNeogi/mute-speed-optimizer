"""Cross-module integration test -- not owned by any single module, since it exercises all of
them together through run.py's own functions. Run: python -m tests.test_end_to_end (from handover/)
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import RunConfig, ShipConfig, OptimizerConfig, DataPaths
from tl_model.tl_model import TLModelConfig
from run import load_route, load_mammal, run_one_seed
from baselines.baselines import compute_heuristic_baselines, select_best_seed

_MUTE_ROOT = Path(os.environ.get("MUTE_ROOT", ROOT.parents[2] / "MUTE"))
LIGHT_DIR = str(_MUTE_ROOT / "v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq")


def _light_available():
    return Path(LIGHT_DIR).exists()


def test_full_pipeline_tiny_run_is_feasible_and_beats_mid_speed():
    if not _light_available():
        print("SKIP test_full_pipeline_tiny_run_is_feasible_and_beats_mid_speed (no local TL model)")
        return

    cfg = RunConfig(
        ship=ShipConfig(),
        optimizer=OptimizerConfig(pop_size=16, n_gen=30, n_seeds=1, use_hard_mammal_constraint=True),
        tl_model=TLModelConfig(kind="light", light_model_dir=LIGHT_DIR),
        data=DataPaths(baseline_route_csv="data/baseline_route.csv", mammal_paths_csv="data/mammal_paths.csv"),
        mammal_ids=[3, 12, 8],
    )
    wp_lat, wp_lon, echo_speed = load_route(ROOT / cfg.data.baseline_route_csv, cfg.optimizer.n_waypoints)
    tracks = [load_mammal(ROOT / cfg.data.mammal_paths_csv, mid) for mid in cfg.mammal_ids]

    baselines = compute_heuristic_baselines(wp_lat, wp_lon, echo_speed, tracks, cfg.ship.ship_class,
                                             cfg.ship.ship_length_ft, cfg.optimizer.sel_max_weight, cfg.tl_model)
    mid_speed_sel = baselines["mid_speed"]["SEL_per_mammal"]

    result = run_one_seed(cfg, wp_lat, wp_lon, tracks, seed=1, baseline_sel_per_mammal=mid_speed_sel)

    assert result["feasible"], "constrained GA should find a feasible solution"
    sel_opt = np.asarray(result["SEL_per_mammal"])
    assert np.all(sel_opt <= np.asarray(mid_speed_sel) + 1e-6), (
        "every mammal's optimized SEL must not exceed its mid_speed baseline -- the whole point "
        "of the hard constraint")


def test_unconstrained_mode_still_runs():
    """use_hard_mammal_constraint=False must still produce a runnable problem (the old formulation,
    kept for comparison) -- it is not required to be feasible/non-regressing, just to execute."""
    if not _light_available():
        print("SKIP test_unconstrained_mode_still_runs (no local TL model)")
        return
    cfg = RunConfig(
        ship=ShipConfig(),
        optimizer=OptimizerConfig(pop_size=8, n_gen=10, n_seeds=1, use_hard_mammal_constraint=False),
        tl_model=TLModelConfig(kind="light", light_model_dir=LIGHT_DIR),
        data=DataPaths(baseline_route_csv="data/baseline_route.csv", mammal_paths_csv="data/mammal_paths.csv"),
        mammal_ids=[12],
    )
    wp_lat, wp_lon, echo_speed = load_route(ROOT / cfg.data.baseline_route_csv, cfg.optimizer.n_waypoints)
    tracks = [load_mammal(ROOT / cfg.data.mammal_paths_csv, mid) for mid in cfg.mammal_ids]
    result = run_one_seed(cfg, wp_lat, wp_lon, tracks, seed=1, baseline_sel_per_mammal=[200.0])
    assert "cost" in result or result.get("feasible") is False


if __name__ == "__main__":
    test_full_pipeline_tiny_run_is_feasible_and_beats_mid_speed()
    test_unconstrained_mode_still_runs()
    print("end_to_end: all tests passed (or skipped where the TL model is absent)")
