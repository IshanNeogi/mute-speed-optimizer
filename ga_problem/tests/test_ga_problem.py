"""Run: python -m ga_problem.tests.test_ga_problem  (from handover/)"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from ga_problem.ga_problem import MultiMammalSpeedGAProblem, build_seeded_population
from route_utils.route_utils import downsample_route
from tl_model.tl_model import TLModelConfig

DATA = ROOT / "data"
N_WAYPOINTS = 30
SHIP_CLASS, SHIP_LENGTH_FT = "Other", 697.18
MIN_V, MAX_V = 8.0, 14.5

_MUTE_ROOT = Path(os.environ.get("MUTE_ROOT", ROOT.parents[2] / "MUTE"))
LIGHT_DIR = str(_MUTE_ROOT / "v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq")


def _light_available():
    return Path(LIGHT_DIR).exists()


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


def test_problem_without_constraint_has_only_acceleration_constraint():
    wp_lat, wp_lon, speed_cp = load_route()
    track = load_mammal(12)
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)
    problem = MultiMammalSpeedGAProblem(wp_lat, wp_lon, [track], SHIP_CLASS, SHIP_LENGTH_FT,
                                         MIN_V, MAX_V, w_time=1.0, sel_max_weight=0.5, tl_config=tl_cfg)
    assert problem.n_constr == 1


def test_problem_with_constraint_has_one_per_mammal_plus_acceleration():
    wp_lat, wp_lon, speed_cp = load_route()
    tracks = [load_mammal(mid) for mid in (3, 12, 8)]
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)
    problem = MultiMammalSpeedGAProblem(wp_lat, wp_lon, tracks, SHIP_CLASS, SHIP_LENGTH_FT,
                                         MIN_V, MAX_V, w_time=1.0, sel_max_weight=0.5, tl_config=tl_cfg,
                                         baseline_sel_per_mammal=[110.0, 110.0, 110.0])
    assert problem.n_constr == 1 + 3


def test_small_ga_run_reaches_feasibility_with_hard_constraint():
    """The core validated finding (SUMMARY.md): adding the hard per-mammal constraint eliminates
    the bimodal "bad basin" failure mode. Small pop/gen smoke test here (not the full pop=32/
    n_gen=200 recommendation, which is too slow for a routine test suite) -- just confirms the
    constrained GA reliably reaches a feasible (G<=0 everywhere) solution, which is the property
    that mattered."""
    if not _light_available():
        print("SKIP test_small_ga_run_reaches_feasibility_with_hard_constraint (no local TL model)")
        return
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.optimize import minimize

    wp_lat, wp_lon, speed_cp = load_route()
    tracks = [load_mammal(mid) for mid in (3, 12, 8)]
    tl_cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)

    # mid_speed baseline (the actual constraint floor used throughout the validated sandbox work
    # -- "no mammal is ever made worse off than the ship always going medium speed")
    mid_speed = np.full(N_WAYPOINTS, (MIN_V + MAX_V) / 2)
    from objective.objective import evaluate_population_multi
    mid_result = evaluate_population_multi(mid_speed[np.newaxis, :], wp_lat, wp_lon, tracks,
                                            SHIP_CLASS, SHIP_LENGTH_FT, w_time=0.0, sel_max_weight=0.5,
                                            tl_config=tl_cfg)
    baseline_sel = mid_result["SEL_per_mammal"][:, 0]

    problem = MultiMammalSpeedGAProblem(wp_lat, wp_lon, tracks, SHIP_CLASS, SHIP_LENGTH_FT,
                                         MIN_V, MAX_V, w_time=1.0, sel_max_weight=0.5, tl_config=tl_cfg,
                                         baseline_sel_per_mammal=baseline_sel)
    min_speed_seed = np.full(N_WAYPOINTS, MIN_V)
    sampling = build_seeded_population(min_speed_seed, pop_size=16, min_v=MIN_V, max_v=MAX_V, seed=7)
    algorithm = GA(pop_size=16, sampling=sampling, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ("n_gen", 40), seed=7, verbose=False)

    assert res.X is not None, "GA found no feasible solution at all"
    assert np.all(res.G <= 1e-6), f"best solution isn't feasible: G={res.G}"


if __name__ == "__main__":
    test_problem_without_constraint_has_only_acceleration_constraint()
    test_problem_with_constraint_has_one_per_mammal_plus_acceleration()
    test_small_ga_run_reaches_feasibility_with_hard_constraint()
    print("ga_problem: all tests passed (or skipped where the TL model is absent)")
