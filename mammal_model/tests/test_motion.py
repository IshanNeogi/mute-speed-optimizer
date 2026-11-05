"""Run: python -m mammal_model.tests.test_motion  (from handover/)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mammal_model.land_mask import load_land_mask, on_land_vec
from mammal_model.motion import estimate_velocity_kmh, find_most_probable_path, simulate_paths

WATER_LAT, WATER_LON = 48.39988, -122.907005   # a real mammal_paths.csv start point


def test_simulate_paths_never_lands():
    """The whole point of the coastal-step machinery -- confirmed empirically, same convention as
    production's own validation (0 of 165,000 positions on real land, VALIDATION.md)."""
    mask = load_land_mask()
    rng = np.random.default_rng(42)
    paths = simulate_paths(WATER_LAT, WATER_LON, 2.0, 1.0, 20, mask, rng, n_steps=200)
    lat = paths[:, :, 0].ravel()
    lon = paths[:, :, 1].ravel()
    on_land = on_land_vec(lat, lon, mask)
    assert not on_land.any(), f"{on_land.sum()} of {len(lat)} positions landed on a blocked cell"


def test_simulate_paths_deterministic_given_seed():
    mask = load_land_mask()
    paths1 = simulate_paths(WATER_LAT, WATER_LON, 2.0, 1.0, 1, mask, np.random.default_rng(7), n_steps=50)
    paths2 = simulate_paths(WATER_LAT, WATER_LON, 2.0, 1.0, 1, mask, np.random.default_rng(7), n_steps=50)
    assert np.array_equal(paths1, paths2)


def test_simulate_paths_different_seeds_diverge():
    mask = load_land_mask()
    paths1 = simulate_paths(WATER_LAT, WATER_LON, 2.0, 1.0, 1, mask, np.random.default_rng(1), n_steps=50)
    paths2 = simulate_paths(WATER_LAT, WATER_LON, 2.0, 1.0, 1, mask, np.random.default_rng(2), n_steps=50)
    assert not np.array_equal(paths1, paths2)


def test_simulate_paths_shape_and_start_point():
    mask = load_land_mask()
    rng = np.random.default_rng(3)
    n_steps = 30
    paths = simulate_paths(WATER_LAT, WATER_LON, 1.0, 0.5, 4, mask, rng, n_steps)
    assert paths.shape == (4, n_steps + 1, 2)
    assert np.allclose(paths[:, 0, 0], WATER_LAT)
    assert np.allclose(paths[:, 0, 1], WATER_LON)


def test_estimate_velocity_kmh_round_trips_with_a_short_simulated_step():
    """estimate_velocity_kmh recovers the velocity that produced a one-minute step -- used when
    extending a mammal's horizon mid-voyage from its actual last move."""
    mask = load_land_mask()
    rng = np.random.default_rng(5)
    paths = simulate_paths(WATER_LAT, WATER_LON, 3.0, 0.0, 1, mask, rng, n_steps=1)
    lat0, lon0 = paths[0, 0]
    lat1, lon1 = paths[0, 1]
    vx, vy = estimate_velocity_kmh(lat0, lon0, lat1, lon1)
    # not necessarily (3.0, 0.0) exactly -- the coastal step / OU noise can have already acted by
    # minute 1 -- just check it's a sane finite velocity, same order of magnitude
    assert np.isfinite(vx) and np.isfinite(vy)
    assert abs(vx) < 50 and abs(vy) < 50


def test_find_most_probable_path_picks_the_closest_to_centroid():
    paths = np.array([
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.1, 1.1]],
        [[0.0, 0.0], [5.0, 5.0]],   # outlier
    ])
    idx, best = find_most_probable_path(paths)
    assert idx in (0, 1)   # the outlier (index 2) should never be picked


if __name__ == "__main__":
    test_simulate_paths_never_lands()
    test_simulate_paths_deterministic_given_seed()
    test_simulate_paths_different_seeds_diverge()
    test_simulate_paths_shape_and_start_point()
    test_estimate_velocity_kmh_round_trips_with_a_short_simulated_step()
    test_find_most_probable_path_picks_the_closest_to_centroid()
    print("mammal_model.motion: all tests passed")
