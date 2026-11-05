"""Run: python -m mammal_model.tests.test_generate  (from handover/)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mammal_model.generate import generate_mammal_track, heading_speed_to_vx_vy

WATER_LAT, WATER_LON = 48.39988, -122.907005
LAND_LAT, LAND_LON = 48.60, -123.15   # confirmed blocked (land/shallow), this session


def test_heading_speed_conversion_matches_compass_convention():
    """0deg (north) -> all northward (vy>0, vx~0); 90deg (east) -> all eastward (vx>0, vy~0) --
    same convention production's predict.py uses (clockwise from north)."""
    vx, vy = heading_speed_to_vx_vy(0.0, 10.0)
    assert abs(vx) < 1e-9 and vy == 10.0
    vx, vy = heading_speed_to_vx_vy(90.0, 10.0)
    assert vx == 10.0 and abs(vy) < 1e-9
    vx, vy = heading_speed_to_vx_vy(180.0, 10.0)
    assert abs(vx) < 1e-9 and abs(vy - (-10.0)) < 1e-9


def test_heading_speed_rejects_negative_speed():
    try:
        heading_speed_to_vx_vy(0.0, -1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_generate_mammal_track_shape_and_start():
    track = generate_mammal_track(WATER_LAT, WATER_LON, heading_deg=270.0, speed_kmh=5.0,
                                   horizon_min=30, seed=7)
    assert len(track) == 31   # horizon_min + 1
    assert track[0]["minute"] == 0
    assert abs(track[0]["lat"] - WATER_LAT) < 1e-5
    assert abs(track[0]["lon"] - WATER_LON) < 1e-5
    assert track[-1]["minute"] == 30


def test_generate_mammal_track_deterministic_given_seed():
    t1 = generate_mammal_track(WATER_LAT, WATER_LON, 270.0, 5.0, 20, seed=11)
    t2 = generate_mammal_track(WATER_LAT, WATER_LON, 270.0, 5.0, 20, seed=11)
    assert t1 == t2


def test_generate_mammal_track_rejects_land_start_by_default():
    try:
        generate_mammal_track(LAND_LAT, LAND_LON, 0.0, 5.0, 10, seed=1)
        assert False, "expected ValueError for a land/shallow-water start"
    except ValueError as e:
        assert "land" in str(e).lower()


def test_generate_mammal_track_allows_land_start_when_bypassed():
    # should not raise, even though the start is on the blocked mask
    track = generate_mammal_track(LAND_LAT, LAND_LON, 0.0, 5.0, 5, seed=1, check_on_land=False)
    assert len(track) == 6


def test_generate_mammal_track_medoid_selection_runs():
    track = generate_mammal_track(WATER_LAT, WATER_LON, 45.0, 3.0, 15, seed=3,
                                   selection="medoid", n_realizations=10)
    assert len(track) == 16


if __name__ == "__main__":
    test_heading_speed_conversion_matches_compass_convention()
    test_heading_speed_rejects_negative_speed()
    test_generate_mammal_track_shape_and_start()
    test_generate_mammal_track_deterministic_given_seed()
    test_generate_mammal_track_rejects_land_start_by_default()
    test_generate_mammal_track_allows_land_start_when_bypassed()
    test_generate_mammal_track_medoid_selection_runs()
    print("mammal_model.generate: all tests passed")
