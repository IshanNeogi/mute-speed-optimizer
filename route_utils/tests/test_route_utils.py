"""Run: python -m route_utils.tests.test_route_utils  (from handover/)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from route_utils.route_utils import (
    cumulative_distance_m, downsample_route, upsample_speed,
    segment_distances_nm, segment_times_hr, waypoint_duration_weights,
    route_distance_nm, constant_speed_for_voyage_time,
)

# A simple synthetic 100-point "route" along a line of longitude -- enough to test the geometry
# without needing the real baseline_route.csv.
N = 100
LAT = np.linspace(48.0, 49.0, N)
LON = np.full(N, -123.0)
SPEED = 10.0 + 2.0 * np.sin(np.linspace(0, 3 * np.pi, N))


def test_cumulative_distance_monotonic_and_starts_at_zero():
    cum = cumulative_distance_m(LAT, LON)
    assert cum[0] == 0.0
    assert np.all(np.diff(cum) > 0)


def test_downsample_upsample_round_trip_accuracy():
    """Round-trip accuracy bound from LOG.md's 30-waypoint validation: 0.46kn RMSE / 2.25kn max
    error on a 9.3-15.4kn baseline range was deemed acceptable there -- use a looser bound here
    since this synthetic profile oscillates faster than the real baseline route did."""
    idx, cum_full, lat_cp, lon_cp, speed_cp, cum_cp = downsample_route(LAT, LON, SPEED, 30)
    assert len(idx) == 30
    speed_full_reconstructed = upsample_speed(cum_full, cum_cp, speed_cp)
    err = speed_full_reconstructed - SPEED
    rmse = np.sqrt(np.mean(err ** 2))
    assert rmse < 1.0, f"round-trip RMSE too high: {rmse:.3f} kn"


def test_downsample_is_identity_when_n_target_exceeds_n_rows():
    idx, cum_full, lat_cp, lon_cp, speed_cp, cum_cp = downsample_route(LAT, LON, SPEED, 1000)
    assert len(idx) == N


def test_segment_kinematics_shapes_and_positivity():
    n_wp = 30
    seg_dist = segment_distances_nm(LAT[:n_wp], LON[:n_wp])
    assert seg_dist.shape == (n_wp - 1,)
    assert np.all(seg_dist > 0)

    x = np.tile(SPEED[:n_wp], (3, 1))  # pop=3, all identical candidates
    seg_time = segment_times_hr(x, seg_dist)
    assert seg_time.shape == (3, n_wp - 1)
    assert np.all(seg_time > 0)

    w = waypoint_duration_weights(seg_time)
    assert w.shape == (3, n_wp)
    # half-segment weighting should sum (per candidate) to the total voyage time
    assert np.allclose(w.sum(axis=1), seg_time.sum(axis=1))


def test_constant_speed_for_voyage_time_round_trips():
    dist_nm = route_distance_nm(LAT, LON)
    assert dist_nm > 0
    time_hr = 10.0
    speed = constant_speed_for_voyage_time(LAT, LON, time_hr)
    assert abs(speed * time_hr - dist_nm) < 1e-9   # speed * time == distance, by construction
    # faster implied speed for a shorter requested time
    assert constant_speed_for_voyage_time(LAT, LON, 5.0) > constant_speed_for_voyage_time(LAT, LON, 10.0)
    try:
        constant_speed_for_voyage_time(LAT, LON, 0.0)
        assert False, "expected ValueError for non-positive time_hr"
    except ValueError:
        pass


if __name__ == "__main__":
    test_cumulative_distance_monotonic_and_starts_at_zero()
    test_downsample_upsample_round_trip_accuracy()
    test_constant_speed_for_voyage_time_round_trips()
    test_downsample_is_identity_when_n_target_exceeds_n_rows()
    test_segment_kinematics_shapes_and_positivity()
    print("route_utils: all tests passed")
