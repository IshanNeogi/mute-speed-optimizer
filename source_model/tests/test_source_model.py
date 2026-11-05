"""Plain test_* functions, pytest-compatible, no pytest required to read them.
Run: python -m source_model.tests.test_source_model  (from handover/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source_model.jomopans_model import jomopans_model, jomopans_model_matrix, get_vessel_ref_speed

F = np.geomspace(10.0, 20000.0, 2000)


def test_vessel_ref_speed_known_and_unknown():
    assert get_vessel_ref_speed("Other") == 7.4
    assert get_vessel_ref_speed("Containership") == 18.0
    try:
        get_vessel_ref_speed("NotAClass")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hump_classes_differ_from_other_below_100hz():
    """Containership gets the low-frequency hump; 'Other' doesn't -- confirmed in LOG.md
    (Containership ~173.6dB peak near 40Hz vs. 'Other' smooth rise, no hump)."""
    ls_container, fd = jomopans_model(F, "Containership", 18.0, 697.18)
    ls_other, _ = jomopans_model(F, "Other", 7.4, 697.18)
    low_band_idx = fd < 100
    assert low_band_idx.sum() > 0
    # Containership's low bands should show much more variation (the hump shape) than Other's
    # smooth monotonic-from-191dB-formula low bands.
    assert np.ptp(ls_container[low_band_idx]) > np.ptp(ls_other[low_band_idx])


def test_broadband_increases_with_speed():
    """Sanity: faster ship -> louder, monotonically, matching the paper's 60*log10(V/V_C) term."""
    levels = []
    for v in (8.0, 11.25, 14.5):
        ls, _ = jomopans_model(F, "Other", v, 697.18)
        levels.append(10 * np.log10(np.sum(10 ** (ls / 10.0))))
    assert levels[0] < levels[1] < levels[2]


def test_matrix_v_leq_zero_does_not_zero_other_candidates():
    """Regression test for the fixed bug: in jomopans_model_matrix, a degenerate V<=0 entry
    anywhere in the population must NOT zero every other candidate's output (the original
    ../MUTE bug, found in sandbox/optimizer/LOG.md 2026-09-29 -- .any() with no axis)."""
    pop, n_wp = 3, 2
    V = np.full((pop, n_wp), 10.0)
    V[1, 0] = 0.0   # one degenerate candidate
    ls, fd = jomopans_model_matrix(F, "Other", V, 697.18)

    # candidate 0 and candidate 2 (both all-valid speeds) must be untouched, non-zero
    assert np.all(ls[0] != 0.0)
    assert np.all(ls[2] != 0.0)
    # candidate 1's degenerate waypoint zeroed, its other (valid) waypoint untouched
    assert np.all(ls[1, 0] == 0.0)
    assert np.all(ls[1, 1] != 0.0)

    # cross-check: candidate 0 matches the scalar jomopans_model at the same speed
    ls_scalar, _ = jomopans_model(F, "Other", 10.0, 697.18)
    assert np.allclose(ls[0, 0], ls_scalar, atol=1e-9)


if __name__ == "__main__":
    test_vessel_ref_speed_known_and_unknown()
    test_hump_classes_differ_from_other_below_100hz()
    test_broadband_increases_with_speed()
    test_matrix_v_leq_zero_does_not_zero_other_candidates()
    print("source_model: all tests passed")
