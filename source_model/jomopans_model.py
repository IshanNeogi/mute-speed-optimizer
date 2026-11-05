"""JOMOPANS-ECHO ship source-level model.

Ported from sandbox/optimizer/source_model/jomopans_model.py, itself a verbatim port of
../MUTE v2/src/noise_source_model/models/jomopans_echo/model.py — pure math, no ROS2 dependency.

One deviation from ../MUTE, deliberate and flagged: the original's `-inf` guard
(`if np.isneginf(...).any(): array[:] = 0.0`) checks `.any()` over the WHOLE array with no axis.
In `jomopans_model_matrix`, where the array is (pop, n_waypoints, n_bands), this means a single
degenerate V<=0 candidate anywhere in a population zeroes the entire generation's source-level
output, not just the offending candidate — found during sandbox validation
(sandbox/optimizer/LOG.md, 2026-09-29), reported but not fixed there (production still has it).
Fixed HERE (this is a fresh codebase, not a change to ../MUTE's scientific core) by masking only
the specific (pop, waypoint) slices that actually went -inf, leaving every other candidate's
output untouched. See tests/test_source_model.py for the regression test that would have caught
the original bug.
"""
import numpy as np

F_DECIDECADE_FULL = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                               400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
                               6300, 8000, 10000, 12500, 16000, 20000])

VESSEL_REF_SPEED_KN = {
    "Fishing": 6.4, "Tug": 3.7, "Naval": 11.1, "Recreational": 10.6, "Government/Research": 8.0,
    "Cruise": 17.1, "Passenger": 9.7, "Bulker": 13.9, "Containership": 18.0,
    "Vehicle Carrier": 15.8, "Tanker": 12.4, "Other": 7.4, "Dredger": 9.5,
}
LOW_FREQ_HUMP_CLASSES = ("Containership", "Vehicle Carrier", "Tanker", "Bulker")
REFERENCE_LENGTH_FT = 300


def get_vessel_ref_speed(vessel_class: str) -> float:
    if vessel_class not in VESSEL_REF_SPEED_KN:
        raise ValueError(f'The vessel class "{vessel_class}" is not recognized.')
    return VESSEL_REF_SPEED_KN[vessel_class]


def _compute_baseline(f, C, V_C):
    """L_s_f_0: below-100Hz branch only fires for the 4 "hump" classes (JOMOPANS-ECHO spec) —
    every other class (including "Other") gets the >=100Hz formula applied across the whole
    frequency range, confirmed by direct comparison in sandbox/optimizer/LOG.md."""
    f_ref, V_ref = 1.0, 1.0
    f_hat = f / f_ref
    L_s_f_0 = np.zeros_like(f, dtype=float)

    idx_below_100 = f_hat < 100
    idx_above_100 = f_hat >= 100
    f_hat_below_100 = f_hat[idx_below_100]
    f_hat_above_100 = f_hat[idx_above_100]

    if f_hat_below_100.size > 0 and C in LOW_FREQ_HUMP_CLASSES:
        K_LF = 208.0
        f1_LF = 600 * (V_ref / V_C)
        D_LF = 0.8 if C in ("Containership", "Bulker") else 1.0
        term1 = K_LF - 40 * np.log10(f1_LF)
        term2 = 10 * np.log10(f_hat_below_100)
        term3 = -10 * np.log10((1 - (f_hat_below_100 / f1_LF) ** 2) ** 2 + D_LF ** 2)
        L_s_f_0[idx_below_100] = term1 + term2 + term3

    if f_hat_above_100.size > 0:
        K = 191.0
        f1 = 480 * (V_ref / V_C)
        D = 4 if C == "Cruise" else 3
        term1 = K - 20 * np.log10(f1)
        if C in LOW_FREQ_HUMP_CLASSES:
            term2 = -10 * np.log10((1 - (f_hat_above_100 / f1)) ** 2 + D ** 2)
            L_s_f_0[idx_above_100] = term1 + term2
        else:
            term2 = -10 * np.log10((1 - (f_hat / f1)) ** 2 + D ** 2)
            L_s_f_0 = term1 + term2

    return L_s_f_0


def _convert_to_decidecade(f, L_s_f_JE):
    f_decidecade = F_DECIDECADE_FULL[(F_DECIDECADE_FULL >= f.min()) & (F_DECIDECADE_FULL <= f.max())]
    out = np.zeros_like(f_decidecade)
    for j, fd in enumerate(f_decidecade):
        idx = np.argmin(np.abs(f - fd))
        out[j] = L_s_f_JE[idx] + 10 * np.log10(0.231 * fd)
    return out, f_decidecade


def jomopans_model(f: np.ndarray, C: str, V: float, l: float):
    """f: frequency array (Hz). C: ship class. V: speed (kn). l: ship length (ft).
    Returns (L_s_JE_decidecade, f_decidecade): source level per decidecade band, dB re 1uPa @ 1m."""
    V_C = get_vessel_ref_speed(C)
    l_0 = REFERENCE_LENGTH_FT
    L_s_f_JE = _compute_baseline(f, C, V_C) + 60 * np.log10(V / V_C) + 20 * np.log10(l / l_0)
    L_s_JE_decidecade, f_decidecade = _convert_to_decidecade(f, L_s_f_JE)
    if np.isneginf(L_s_JE_decidecade).any():
        L_s_JE_decidecade[np.isneginf(L_s_JE_decidecade)] = 0.0
    return L_s_JE_decidecade, f_decidecade


def jomopans_model_matrix(f: np.ndarray, C: str, V: np.ndarray, l: float):
    """f: frequency array (Hz). C: ship class. V: (pop, n_waypoints) speed matrix (kn).
    l: ship length (ft). Returns (L_s_JE_decidecade, f_decidecade), shape (pop, n_waypoints, n_bands).

    FIX vs. ../MUTE (see module docstring): zeroing on a -inf hit is masked per-(pop,waypoint)
    slice, not applied to the whole array — a degenerate V<=0 candidate no longer silently zeroes
    every other candidate's source level in the same generation.
    """
    V_C = get_vessel_ref_speed(C)
    l_0 = REFERENCE_LENGTH_FT

    L_s_f_0 = _compute_baseline(f, C, V_C).reshape(1, 1, -1)
    term_velocity = (60 * np.log10(V / V_C))[:, :, np.newaxis]
    term_length = np.array(20 * np.log10(l / l_0)).reshape(1, 1, 1)
    L_s_f_JE = L_s_f_0 + term_velocity + term_length  # (pop, n_wp, len(f))

    f_decidecade = F_DECIDECADE_FULL[(F_DECIDECADE_FULL >= f.min()) & (F_DECIDECADE_FULL <= f.max())]
    pop, n_wp = V.shape
    L_s_JE_decidecade = np.zeros((pop, n_wp, len(f_decidecade)))
    for j, fd in enumerate(f_decidecade):
        idx = np.argmin(np.abs(f - fd))
        L_s_JE_decidecade[:, :, j] = L_s_f_JE[:, :, idx] + 10 * np.log10(0.231 * fd)

    bad = np.isneginf(L_s_JE_decidecade)
    if bad.any():
        # mask per-(pop, waypoint) slice -- only the degenerate candidate's bands go to 0,
        # not every candidate's (the original ../MUTE bug this fixes)
        bad_slice = bad.any(axis=-1)
        L_s_JE_decidecade[bad_slice] = 0.0

    return L_s_JE_decidecade, f_decidecade
