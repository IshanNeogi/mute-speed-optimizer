"""Fixed-speed reference heuristics (min/mid/max constant speed + the real recorded voyage) and
the seed-selection rule used to pick one result out of a multi-seed GA run.

Ported from sandbox/optimizer/final_optimizer_docs/common.py's compute_heuristic_baselines /
select_best_seed -- confirmed FINAL/validated there (2026-10-08).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objective.objective import evaluate_population_multi
from tl_model.tl_model import TLModelConfig


def compute_heuristic_baselines(wp_lat, wp_lon, echo_speed, tracks, ship_class, ship_length_ft,
                                 sel_max_weight, tl_config: TLModelConfig, min_v=8.0, max_v=14.5,
                                 mid_v=None, received_level_mode="peak_band"):
    """Returns dict: profile_name -> {"speed", "SEL_per_mammal", "NL_max_per_mammal", "T_hr"}
    for min_speed / mid_speed / max_speed / echo_ais, evaluated against the given mammal tracks.

    `mid_speed` is also the constraint floor when `optimizer.use_hard_mammal_constraint` is on
    (see run.py) -- "no mammal is ever made worse off than the ship going this speed the whole
    way." Defaults to the arithmetic midpoint of (min_v, max_v) (the validated setting -- see
    SCIENTIFIC_FINDINGS.md), but any value in [min_v, max_v] can be passed explicitly via `mid_v`
    to use a different constraint reference speed.

    `received_level_mode` MUST match whatever the GA run itself uses ("peak_band" or "broadband",
    see objective/objective.py) -- the constraint compares SEL values directly, so baseline and
    candidate must be computed under the same convention or the comparison is meaningless."""
    n_wp = len(wp_lat)
    if mid_v is None:
        mid_v = (min_v + max_v) / 2.0
    elif not (min_v <= mid_v <= max_v):
        raise ValueError(f"mid_v={mid_v} must be within [min_v={min_v}, max_v={max_v}]")
    profiles = {
        "min_speed": np.full(n_wp, min_v),
        "mid_speed": np.full(n_wp, mid_v),
        "max_speed": np.full(n_wp, max_v),
        "echo_ais": np.asarray(echo_speed).copy(),
    }
    out = {}
    for name, profile in profiles.items():
        res = evaluate_population_multi(profile[np.newaxis, :], wp_lat, wp_lon, tracks, ship_class,
                                         ship_length_ft, 0.0, sel_max_weight, tl_config,
                                         received_level_mode=received_level_mode)
        out[name] = {
            "speed": profile.tolist(),
            "SEL_per_mammal": res["SEL_per_mammal"][:, 0].tolist(),
            "NL_max_per_mammal": res["NL_per_mammal"][:, 0, :].max(axis=1).tolist(),
            "T_hr": float(res["T_hr"][0]),
        }
    return out


def select_best_seed(seed_results):
    """Among feasible seeds, prefer those that also beat echo_ais for every mammal, and among
    those pick the lowest voyage time. If none beat echo_ais, fall back to lowest voyage time
    among all feasible seeds (still guaranteed to beat mid_speed by construction, if the hard
    constraint was used). If no seed is feasible at all, returns (None, "NONE_FEASIBLE") -- flag
    this loudly, never silently adopt an infeasible result."""
    feasible = [r for r in seed_results if r.get("feasible")]
    if not feasible:
        return None, "NONE_FEASIBLE"
    echo_qualified = [r for r in feasible if r.get("beats_echo_ais")]
    if echo_qualified:
        return min(echo_qualified, key=lambda r: r["T_hr"]), "beats_echo_ais"
    return min(feasible, key=lambda r: r["T_hr"]), "fastest_feasible_fallback"
