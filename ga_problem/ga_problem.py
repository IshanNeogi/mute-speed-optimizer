"""pymoo Problem wrapping objective.evaluate_population_multi. Enforces:
  1. an acceleration constraint (same as production's SpeedOptimizationProblem)
  2. an OPTIONAL per-mammal non-regression constraint: when `baseline_sel_per_mammal` is given,
     any feasible solution must have SEL_mammal_i(x) <= baseline_sel_per_mammal[i] for every
     mammal i -- this is the validated core contribution (sandbox/optimizer/SUMMARY.md /
     LOG.md): the unconstrained weighted-sum formulation is bimodal/seed-unreliable (52-100%
     "bad basin" across 115 tested seed/config combos); adding this constraint eliminated that
     failure mode entirely (0/35 bad-basin across every constrained trial tested). It's a
     constructor argument here, not hardcoded on, so you can reproduce BOTH behaviors for
     comparison -- pass `baseline_sel_per_mammal=None` to get the old unconstrained formulation.
"""
import sys
from pathlib import Path

import numpy as np
from pymoo.core.problem import Problem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from objective.objective import evaluate_population_multi
from route_utils.route_utils import segment_distances_nm
from tl_model.tl_model import TLModelConfig


def build_seeded_population(seed_speed: np.ndarray, pop_size: int, min_v: float, max_v: float,
                             seed: int) -> np.ndarray:
    """row0 = seed_speed's shape, row1 = flat mean speed, rest = seed_speed + random jitter.
    `seed_speed` is whatever profile you want the search to start near -- the baseline (echo_ais)
    profile, or (per the min-speed-seeded-GA finding, LOG.md 2026-10-01) np.full(n_wp, min_v):
    two different seed SHAPES at the same hyperparameters were found to land in optima 4+dB apart
    for the multi-mammal case -- min-speed seeding is the validated standard going forward, not
    baseline-shaped, but this function supports either."""
    n_wp = len(seed_speed)
    shaped = np.clip(seed_speed, min_v, max_v)
    flat = np.full(n_wp, float(np.clip(np.mean(shaped), min_v, max_v)))
    pop = np.tile(shaped, (pop_size, 1))
    pop[0] = flat
    if pop_size > 2:
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.5, 0.5, size=(pop_size - 2, n_wp))
        pop[2:] = np.clip(shaped + jitter, min_v, max_v)
    return pop


class MultiMammalSpeedGAProblem(Problem):
    def __init__(self, wp_lat, wp_lon, mammal_tracks, ship_class, ship_length_ft,
                 min_v, max_v, w_time, sel_max_weight, tl_config: TLModelConfig,
                 acc_limit=9.0, baseline_sel_per_mammal=None, received_level_mode="peak_band"):
        n_wp = len(wp_lat)
        n_mammal_constr = len(mammal_tracks) if baseline_sel_per_mammal is not None else 0
        super().__init__(n_var=n_wp, n_obj=1, n_constr=1 + n_mammal_constr, xl=min_v, xu=max_v)
        self.wp_lat, self.wp_lon = wp_lat, wp_lon
        self.mammal_tracks = mammal_tracks
        self.ship_class, self.ship_length_ft = ship_class, ship_length_ft
        self.w_time, self.sel_max_weight = w_time, sel_max_weight
        self.tl_config = tl_config
        self.acc_limit = acc_limit
        self.received_level_mode = received_level_mode
        # baseline_sel_per_mammal MUST already be computed under the same received_level_mode --
        # comparing a broadband-mode candidate's SEL against a peak_band-mode baseline would be
        # apples-to-oranges and silently break the constraint's meaning. Callers (run.py/baselines)
        # are responsible for this; not re-checked here since there's no way to verify after the
        # fact from a plain number.
        self.baseline_sel_per_mammal = (np.asarray(baseline_sel_per_mammal) if baseline_sel_per_mammal is not None
                                         else None)
        self.seg_dist = segment_distances_nm(wp_lat, wp_lon)
        self.last_eval = None

    def _evaluate(self, x, out, *args, **kwargs):
        result = evaluate_population_multi(x, self.wp_lat, self.wp_lon, self.mammal_tracks,
                                            self.ship_class, self.ship_length_ft, self.w_time,
                                            self.sel_max_weight, self.tl_config,
                                            received_level_mode=self.received_level_mode)
        self.last_eval = result
        out["F"] = result["cost"]

        if x.shape[1] >= 2:
            avg_speed = np.clip((x[:, :-1] + x[:, 1:]) / 2.0, 1e-3, None)
            seg_time_hr = self.seg_dist[np.newaxis, :] / avg_speed
            acc = np.abs(np.diff(x, axis=1)) / np.clip(seg_time_hr, 1e-6, None)
            acc_constr = np.max(acc, axis=1) - self.acc_limit
        else:
            acc_constr = np.full(x.shape[0], -self.acc_limit)

        if self.baseline_sel_per_mammal is not None:
            # SEL_per_mammal: (M, pop) -> per-mammal constraint (pop, M), <=0 means that mammal improved
            mammal_constr = result["SEL_per_mammal"].T - self.baseline_sel_per_mammal[np.newaxis, :]
            out["G"] = np.concatenate([acc_constr.reshape(-1, 1), mammal_constr], axis=1)
        else:
            out["G"] = acc_constr.reshape(-1, 1)
