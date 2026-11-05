#!/usr/bin/env python3
"""Entrypoint: load a config, run the multi-seed constrained (or unconstrained, per config) GA for
one mammal set, write results.

    python run.py --config config/default_config.yaml

Every tunable lives in the YAML, not here -- see config/default_config.yaml for what each one
means and which findings justify its default.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from baselines.baselines import compute_heuristic_baselines, select_best_seed
from config import RunConfig, config_to_dict, load_config
from ga_problem.ga_problem import MultiMammalSpeedGAProblem, build_seeded_population
from objective.objective import evaluate_population_multi
from operators.operators import AdaptiveCrossover, AdaptiveMutation
from route_utils.route_utils import (
    constant_speed_for_voyage_time, downsample_route, segment_distances_nm, segment_times_hr,
)


def load_route(csv_path: Path, n_waypoints: int):
    rows = list(csv.DictReader(open(csv_path)))
    lat = np.array([float(r["lat"]) for r in rows])
    lon = np.array([float(r["lon"]) for r in rows])
    speed = np.array([float(r["speed_kn_estimated"]) for r in rows])
    _, _, wp_lat, wp_lon, speed_cp, _ = downsample_route(lat, lon, speed, n_waypoints)
    return wp_lat, wp_lon, speed_cp


def load_mammal(csv_path: Path, mammal_id: int):
    rows = list(csv.DictReader(open(csv_path)))
    track = sorted((int(r["minute"]), float(r["lat"]), float(r["lon"]))
                   for r in rows if int(r["mammal_id"]) == mammal_id)
    return np.array([t[0] for t in track], float), np.array([t[1] for t in track]), np.array([t[2] for t in track])


def run_one_seed(cfg: RunConfig, wp_lat, wp_lon, tracks, seed: int, baseline_sel_per_mammal):
    ship, opt = cfg.ship, cfg.optimizer
    problem = MultiMammalSpeedGAProblem(
        wp_lat, wp_lon, tracks, ship.ship_class, ship.ship_length_ft,
        ship.min_speed_kn, ship.max_speed_kn, opt.w_time, opt.sel_max_weight, cfg.tl_model,
        acc_limit=ship.acceleration_limit_kn_per_hr,
        baseline_sel_per_mammal=baseline_sel_per_mammal if opt.use_hard_mammal_constraint else None,
        received_level_mode=opt.received_level_mode,
    )

    if opt.seed_population_shape == "min_speed":
        seed_shape = np.full(opt.n_waypoints, ship.min_speed_kn)
    else:
        _, _, echo_speed = load_route(HERE / cfg.data.baseline_route_csv, opt.n_waypoints)
        seed_shape = echo_speed

    sampling = build_seeded_population(seed_shape, opt.pop_size, ship.min_speed_kn, ship.max_speed_kn, seed)
    algorithm = GA(
        pop_size=opt.pop_size, eliminate_duplicates=True, sampling=sampling,
        mutation=AdaptiveMutation(opt.mutation_initial_rate, opt.mutation_final_rate, opt.n_gen),
        crossover=AdaptiveCrossover(opt.crossover_initial_rate, opt.crossover_final_rate, opt.n_gen),
    )

    t0 = time.time()
    res = minimize(problem, algorithm, termination=("n_gen", opt.n_gen), seed=seed, verbose=False,
                    save_history=True)
    elapsed_s = time.time() - t0

    # per-generation best feasible cost -- for a convergence plot (viz/convergence.py). Only kept
    # for generations where SOME feasible individual existed (res.history entries with no
    # feasible opt have opt=None); gaps are expected early in a run under a hard constraint.
    cost_history = [float(e.opt.get("F")[0, 0]) for e in res.history if e.opt is not None]

    if res.X is None:
        return {"seed": seed, "feasible": False, "elapsed_s": elapsed_s, "cost_history": cost_history}

    out = evaluate_population_multi(res.X[np.newaxis, :], wp_lat, wp_lon, tracks, ship.ship_class,
                                     ship.ship_length_ft, opt.w_time, opt.sel_max_weight, cfg.tl_model,
                                     received_level_mode=opt.received_level_mode)
    sel_per_mammal = out["SEL_per_mammal"][:, 0]
    feasible = bool(np.all(sel_per_mammal <= np.asarray(baseline_sel_per_mammal) + 1e-6))
    return {
        "seed": seed, "optimized_speed": res.X.tolist(), "feasible": feasible, "elapsed_s": elapsed_s,
        "cost": float(out["cost"][0]), "SEL_max": float(out["SEL_max"][0]),
        "SEL_mean": float(out["SEL_mean"][0]), "T_hr": float(out["T_hr"][0]),
        "SEL_per_mammal": sel_per_mammal.tolist(), "cost_history": cost_history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config" / "default_config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    wp_lat, wp_lon, echo_speed = load_route(HERE / cfg.data.baseline_route_csv, cfg.optimizer.n_waypoints)
    tracks = [load_mammal(HERE / cfg.data.mammal_paths_csv, mid) for mid in cfg.mammal_ids]

    print(f"Running {len(cfg.mammal_ids)} mammal(s): {cfg.mammal_ids}, "
          f"{cfg.optimizer.n_seeds} seed(s), constraint={'ON' if cfg.optimizer.use_hard_mammal_constraint else 'OFF'}")

    # constraint_voyage_time_hr -> an equivalent constant speed (route_distance_nm / time_hr),
    # computed here rather than inside baselines.compute_heuristic_baselines so that function can
    # stay a plain speed-in/speed-out primitive. Default (None) = echo_ais's own voyage time at
    # THIS run's control-point resolution (same segment kinematics evaluate_population_multi uses
    # for the "echo_ais" baseline, so this matches baselines["echo_ais"]["T_hr"] exactly below).
    voyage_time_hr = cfg.optimizer.constraint_voyage_time_hr
    if voyage_time_hr is None:
        echo_seg_dist = segment_distances_nm(wp_lat, wp_lon)
        echo_seg_time = segment_times_hr(echo_speed[np.newaxis, :], echo_seg_dist)
        voyage_time_hr = float(echo_seg_time.sum())
    constraint_speed_kn = constant_speed_for_voyage_time(wp_lat, wp_lon, voyage_time_hr)
    print(f"Constraint ('mid_speed') voyage time: {voyage_time_hr:.2f} hr -> {constraint_speed_kn:.2f} kn")

    baselines = compute_heuristic_baselines(wp_lat, wp_lon, echo_speed, tracks, cfg.ship.ship_class,
                                             cfg.ship.ship_length_ft, cfg.optimizer.sel_max_weight,
                                             cfg.tl_model, cfg.ship.min_speed_kn, cfg.ship.max_speed_kn,
                                             mid_v=constraint_speed_kn,
                                             received_level_mode=cfg.optimizer.received_level_mode)
    mid_speed_sel = baselines["mid_speed"]["SEL_per_mammal"]
    echo_ais_sel = baselines["echo_ais"]["SEL_per_mammal"]

    seed_results = []
    for seed in range(1, cfg.optimizer.n_seeds + 1):
        print(f"  seed {seed}/{cfg.optimizer.n_seeds} ...", end=" ", flush=True)
        result = run_one_seed(cfg, wp_lat, wp_lon, tracks, seed, mid_speed_sel)
        result["beats_echo_ais"] = bool(result.get("feasible") and
                                         np.all(np.asarray(result["SEL_per_mammal"]) <= np.asarray(echo_ais_sel) + 1e-6))
        seed_results.append(result)
        print(f"feasible={result.get('feasible')} T_hr={result.get('T_hr', float('nan')):.2f} "
              f"({result['elapsed_s']:.1f}s)")

    best, branch = select_best_seed(seed_results)
    out_dir = HERE / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "seed_results.json", "w") as f:
        json.dump(seed_results, f, indent=2)
    with open(out_dir / "baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"selected_seed": best, "selection_branch": branch, "mammal_ids": cfg.mammal_ids}, f, indent=2)
    # Everything needed to deterministically reconstruct the full-order result later, without
    # rerunning the GA -- see reconstruct/reconstruct.py. The GA search itself is the only
    # non-reproducible step; its OUTPUT (seed_results.json's optimized_speed) plus this manifest
    # (exact config) plus the fixed data/ CSVs fully determine the rest via pure functions.
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    print(f"\nSelection rule: {branch}")
    if best is None:
        print("NO FEASIBLE SEED FOUND -- nothing to report. Check config (acc_limit, speed bounds, w_time).")
    else:
        print(f"Selected seed {best['seed']}: T_hr={best['T_hr']:.2f} (echo_ais={echo_ais_sel and baselines['echo_ais']['T_hr']:.2f})")
        for mid, sel_opt, sel_echo in zip(cfg.mammal_ids, best["SEL_per_mammal"], echo_ais_sel):
            print(f"  mammal {mid}: {sel_echo:.2f}dB (echo) -> {sel_opt:.2f}dB (optimized), "
                  f"{sel_echo - sel_opt:+.2f}dB")
    print(f"\nWrote results to {out_dir}/")


if __name__ == "__main__":
    main()
