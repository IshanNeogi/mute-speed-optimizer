#!/usr/bin/env python3
"""CLI: generate some or all publication figures for a completed run.

    python -m viz.make_figures --results results/ --figures route,sel,timeline,convergence --formats pdf,eps

Materializes a full-order CSV via reconstruct (if one isn't already in results/) for the figures
that need it (route, timeline) -- no need to run that step separately first.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reconstruct.reconstruct import load_mammal, reconstruct_from_results_dir
from viz import convergence, route_map, sel_comparison, style, timeline

ALL_FIGURES = ("route", "sel", "timeline", "convergence")


def _ensure_full_order_csv(results_dir: Path, seed: int = None):
    existing = sorted(results_dir.glob("full_order_seed*.csv"))
    if existing and seed is None:
        return existing[0]
    if seed is not None:
        candidate = results_dir / f"full_order_seed{seed}.csv"
        if candidate.exists():
            return candidate
    return reconstruct_from_results_dir(results_dir, ROOT, seed=seed)


def _load_full_order_rows(csv_path: Path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            r[k] = float(v)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="a run's output directory (must contain run_manifest.json)")
    parser.add_argument("--figures", default=",".join(ALL_FIGURES))
    parser.add_argument("--formats", default="pdf,eps")
    parser.add_argument("--context", default="poster", choices=["poster", "journal"])
    parser.add_argument("--seed", type=int, default=None, help="which seed to plot (default: the selected one)")
    parser.add_argument("--out", default=None, help="output dir for figures (default: <results>/figures)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    figures = [f.strip() for f in args.figures.split(",") if f.strip()]
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    out_dir = Path(args.out) if args.out else results_dir / "figures"

    style.apply_style(context=args.context)

    manifest = json.load(open(results_dir / "run_manifest.json"))
    mammal_ids = manifest["mammal_ids"]
    baselines = json.load(open(results_dir / "baselines.json"))
    summary = json.load(open(results_dir / "summary.json"))

    seed = args.seed
    if seed is None:
        selected = summary["selected_seed"]
        if selected is None:
            print("No feasible selected seed in summary.json -- nothing to plot.")
            return
        seed = selected["seed"]

    need_full_order = {"route", "timeline"} & set(figures)
    full_order_rows = None
    if need_full_order:
        csv_path = _ensure_full_order_csv(results_dir, seed=seed)
        full_order_rows = _load_full_order_rows(csv_path)
        print(f"Using full-order data: {csv_path}")

    written = []

    if "route" in figures:
        mammal_tracks = {mid: load_mammal(ROOT / manifest["data"]["mammal_paths_csv"], mid) for mid in mammal_ids}
        written += route_map.plot_route_map(full_order_rows, mammal_tracks, out_dir,
                                             context=args.context, formats=formats)

    if "sel" in figures:
        echo_sel = baselines["echo_ais"]["SEL_per_mammal"]
        seed_results = json.load(open(results_dir / "seed_results.json"))
        chosen = next(r for r in seed_results if r["seed"] == seed)
        written += sel_comparison.plot_sel_comparison(mammal_ids, echo_sel, chosen["SEL_per_mammal"],
                                                        out_dir, context=args.context, formats=formats)

    if "timeline" in figures:
        written += timeline.plot_received_level_timeline(full_order_rows, mammal_ids, out_dir,
                                                           context=args.context, formats=formats)
        written += timeline.plot_cumulative_sel_timeline(full_order_rows, mammal_ids, out_dir,
                                                          context=args.context, formats=formats)

    if "convergence" in figures:
        seed_results = json.load(open(results_dir / "seed_results.json"))
        written += convergence.plot_convergence(seed_results, out_dir, context=args.context, formats=formats)

    print(f"\nWrote {len(written)} file(s) to {out_dir}/:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
