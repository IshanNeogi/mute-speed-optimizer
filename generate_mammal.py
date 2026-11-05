#!/usr/bin/env python3
"""Generate a new mammal track from a starting location/heading/speed and write it into a
mammal_paths.csv-schema file, ready to reference from a run config's data.mammal_paths_csv.

    python generate_mammal.py --lat 48.6 --lon -123.2 --heading-deg 270 --speed-kn 4 \\
        --horizon-min 600 --seed 7 --base-csv data/mammal_paths.csv --out data/my_mammals.csv

Never modifies --base-csv -- always writes a new file (see AGENTS.md: the shipped
data/mammal_paths.csv is pinned, every test/finding in this repo is tied to its exact contents).
Omit --base-csv to write a file containing only the new track; include it to carry its rows
through into --out alongside the new one, so you can mix existing test mammals with your own.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mammal_model.generate import generate_mammal_track
from mammal_model.io import next_mammal_id, write_mammal_csv

KN_TO_KMH = 1.852


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lat", type=float, required=True, help="start latitude, deg WGS84")
    parser.add_argument("--lon", type=float, required=True, help="start longitude, deg WGS84")
    parser.add_argument("--heading-deg", type=float, required=True,
                         help="direction of travel, deg clockwise from north (compass convention)")
    speed_group = parser.add_mutually_exclusive_group(required=True)
    speed_group.add_argument("--speed-kn", type=float, help="speed over ground, knots")
    speed_group.add_argument("--speed-kmh", type=float, help="speed over ground, km/h (the model's native unit)")
    parser.add_argument("--horizon-min", type=int, default=1000,
                         help="projection horizon, minutes (default 1000, matching the shipped test mammals)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (omit for fresh randomness each run)")
    parser.add_argument("--selection", choices=["random", "medoid"], default="random",
                         help="'random' = one realization (default); 'medoid' = closest-to-mean of "
                              "--n-realizations runs")
    parser.add_argument("--n-realizations", type=int, default=1000, help="only used with --selection medoid")
    parser.add_argument("--mammal-id", type=int, default=None,
                         help="defaults to one past the highest id in --base-csv (or 0 if none given)")
    parser.add_argument("--base-csv", default=None, help="existing mammal_paths.csv to carry through (read-only)")
    parser.add_argument("--out", required=True, help="output path -- never the same as --base-csv")
    parser.add_argument("--overwrite-id", action="store_true",
                         help="if --mammal-id already exists in --base-csv, replace it instead of erroring")
    parser.add_argument("--allow-land-start", action="store_true",
                         help="skip the on-land/shallow-water check on the start position (not recommended)")
    args = parser.parse_args()

    if args.base_csv and Path(args.base_csv).resolve() == Path(args.out).resolve():
        parser.error("--out must not be the same file as --base-csv -- this package never edits "
                      "mammal data files in place (see AGENTS.md)")

    speed_kmh = args.speed_kn * KN_TO_KMH if args.speed_kn is not None else args.speed_kmh
    mammal_id = args.mammal_id if args.mammal_id is not None else next_mammal_id(args.base_csv)

    print(f"Generating mammal {mammal_id}: start=({args.lat}, {args.lon}), heading={args.heading_deg} deg, "
          f"speed={speed_kmh:.2f} km/h, horizon={args.horizon_min} min, seed={args.seed}")

    track = generate_mammal_track(
        args.lat, args.lon, args.heading_deg, speed_kmh, args.horizon_min,
        seed=args.seed, selection=args.selection, n_realizations=args.n_realizations,
        check_on_land=not args.allow_land_start,
    )

    out_path = write_mammal_csv(track, mammal_id, args.out, base_csv=args.base_csv, overwrite_id=args.overwrite_id)
    print(f"Wrote {len(track)} points for mammal_id={mammal_id} -> {out_path}")
    print(f"To use it: set data.mammal_paths_csv to '{args.out}' and include {mammal_id} in "
          f"mammal_ids, in your run config.")


if __name__ == "__main__":
    main()
