"""Write a generated track into the project's mammal_paths.csv schema (mammal_id, minute, lat,
lon), optionally merged with an existing file -- WITHOUT ever modifying that existing file (see
AGENTS.md: data/mammal_paths.csv is pinned, every test/finding is tied to its exact contents).
Always writes to a new path; `base_csv` is read-only input, never touched.
"""
import csv
from pathlib import Path
from typing import Optional


def next_mammal_id(base_csv: Optional[Path]) -> int:
    """One past the highest mammal_id in base_csv, or 0 if base_csv is None/empty."""
    if base_csv is None or not Path(base_csv).exists():
        return 0
    rows = list(csv.DictReader(open(base_csv)))
    if not rows:
        return 0
    return max(int(r["mammal_id"]) for r in rows) + 1


def write_mammal_csv(track: list, mammal_id: int, out_csv: Path, base_csv: Optional[Path] = None,
                      overwrite_id: bool = False):
    """Writes out_csv = (base_csv's rows, if given) + track's rows under mammal_id.
    track: list of {"minute", "lat", "lon"} dicts, e.g. from generate.generate_mammal_track.
    base_csv is read, never written to. Raises if mammal_id already exists in base_csv unless
    overwrite_id=True (then that mammal_id's rows from base_csv are dropped before appending
    the new track under the same id)."""
    out_csv = Path(out_csv)
    existing_rows = []
    if base_csv is not None and Path(base_csv).exists():
        existing_rows = list(csv.DictReader(open(base_csv)))
        existing_ids = {int(r["mammal_id"]) for r in existing_rows}
        if mammal_id in existing_ids:
            if not overwrite_id:
                raise ValueError(
                    f"mammal_id {mammal_id} already exists in {base_csv} -- pick a different id "
                    f"(see mammal_model.io.next_mammal_id), or pass overwrite_id=True to replace it.")
            existing_rows = [r for r in existing_rows if int(r["mammal_id"]) != mammal_id]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mammal_id", "minute", "lat", "lon"])
        w.writeheader()
        for r in existing_rows:
            w.writerow({"mammal_id": r["mammal_id"], "minute": r["minute"], "lat": r["lat"], "lon": r["lon"]})
        for p in track:
            w.writerow({"mammal_id": mammal_id, "minute": p["minute"], "lat": p["lat"], "lon": p["lon"]})
    return out_csv
