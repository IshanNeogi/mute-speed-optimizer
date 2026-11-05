"""Run: python -m mammal_model.tests.test_io  (from handover/)"""
import csv
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mammal_model.io import next_mammal_id, write_mammal_csv

ROOT = Path(__file__).resolve().parents[2]
REAL_MAMMAL_CSV = ROOT / "data" / "mammal_paths.csv"


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_next_mammal_id_with_no_base_csv():
    assert next_mammal_id(None) == 0


def test_next_mammal_id_from_real_data():
    # the shipped data uses ids 0-19 -- next should be 20
    assert next_mammal_id(REAL_MAMMAL_CSV) == 20


def test_write_mammal_csv_never_touches_base_csv():
    """The core safety property: generating a new mammal must not mutate the pinned shipped data
    file, byte for byte."""
    before = _sha256(REAL_MAMMAL_CSV)
    track = [{"minute": i, "lat": 48.5 + i * 0.001, "lon": -123.0} for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "merged.csv"
        write_mammal_csv(track, mammal_id=20, out_csv=out, base_csv=REAL_MAMMAL_CSV)
        assert out.exists()
    after = _sha256(REAL_MAMMAL_CSV)
    assert before == after, "base_csv was modified -- this must never happen"


def test_write_mammal_csv_merges_base_and_new_track():
    track = [{"minute": i, "lat": 48.5, "lon": -123.0} for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "merged.csv"
        write_mammal_csv(track, mammal_id=20, out_csv=out, base_csv=REAL_MAMMAL_CSV)
        rows = list(csv.DictReader(open(out)))
        ids = {int(r["mammal_id"]) for r in rows}
        assert ids == set(range(21))   # 0-19 from base + the new 20
        new_rows = [r for r in rows if int(r["mammal_id"]) == 20]
        assert len(new_rows) == 3


def test_write_mammal_csv_without_base_csv_writes_only_the_new_track():
    track = [{"minute": i, "lat": 48.5, "lon": -123.0} for i in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "new_only.csv"
        write_mammal_csv(track, mammal_id=0, out_csv=out, base_csv=None)
        rows = list(csv.DictReader(open(out)))
        assert len(rows) == 4
        assert {int(r["mammal_id"]) for r in rows} == {0}


def test_write_mammal_csv_rejects_duplicate_id_unless_overwrite():
    track = [{"minute": 0, "lat": 48.5, "lon": -123.0}]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dup.csv"
        try:
            write_mammal_csv(track, mammal_id=5, out_csv=out, base_csv=REAL_MAMMAL_CSV)   # id 5 already exists
            assert False, "expected ValueError for a duplicate mammal_id"
        except ValueError:
            pass
        # with overwrite_id=True it should succeed and REPLACE id 5's rows
        write_mammal_csv(track, mammal_id=5, out_csv=out, base_csv=REAL_MAMMAL_CSV, overwrite_id=True)
        rows = list(csv.DictReader(open(out)))
        id5_rows = [r for r in rows if int(r["mammal_id"]) == 5]
        assert len(id5_rows) == 1   # only the new single-point track, not the original 1000 rows


if __name__ == "__main__":
    test_next_mammal_id_with_no_base_csv()
    test_next_mammal_id_from_real_data()
    test_write_mammal_csv_never_touches_base_csv()
    test_write_mammal_csv_merges_base_and_new_track()
    test_write_mammal_csv_without_base_csv_writes_only_the_new_track()
    test_write_mammal_csv_rejects_duplicate_id_unless_overwrite()
    print("mammal_model.io: all tests passed")
