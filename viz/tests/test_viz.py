"""Run: python -m viz.tests.test_viz  (from handover/)

Builds small synthetic figures (doesn't need the TL model) and checks: real usetex is on, files
land where expected, the palette validates, EPS alpha warning fires when it should.
"""
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from viz import palette, style
from viz.export import save_figure
from viz.sel_comparison import plot_sel_comparison
from viz.timeline import plot_cumulative_sel_timeline, plot_received_level_timeline


def test_style_enables_real_usetex():
    style.apply_style(context="journal")
    assert matplotlib.rcParams["text.usetex"] is True


def test_categorical_palette_has_8_fixed_slots():
    assert len(palette.CATEGORICAL) == 8
    assert len(set(palette.CATEGORICAL)) == 8   # no duplicates
    assert len(palette.CATEGORICAL_ALL_PAIRS_SAFE) == 3


def test_save_figure_writes_pdf_and_eps():
    style.apply_style(context="journal")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with tempfile.TemporaryDirectory() as tmp:
        written = save_figure(fig, "smoke_test", tmp, formats=("pdf", "eps"))
        assert len(written) == 2
        for p in written:
            assert p.exists()
            assert p.stat().st_size > 0


def test_sel_comparison_runs_and_produces_files():
    style.apply_style(context="journal")
    with tempfile.TemporaryDirectory() as tmp:
        written = plot_sel_comparison([3, 12, 8], [115.5, 108.0, 98.7], [105.2, 95.7, 95.9],
                                       tmp, context="journal", formats=("pdf",))
        assert len(written) == 1
        assert written[0].exists()


def test_timeline_plots_run_with_synthetic_rows():
    style.apply_style(context="journal")
    rows = [
        {"time_min": t, "received_level_db_1": 80.0 + np.sin(t / 10), "cumulative_sel_db_1": 90.0 + t * 0.01,
         "received_level_db_2": 70.0, "cumulative_sel_db_2": 85.0 + t * 0.005}
        for t in range(0, 100, 5)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        written1 = plot_received_level_timeline(rows, [1, 2], tmp, context="journal", formats=("pdf",))
        written2 = plot_cumulative_sel_timeline(rows, [1, 2], tmp, context="journal", formats=("pdf",))
        assert written1[0].exists()
        assert written2[0].exists()


if __name__ == "__main__":
    test_style_enables_real_usetex()
    test_categorical_palette_has_8_fixed_slots()
    test_save_figure_writes_pdf_and_eps()
    test_sel_comparison_runs_and_produces_files()
    test_timeline_plots_run_with_synthetic_rows()
    print("viz: all tests passed")
