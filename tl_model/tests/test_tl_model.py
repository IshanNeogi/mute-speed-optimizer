"""Run: python -m tl_model.tests.test_tl_model  (from handover/)

Skips cleanly (prints SKIP, doesn't fail) if the external model files aren't present locally --
they're not shipped with this repo, see data/README.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tl_model.tl_model import TLModelConfig, TLModelNotFoundError, query_tl

# Defaults match this dev machine's layout (../MUTE checked out next to MUTEApp); override via env
# vars or edit here if your layout differs. See config/default_config.yaml for the normal way to
# point at these paths when actually running the optimizer.
import os
_MUTE_ROOT = Path(os.environ.get("MUTE_ROOT", Path(__file__).resolve().parents[5] / "MUTE"))
LIGHT_DIR = str(_MUTE_ROOT / "v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq")
FULL_DIR = str(_MUTE_ROOT / "v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model")

# A source/receiver pair known to be in-domain, in-water (San Juan Islands cluster).
S_LAT, S_LON = 48.60, -123.15
R_LAT, R_LON = np.array([48.65]), np.array([-123.10])


def _light_available():
    return Path(LIGHT_DIR).exists()


def _full_available():
    return Path(FULL_DIR).exists()


def test_light_model_loads_and_queries():
    if not _light_available():
        print("SKIP test_light_model_loads_and_queries (no local model files)")
        return
    cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)
    tl = query_tl(cfg, S_LAT, S_LON, R_LAT, R_LON)
    assert tl.shape == (1,)
    assert 0 < tl[0] < 300  # sane TL range, not a sentinel/garbage value at this close range


def test_missing_model_dir_raises_clear_error():
    cfg = TLModelConfig(kind="light", light_model_dir="/nonexistent/path/xyz")
    try:
        query_tl(cfg, S_LAT, S_LON, R_LAT, R_LON)
        assert False, "expected TLModelNotFoundError"
    except TLModelNotFoundError as e:
        assert "light_model_dir" in str(e)


def test_trust_radius_masks_far_queries():
    if not _light_available():
        print("SKIP test_trust_radius_masks_far_queries (no local model files)")
        return
    cfg = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)
    far_lat, far_lon = np.array([51.0]), np.array([-120.0])  # ~300km+ away
    tl_unmasked = query_tl(cfg, S_LAT, S_LON, far_lat, far_lon)
    tl_masked = query_tl(cfg, S_LAT, S_LON, far_lat, far_lon, max_trusted_radius_km=90.0)
    assert np.isfinite(tl_unmasked[0])   # model still returns SOMETHING (possibly extrapolated)
    assert np.isnan(tl_masked[0])        # but we don't trust it past the radius


def test_light_vs_full_agree_within_1db():
    """Cross-check from LOG.md: light vs. full mean-across-29-bands agree within ~1dB in-domain."""
    if not (_light_available() and _full_available()):
        print("SKIP test_light_vs_full_agree_within_1db (no local model files)")
        return
    cfg_light = TLModelConfig(kind="light", light_model_dir=LIGHT_DIR)
    cfg_full = TLModelConfig(kind="full", full_model_dir=FULL_DIR)
    tl_light = query_tl(cfg_light, S_LAT, S_LON, R_LAT, R_LON)
    tl_full = query_tl(cfg_full, S_LAT, S_LON, R_LAT, R_LON)
    tl_full_mean = tl_full.mean(axis=1)
    assert abs(tl_light[0] - tl_full_mean[0]) < 1.0


if __name__ == "__main__":
    test_light_model_loads_and_queries()
    test_missing_model_dir_raises_clear_error()
    test_trust_radius_masks_far_queries()
    test_light_vs_full_agree_within_1db()
    print("tl_model: all tests passed (or skipped where model files are absent)")
