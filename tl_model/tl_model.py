"""Unified transmission-loss model interface -- picks "light" or "full" via config, caches the
loaded model process-wide (same pattern as production's noise/model_cache.py, so a GA run or a
replan doesn't reload a 178MB-1.3GB pickle every call), and provides an optional trust-radius
safeguard for the light model.

Why both models are here, and why light is the actual GA/optimizer path:
- light (single-freq RBF) is ~10x faster per query than full (29-band + PCA), confirmed by direct
  benchmark (sandbox/optimizer/LOG.md, 2026-09-29): MAE 0.23-0.73dB, RMSE 0.38-0.96dB, correlation
  1.0000 vs. full's mean-across-29-bands. This is what the optimizer, the poster results, and the
  voyage animation all actually used.
- full is kept for TL-characterization work (sentinel-region mapping, per-band comparisons) -- not
  needed for a real optimizer run, but useful if you want to re-verify the agreement claim yourself
  or explore per-band TL.

Trust-radius safeguard: the light model's training data concentrates along real shipping lanes;
querying it well outside where it's ever been validated (>90-100km from a source, in this
sandbox's own testing) can return a physically-backward result (received level not decaying
monotonically with distance) because RBF interpolators have no extrapolation guarantee outside
their support region (found during the full-domain voyage animation, this session). `TLModel.query`
can optionally blank anything beyond `max_trusted_radius_km` rather than trust the model there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pyproj import Geod

GEOD = Geod(ellps="WGS84")

_model_cache: dict[str, object] = {}


@dataclass
class TLModelConfig:
    kind: Literal["light", "full"] = "light"
    light_model_dir: Optional[str] = None   # dir containing model.pkl
    full_model_dir: Optional[str] = None    # dir containing interpolators.pkl (+ ipca.pkl, yobs_mean.pkl)
    full_n_components: int = 10             # only used if (re)training the full model; import ignores it


class TLModelNotFoundError(FileNotFoundError):
    pass


def load_tl_model(config: TLModelConfig):
    """Process-wide cached load. Returns a callable model: model(query) -> TL array.
    query columns: [source_lat, source_lon, receiver_lat, receiver_lon] (matches production's
    column order). light model returns shape (n, 1) (squeeze to (n,) yourself if needed); full
    model returns shape (n, 29) (one column per decidecade band)."""
    key = f"{config.kind}:{config.light_model_dir}:{config.full_model_dir}"
    if key in _model_cache:
        return _model_cache[key]

    if config.kind == "light":
        model_dir = config.light_model_dir
        if not model_dir or not Path(model_dir).exists():
            raise TLModelNotFoundError(
                f"light TL model directory not found: {model_dir!r}. Either set the MUTE_LIGHT_MODEL_DIR "
                f"environment variable, or set tl_model.light_model_dir in your config, to a directory "
                f"containing model.pkl (see data/README.md).")
        from . import rbf_light
        model = rbf_light.RBFInterpolatorCustom.import_model(model_dir)
    elif config.kind == "full":
        model_dir = config.full_model_dir
        if not model_dir or not Path(model_dir).exists():
            raise TLModelNotFoundError(
                f"full TL model directory not found: {model_dir!r}. Either set the MUTE_FULL_MODEL_DIR "
                f"environment variable, or set tl_model.full_model_dir in your config, to a directory "
                f"containing interpolators.pkl (see data/README.md).")
        from . import rbf_full
        model = rbf_full.RBFInterpolatorCustom.import_model(model_dir)
    else:
        raise ValueError(f"unknown TL model kind: {config.kind!r} (expected 'light' or 'full')")

    _model_cache[key] = model
    return model


def query_tl(config: TLModelConfig, source_lat: float, source_lon: float,
             receiver_lat: np.ndarray, receiver_lon: np.ndarray,
             max_trusted_radius_km: Optional[float] = None) -> np.ndarray:
    """TL (dB) from one source to each receiver point. light model -> shape (n,) scalar TL.
    full model -> shape (n, 29), mean-across-bands if you want a single number (see
    objective/objective.py for how the optimizer reduces it). If max_trusted_radius_km is set,
    anything farther than that from the source is NaN rather than a possibly-extrapolated value."""
    model = load_tl_model(config)
    receiver_lat = np.asarray(receiver_lat)
    receiver_lon = np.asarray(receiver_lon)
    query = np.column_stack([
        np.full(receiver_lat.shape, source_lat), np.full(receiver_lon.shape, source_lon),
        receiver_lat, receiver_lon,
    ])
    tl = model(query)
    if config.kind == "light":
        tl = tl.ravel()

    if max_trusted_radius_km is not None:
        _, _, dist_m = GEOD.inv(np.full(receiver_lon.shape, source_lon), np.full(receiver_lat.shape, source_lat),
                                 receiver_lon, receiver_lat)
        dist_km = dist_m / 1000.0
        beyond = dist_km > max_trusted_radius_km
        if tl.ndim == 1:
            tl = np.where(beyond, np.nan, tl)
        else:
            tl = np.where(beyond[:, None], np.nan, tl)
    return tl
