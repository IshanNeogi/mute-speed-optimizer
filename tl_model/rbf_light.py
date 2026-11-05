"""Custom RBF transmission-loss interpolator -- "light" single-frequency variant.

Verbatim port from ../MUTE v2/src/noise_propagation_model/noise_propagation_model/models/RBF/utils2.py
(via sandbox/optimizer/received_model/rbf_light.py). Pure math/scipy, zero ROS2 dependency -- no
logic changes (dropped 3 unused imports: MinMaxScaler, geopandas, shapely.box, ProcessPoolExecutor
-- none were referenced anywhere in the original class).

The trained model file (model_single_freq/model.pkl, ~178MB) is NOT shipped here -- point
tl_model.TLModelConfig.light_model_dir at your own copy (e.g. ../MUTE's path). See data/README.md.
"""
import os
import pickle

import numpy as np
from scipy.interpolate import RBFInterpolator


class RBFInterpolatorCustom(RBFInterpolator):
    """Vector-valued RBF interpolator. xobs: (n_points, n_features). yobs: (n_points, n_outputs)."""

    def __init__(self, xobs, yobs, model_path, *args, **kwargs):
        assert yobs.ndim == 2, "yobs must be a 2D array with multiple components (columns)."
        xobs, yobs = self._remove_duplicates(xobs, yobs)
        self.model_path = model_path
        self.num_components = yobs.shape[1]
        self.interpolators = [
            RBFInterpolator(xobs, yobs[:, i], neighbors=75, kernel="gaussian", epsilon=1000.0, *args, **kwargs)
            for i in range(self.num_components)
        ]

    def __call__(self, x):
        return np.column_stack([interp(x) for interp in self.interpolators])

    def _remove_duplicates(self, xobs, yobs, tolerance=1e-5):
        _, unique_indices = np.unique(np.round(xobs / tolerance), axis=0, return_index=True)
        return xobs[unique_indices], yobs[unique_indices]

    def export_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        with open(os.path.join(self.model_path, "model.pkl"), "wb") as f:
            pickle.dump(self.interpolators, f)

    @staticmethod
    def import_model(model_path):
        with open(os.path.join(model_path, "model.pkl"), "rb") as f:
            interpolators = pickle.load(f)
        instance = RBFInterpolatorCustom.__new__(RBFInterpolatorCustom)
        instance.interpolators = interpolators
        instance.model_path = model_path
        return instance
