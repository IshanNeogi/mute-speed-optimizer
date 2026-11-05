"""Custom RBF transmission-loss interpolator -- "full" 29-band + IncrementalPCA variant.

Verbatim port from ../MUTE v2/src/noise_propagation_model/noise_propagation_model/models/RBF/
utils_rbf.py (via sandbox/optimizer/received_model/rbf_full.py). Pure math/scipy/sklearn, zero
ROS2 dependency -- no logic changes (dropped 1 unused import: MinMaxScaler; ProcessPoolExecutor
also unused in the original, dropped). NOT the model production currently wires up (production
uses the light model only) -- kept here for TL-characterization work (sentinel-region mapping,
model agreement checks) per the project's "port both, config chooses" decision.

Class name is identical to rbf_light.py's `RBFInterpolatorCustom` (same in the original ../MUTE
source) -- always import under the module name (`rbf_full` vs. `rbf_light`), never both unqualified,
to avoid collisions.

The trained model files (../MUTE's models/RBF/model/: interpolators.pkl 1.31GB, ipca.pkl 3.7KB,
yobs_mean.pkl 379B) are NOT shipped here -- point tl_model.TLModelConfig.full_model_dir at your own
copy. See data/README.md.
"""
import os
import pickle

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import IncrementalPCA


class RBFInterpolatorCustom(RBFInterpolator):
    """Vector-valued RBF interpolator with optional IncrementalPCA dimensionality reduction on
    the output (yobs) side. xobs: (n_points, n_features). yobs: (n_points, n_outputs)."""

    def __init__(self, xobs, yobs, model_path, n_components, *args, **kwargs):
        assert yobs.ndim == 2, "yobs must be a 2D array with multiple components (columns)."
        xobs, yobs = self._remove_duplicates(xobs, yobs)
        self.model_path = model_path

        if n_components >= yobs.shape[1]:
            yobs_transformed = yobs
            self.ipca = None
            self.yobs_mean = None
        else:
            yobs_transformed = self._create_pca(yobs, n_components)

        self.num_components = yobs_transformed.shape[1]
        self.interpolators = [
            RBFInterpolator(xobs, yobs_transformed[:, i], neighbors=75, kernel="gaussian",
                             epsilon=1000.0, *args, **kwargs)
            for i in range(self.num_components)
        ]

    def _create_pca(self, yobs, n_components, batch_size=10000):
        self.yobs_mean = yobs.mean(axis=0)
        yobs_centered = yobs - self.yobs_mean
        self.ipca = IncrementalPCA(n_components=n_components)
        n = yobs_centered.shape[0]
        for i in range(0, n, batch_size):
            self.ipca.partial_fit(yobs_centered[i:i + batch_size])
        batches = [self.ipca.transform(yobs_centered[i:i + batch_size]) for i in range(0, n, batch_size)]
        return np.vstack(batches)

    def __call__(self, x):
        y_transformed = np.column_stack([interp(x) for interp in self.interpolators])
        if self.ipca is None:
            return y_transformed
        batch_size = 10000
        n = y_transformed.shape[0]
        batches = [self.ipca.inverse_transform(y_transformed[i:i + batch_size]) for i in range(0, n, batch_size)]
        return np.vstack(batches) + self.yobs_mean

    def _remove_duplicates(self, xobs, yobs, tolerance=1e-5):
        _, unique_indices = np.unique(np.round(xobs / tolerance), axis=0, return_index=True)
        return xobs[unique_indices], yobs[unique_indices]

    def export_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        with open(os.path.join(self.model_path, "interpolators.pkl"), "wb") as f:
            pickle.dump(self.interpolators, f)
        if self.ipca is not None:
            with open(os.path.join(self.model_path, "ipca.pkl"), "wb") as f:
                pickle.dump(self.ipca, f)
            with open(os.path.join(self.model_path, "yobs_mean.pkl"), "wb") as f:
                pickle.dump(self.yobs_mean, f)

    @staticmethod
    def import_model(model_path):
        with open(os.path.join(model_path, "interpolators.pkl"), "rb") as f:
            interpolators = pickle.load(f)
        ipca_file = os.path.join(model_path, "ipca.pkl")
        if os.path.exists(ipca_file):
            with open(ipca_file, "rb") as f:
                ipca = pickle.load(f)
            with open(os.path.join(model_path, "yobs_mean.pkl"), "rb") as f:
                yobs_mean = pickle.load(f)
        else:
            ipca, yobs_mean = None, None

        instance = RBFInterpolatorCustom.__new__(RBFInterpolatorCustom)
        instance.interpolators = interpolators
        instance.num_components = len(interpolators)
        instance.ipca = ipca
        instance.yobs_mean = yobs_mean
        instance.model_path = model_path
        return instance
