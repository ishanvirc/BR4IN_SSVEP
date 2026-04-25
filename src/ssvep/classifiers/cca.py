"""Standard CCA classifier for SSVEP decoding.

For each candidate stimulation frequency f_k we build a reference signal
block of sin/cos harmonics. For a test trial X (n_channels, n_samples) we
fit ``sklearn.cross_decomposition.CCA(n_components=1)`` between X.T and
the reference block, take the top canonical correlation ρ_k, and predict
the class with the largest ρ_k.

The classifier needs no training data — `.fit` is a no-op kept only for
sklearn-style API compatibility (so it can be slotted into the same CV
harness as supervised classifiers).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.cross_decomposition import CCA

from ..features import cca_reference_signals


class CCAClassifier:
    def __init__(
        self,
        stim_freqs: Iterable[float],
        fs: float,
        n_harmonics: int = 2,
    ):
        self.stim_freqs = np.asarray(list(stim_freqs), dtype=np.float64)
        self.fs = float(fs)
        self.n_harmonics = int(n_harmonics)
        self._refs_cache: tuple[int, np.ndarray] | None = None

    # sklearn-style API -------------------------------------------------
    def fit(self, X=None, y=None) -> "CCAClassifier":
        """No-op. Reference signals are built lazily in :meth:`predict`."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices (0..n_classes-1) for each trial.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)

        Returns
        -------
        y_pred : ndarray, shape (n_trials,), dtype int64 — class indices into
                 ``self.stim_freqs``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (n_trials, n_channels, n_samples); got {X.shape}")
        n_trials, n_channels, n_samples = X.shape

        refs = self._get_refs(n_samples)  # (n_freqs, 2*H, n_samples)
        n_freqs = refs.shape[0]

        # Underdetermined check: skip per-trial CCA if dims won't support it.
        min_dim = 2 * self.n_harmonics
        if n_samples < n_channels + min_dim:
            return np.zeros(n_trials, dtype=np.int64)

        rhos = np.zeros((n_trials, n_freqs), dtype=np.float64)
        for i in range(n_trials):
            xi = X[i].T  # (n_samples, n_channels)
            for k in range(n_freqs):
                yi = refs[k].T  # (n_samples, 2*H)
                cca = CCA(n_components=1, max_iter=500)
                try:
                    u, v = cca.fit_transform(xi, yi)
                except Exception:
                    rhos[i, k] = 0.0
                    continue
                u = u[:, 0]
                v = v[:, 0]
                if np.std(u) < 1e-12 or np.std(v) < 1e-12:
                    rhos[i, k] = 0.0
                else:
                    rhos[i, k] = float(np.corrcoef(u, v)[0, 1])
        return np.argmax(rhos, axis=1).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean accuracy. ``y`` must be integer class indices into ``stim_freqs``."""
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y).astype(np.int64)))

    # Internals ---------------------------------------------------------
    def _get_refs(self, n_samples: int) -> np.ndarray:
        if self._refs_cache is not None and self._refs_cache[0] == n_samples:
            return self._refs_cache[1]
        refs = cca_reference_signals(
            self.stim_freqs, self.n_harmonics, self.fs, n_samples
        )
        self._refs_cache = (n_samples, refs)
        return refs
