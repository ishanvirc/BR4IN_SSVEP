"""Task-Related Component Analysis (TRCA) — STUB.

Reference: Nakanishi et al. 2017, "Enhancing Detection of SSVEPs for a
High-Speed Brain Speller Using Task-Related Component Analysis."

Sketch of the algorithm to implement:

For each class ``c`` we learn a spatial filter ``w_c ∈ R^{n_channels}``
that maximizes inter-trial reproducibility within class ``c``:

    w_c = argmax_w   (w^T S_c w) / (w^T Q_c w)

where, given training trials ``X_c^{(1)}, ..., X_c^{(K)}`` ∈ R^{n_ch × n_samp}:

    S_c = sum_{i ≠ j}   X_c^{(i)} (X_c^{(j)})^T        (inter-trial covariance)
    Q_c = sum_i         X_c^{(i)} (X_c^{(i)})^T        (intra-trial covariance)

This is a generalized eigenvalue problem ``S_c w = λ Q_c w``; ``w_c`` is the
eigenvector with the largest eigenvalue. Solve with
``scipy.linalg.eigh(S_c, Q_c)`` and take the last column.

Templates: ``T_c = mean_i X_c^{(i)}`` (n_ch × n_samp).

Prediction for a test trial ``X``: for each class compute
``ρ_c = corr( w_c^T X ,  w_c^T T_c )`` and pick ``argmax_c ρ_c``.

The published version of TRCA also adds the CCA correlation against
sin/cos references and combines via ensemble (use eigenvectors from all
classes, not just class ``c``). Start with the basic single-filter
version; layer on the ensemble + CCA-augmented score if time permits.

Notes for implementation:
- ``fit(X, y)`` is required (TRCA is supervised, unlike CCA).
- Returns class indices in the same labeling scheme used by ``y``;
  remember to map class labels ↔ ``stim_freqs`` indices consistently.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class TRCAClassifier:
    def __init__(
        self,
        stim_freqs: Iterable[float],
        fs: float,
        ensemble: bool = False,
    ):
        self.stim_freqs = np.asarray(list(stim_freqs), dtype=np.float64)
        self.fs = float(fs)
        self.ensemble = bool(ensemble)
        # Populated by .fit:
        self.spatial_filters_: np.ndarray | None = None  # (n_classes, n_channels)
        self.templates_: np.ndarray | None = None        # (n_classes, n_channels, n_samples)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCAClassifier":
        raise NotImplementedError(
            "TRCAClassifier.fit is a stub — solve the per-class generalized "
            "eigenvalue problem S_c w = λ Q_c w and store spatial filters + "
            "trial-averaged templates. See module docstring for the recipe "
            "(Nakanishi et al. 2017)."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "TRCAClassifier.predict is a stub — score each class via "
            "corr(w_c^T X_test, w_c^T T_c) and argmax. See module docstring."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
