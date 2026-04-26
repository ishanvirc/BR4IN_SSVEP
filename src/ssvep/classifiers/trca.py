"""TRCA classifier per Nakanishi et al. 2018.

Algorithm ported from meegkit (BSD-3-Clause), specifically the standalone
trca() function from meegkit/trca.py:
https://github.com/nbara/python-meegkit/blob/master/meegkit/trca.py

Original algorithm: Nakanishi et al., "Enhancing Detection of SSVEPs for a
High-Speed Brain Speller Using Task-Related Component Analysis", IEEE TBME
65(1):104-112, 2018.

See LICENSES/meegkit-BSD-3.txt for the full BSD-3 license text.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.linalg as linalg


def _theshapeof(X: np.ndarray) -> tuple[int, int, int]:
    """Return (n_samples, n_chans, n_trials). Inlined from meegkit.utils."""
    if X.ndim == 3:
        n_samples, n_chans, n_trials = X.shape
    elif X.ndim == 2:
        n_samples, n_chans = X.shape
        n_trials = 1
    else:
        raise ValueError(f"Expected 2D or 3D array; got shape {X.shape}")
    return n_samples, n_chans, n_trials


def _trca(X: np.ndarray) -> np.ndarray:
    """Task-related component analysis spatial filter (single class).

    Ported from meegkit/trca.py (BSD-3). Input shape:
    ``(n_samples, n_chans[, n_trials])``. Returns a real spatial filter of
    shape ``(n_chans,)`` — the eigenvector with the largest eigenvalue of
    the generalized eigenproblem ``S w = λ Q w``, where ``Q`` is the
    within-class scatter on concatenated trials and ``S`` is the inter-trial
    scatter summed over distinct trial pairs.
    """
    n_samples, n_chans, n_trials = _theshapeof(X)
    if X.ndim == 2:
        X = X[..., None]

    UX = np.zeros((n_chans, n_samples * n_trials))
    for trial in range(n_trials):
        UX[:, trial * n_samples:(trial + 1) * n_samples] = X[..., trial].T
    UX -= np.mean(UX, 1)[:, None]
    Q = UX @ UX.T

    S = np.zeros((n_chans, n_chans))
    for i in range(n_trials - 1):
        x1 = np.squeeze(X[..., i]).copy()
        x1 -= np.mean(x1, 0)
        for j in range(i + 1, n_trials):
            x2 = np.squeeze(X[..., j]).copy()
            x2 -= np.mean(x2, 0)
            S = S + x1.T @ x2 + x2.T @ x1

    lambdas, W = linalg.eig(S, Q, left=True, right=False)
    W_best = W[:, int(np.argmax(np.real(lambdas)))]
    return np.real(W_best)


class TRCAClassifier:
    """TRCA classifier matching the CCAClassifier interface.

    Parameters
    ----------
    stim_freqs : iterable of stim frequencies. TRCA is data-driven and does
        not use the frequencies in math; carried for API symmetry with
        CCAClassifier.
    fs : sampling rate (Hz). Carried for API symmetry; not used internally.
    ensemble : if True (default), use ensemble TRCA per Nakanishi 2018 —
        each class is scored by projecting test + template through the full
        stack of class-wise spatial filters and correlating the flattened
        result. If False, only the class's own filter is used.

    The caller-facing API uses ``(n_trials, n_channels, n_samples)`` —
    sklearn / CCAClassifier convention. The internal :func:`_trca` helper
    expects ``(n_samples, n_channels, n_trials)``; the transpose happens at
    the boundary and never leaks to callers.
    """

    def __init__(
        self,
        stim_freqs: Iterable[float],
        fs: float,
        ensemble: bool = True,
    ):
        self.stim_freqs = np.asarray(list(stim_freqs), dtype=np.float64)
        self.fs = float(fs)
        self.ensemble = bool(ensemble)
        self.classes_: np.ndarray | None = None
        self.spatial_filters_: dict[int, np.ndarray] | None = None
        self.templates_: dict[int, np.ndarray] | None = None
        self.W_ensemble_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCAClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.int64)
        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape (n_trials, n_channels, n_samples); got {X.shape}"
            )

        self.classes_ = np.unique(y)
        self.spatial_filters_ = {}
        self.templates_ = {}

        for c in self.classes_:
            c_int = int(c)
            X_c = X[y == c]
            if X_c.shape[0] < 2:
                raise ValueError(
                    f"TRCA needs >=2 trials per class; class {c_int} has {X_c.shape[0]}."
                )
            X_internal = X_c.transpose(2, 1, 0)
            self.spatial_filters_[c_int] = _trca(X_internal)
            self.templates_[c_int] = X_c.mean(axis=0)

        if self.ensemble:
            self.W_ensemble_ = np.column_stack(
                [self.spatial_filters_[int(c)] for c in self.classes_]
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.spatial_filters_ is None:
            raise RuntimeError("TRCAClassifier.predict called before fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape (n_trials, n_channels, n_samples); got {X.shape}"
            )

        classes = self.classes_
        preds = np.empty(X.shape[0], dtype=np.int64)
        for i, x_test in enumerate(X):
            rhos = np.empty(len(classes), dtype=np.float64)
            for k, c in enumerate(classes):
                template = self.templates_[int(c)]
                if self.ensemble:
                    proj_test = self.W_ensemble_.T @ x_test
                    proj_temp = self.W_ensemble_.T @ template
                    a = proj_test.ravel()
                    b = proj_temp.ravel()
                else:
                    w = self.spatial_filters_[int(c)]
                    a = w @ x_test
                    b = w @ template
                if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                    rhos[k] = 0.0
                else:
                    rhos[k] = float(np.corrcoef(a, b)[0, 1])
            preds[i] = int(classes[int(np.argmax(rhos))])
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(np.int64)))
