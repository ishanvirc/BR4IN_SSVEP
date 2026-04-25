"""Filter-Bank CCA (FBCCA) — STUB.

Reference: Chen et al. 2015, "Filter bank canonical correlation analysis
for implementing a high-speed SSVEP-based brain–computer interface."

Sketch of the algorithm to implement:

1. Decompose each test trial into ``N`` sub-band signals using zero-phase
   bandpass filters with shared upper edge but increasing lower edges,
   e.g. passbands ``[8, 90], [16, 90], [24, 90], ...`` Hz. Chen reports
   N = 7 sub-bands.
2. For each sub-band ``n`` and each candidate frequency ``f_k``, run the
   same CCA-vs-sin/cos-references procedure as the plain ``CCAClassifier``,
   producing a per-sub-band correlation ``ρ_n,k``.
3. Combine sub-band correlations with weights
        w_n = n^(-a) + b      (Chen reports a ≈ 1.25, b ≈ 0.25)
   into a per-class score
        S_k = sum_n  w_n * ρ_{n,k}^2
   Predict ``argmax_k S_k``.

Notes for implementation:
- Use ``mne.filter.filter_data`` or ``scipy.signal.butter + filtfilt`` for
  the sub-band decomposition. Match the order Chen used (Chebyshev I,
  order 4 in some implementations).
- Re-use :func:`ssvep.features.cca_reference_signals` for the reference
  block — same as plain CCA.
- The only knobs the team should expose: number of sub-bands, a, b.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class FBCCAClassifier:
    def __init__(
        self,
        stim_freqs: Iterable[float],
        fs: float,
        n_harmonics: int = 2,
        n_subbands: int = 7,
        a: float = 1.25,
        b: float = 0.25,
    ):
        self.stim_freqs = np.asarray(list(stim_freqs), dtype=np.float64)
        self.fs = float(fs)
        self.n_harmonics = int(n_harmonics)
        self.n_subbands = int(n_subbands)
        self.a = float(a)
        self.b = float(b)

    def fit(self, X=None, y=None) -> "FBCCAClassifier":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "FBCCAClassifier.predict is a stub — implement filter-bank "
            "decomposition + per-sub-band CCA + weighted fusion. See module "
            "docstring for the recipe (Chen et al. 2015)."
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
