"""FBCCA classifier per Chen et al. 2015.

Algorithm structure adapted from
github.com/eugeneALU/CECNL_RealTimeBCI/blob/master/fbcca.py
(no explicit license stated; algorithm is public, reproduced from
Chen 2015 J. Neural Eng. 12:046008).

Sub-band design and weighting per Chen, X., Wang, Y., Gao, S.,
Jung, T.-P., Gao, X. (2015). "Filter bank canonical correlation
analysis for implementing a high-speed SSVEP-based brain-computer
interface." Journal of Neural Engineering, 12(4):046008.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.signal as signal
from sklearn.cross_decomposition import CCA

from ..features import cca_reference_signals


def _build_subband_filters(fs: float, num_subbands: int):
    """Chen 2015 §3.2.1 M3 design — Chebyshev I bandpass filter bank.

    Sub-band ``n`` (1-indexed): passband ``[6 + 8*(n-1), 90]`` Hz,
    stopband ``[4 + 8*(n-1), 100]`` Hz, gpass=3 dB, gstop=40 dB,
    ripple Rp=0.5 dB. Returns a list of ``(b, a)`` tuples for
    :func:`scipy.signal.filtfilt`.
    """
    nyq = fs / 2.0
    filters = []
    for n in range(1, num_subbands + 1):
        wp = ((6 + 8 * (n - 1)) / nyq, 90.0 / nyq)
        ws = ((4 + 8 * (n - 1)) / nyq, 100.0 / nyq)
        N, Wn = signal.cheb1ord(wp, ws, gpass=3, gstop=40)
        b, a = signal.cheby1(N, rp=0.5, Wn=Wn, btype="bandpass")
        filters.append((b, a))
    return filters


def _max_cca_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Maximum canonical correlation between ``X`` (n_chans, n_samples) and
    ``Y`` (2*H, n_samples). Mirrors :class:`CCAClassifier`'s pattern:
    ``n_components=1``, ``max_iter=500``, :func:`numpy.corrcoef`, std-floor
    edge case for degenerate fits.
    """
    cca = CCA(n_components=1, max_iter=500)
    try:
        u, v = cca.fit_transform(X.T, Y.T)
    except Exception:
        return 0.0
    u = u[:, 0]
    v = v[:, 0]
    if np.std(u) < 1e-12 or np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(u, v)[0, 1])


class FBCCAClassifier:
    """FBCCA classifier matching the :class:`CCAClassifier` interface.

    Parameters
    ----------
    stim_freqs : iterable of stim frequencies (n_classes,).
    fs : sampling rate (Hz).
    num_harmonics : harmonics in CCA reference signals (default 5).
    num_subbands : filter bank sub-bands (default 5 per Chen 2015).

    The caller-facing API uses ``(n_trials, n_chans, n_samples)`` —
    sklearn / CCAClassifier convention. ``fit`` is a no-op (FBCCA is
    data-driven only via the reference signals); the method exists for
    API symmetry with the LOBO harness.
    """

    def __init__(
        self,
        stim_freqs: Iterable[float],
        fs: float,
        num_harmonics: int = 5,
        num_subbands: int = 5,
    ):
        self.stim_freqs = np.asarray(list(stim_freqs), dtype=np.float64)
        self.fs = float(fs)
        self.num_harmonics = int(num_harmonics)
        self.num_subbands = int(num_subbands)
        self._fb_weights = np.array(
            [n ** (-1.25) + 0.25 for n in range(1, self.num_subbands + 1)],
            dtype=np.float64,
        )
        self._subband_filters = _build_subband_filters(self.fs, self.num_subbands)
        self._refs_cache: tuple[int, np.ndarray] | None = None

    def _get_refs(self, n_samples: int) -> np.ndarray:
        if self._refs_cache is not None and self._refs_cache[0] == n_samples:
            return self._refs_cache[1]
        refs = cca_reference_signals(
            self.stim_freqs, self.num_harmonics, self.fs, n_samples
        )
        self._refs_cache = (n_samples, refs)
        return refs

    def fit(self, X=None, y=None) -> "FBCCAClassifier":
        """No-op. References are built lazily in :meth:`predict`."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape (n_trials, n_channels, n_samples); got {X.shape}"
            )
        n_trials, n_channels, n_samples = X.shape
        refs = self._get_refs(n_samples)
        n_classes = refs.shape[0]

        preds = np.empty(n_trials, dtype=np.int64)
        for i in range(n_trials):
            x_test = X[i]
            rho_per_class = np.zeros(n_classes, dtype=np.float64)
            for b_idx, (bb, aa) in enumerate(self._subband_filters):
                X_b = signal.filtfilt(bb, aa, x_test, axis=-1)
                for k in range(n_classes):
                    rho = _max_cca_correlation(X_b, refs[k])
                    rho_per_class[k] += self._fb_weights[b_idx] * (rho ** 2)
            preds[i] = int(np.argmax(rho_per_class))
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(np.int64)))
