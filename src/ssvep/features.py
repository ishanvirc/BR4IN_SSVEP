"""Feature extraction primitives: reference signals, PSD, SNR."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import signal as sps


def cca_reference_signals(
    freqs: Iterable[float],
    n_harmonics: int,
    fs: float,
    n_samples: int,
) -> np.ndarray:
    """Build sin/cos reference signals for CCA-based SSVEP decoding.

    For each candidate stimulation frequency f_k and harmonic h in 1..H,
    we emit two rows: ``sin(2π · h · f_k · t)`` and ``cos(2π · h · f_k · t)``
    where ``t = arange(n_samples) / fs``.

    Returns
    -------
    refs : ndarray, shape (n_freqs, 2 * n_harmonics, n_samples)
        ``refs[k]`` is the reference signal block for class k, with rows
        ordered ``[sin(h=1), cos(h=1), sin(h=2), cos(h=2), ...]``.
    """
    freqs_arr = np.asarray(list(freqs), dtype=np.float64)
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    t = np.arange(n_samples, dtype=np.float64) / float(fs)
    n_freqs = freqs_arr.size
    refs = np.empty((n_freqs, 2 * n_harmonics, n_samples), dtype=np.float64)
    for k, f in enumerate(freqs_arr):
        for h in range(1, n_harmonics + 1):
            phase = 2.0 * np.pi * h * f * t
            refs[k, 2 * (h - 1)] = np.sin(phase)
            refs[k, 2 * (h - 1) + 1] = np.cos(phase)
    return refs


def psd(
    X: np.ndarray,
    fs: float,
    fmin: float = 0.0,
    fmax: float | None = None,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD across the last axis of X.

    Parameters
    ----------
    X       : ndarray, shape (..., n_samples)
    fs      : sampling rate (Hz)
    fmin    : low end of returned band (inclusive)
    fmax    : high end of returned band (inclusive). None → Nyquist.
    nperseg : Welch segment length; defaults to min(256, n_samples).

    Returns
    -------
    freqs : ndarray, shape (n_freq_bins,)
    pxx   : ndarray, shape (..., n_freq_bins)
    """
    n_samples = X.shape[-1]
    if nperseg is None:
        nperseg = int(min(256, n_samples))
    f, p = sps.welch(X, fs=fs, nperseg=nperseg, axis=-1)
    if fmax is None:
        fmax = float(fs) / 2.0
    mask = (f >= fmin) & (f <= fmax)
    return f[mask], p[..., mask]


def snr_at_freq(
    X: np.ndarray,
    fs: float,
    target_freq: float,
    neighbor_band: float = 1.0,
    nperseg: int | None = None,
) -> np.ndarray:
    """SNR at ``target_freq``: power at target / mean power in flanking band.

    Flanking band is ``[target_freq - neighbor_band, target_freq + neighbor_band]``
    excluding a small ±0.1 Hz core around the target.

    Parameters
    ----------
    X            : ndarray, shape (..., n_samples)
    fs           : sampling rate (Hz)
    target_freq  : frequency of interest (Hz)
    neighbor_band: half-width of flanking band (Hz)

    Returns
    -------
    snr : ndarray, shape (...) — same leading dims as X
    """
    f, p = psd(X, fs=fs, nperseg=nperseg)
    target_idx = int(np.argmin(np.abs(f - target_freq)))
    target_power = p[..., target_idx]
    flank_mask = (
        (f >= target_freq - neighbor_band)
        & (f <= target_freq + neighbor_band)
        & (np.abs(f - target_freq) > 0.1)
    )
    if not flank_mask.any():
        return np.full(target_power.shape, np.nan)
    flank_power = p[..., flank_mask].mean(axis=-1)
    return target_power / np.where(flank_power > 0, flank_power, np.nan)
