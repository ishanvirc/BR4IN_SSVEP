"""Channel selection utilities for SSVEP decoding.

Two analyses for the hackathon:
  (a) Does a 4-channel occipital subset match 8-channel CCA performance?
  (b) Does subject 2's optimal channel subset differ from subject 1's?
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .features import snr_at_freq
from .evaluation import leave_one_block_out_cv


def rank_channels_by_snr(
    X: np.ndarray,
    y: np.ndarray,
    fs: float,
    stim_freqs: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Rank EEG channels by mean SSVEP SNR across all stimulation frequencies.

    For each channel, computes the mean SNR at each stimulation frequency
    using only trials where that frequency was the target, then averages
    across frequencies.

    Parameters
    ----------
    X          : ndarray, shape (n_trials, n_channels, n_samples)
    y          : ndarray, shape (n_trials,) — integer class indices into stim_freqs
    fs         : float — sampling rate (Hz)
    stim_freqs : sequence of floats — stimulation frequencies (Hz)

    Returns
    -------
    snr_scores : ndarray, shape (n_channels,) — mean SNR per channel (higher = better)
    ranking    : ndarray, shape (n_channels,) — channel indices sorted best→worst
    """
    stim_freqs_arr = np.asarray(list(stim_freqs), dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_channels = X.shape[1]

    snr_accum = np.zeros(n_channels, dtype=np.float64)
    n_counted = 0

    for c, freq in enumerate(stim_freqs_arr):
        mask = y == c
        if not mask.any():
            continue
        X_class = X[mask]  # (n_class_trials, n_channels, n_samples)
        # snr_at_freq accepts (..., n_samples) -> returns (...) leading shape
        snr = snr_at_freq(X_class, fs, freq)  # (n_class_trials, n_channels)
        snr_accum += np.nanmean(snr, axis=0)
        n_counted += 1

    if n_counted == 0:
        return np.zeros(n_channels), np.arange(n_channels)

    snr_scores = snr_accum / n_counted
    ranking = np.argsort(-snr_scores)  # descending: best channel first
    return snr_scores, ranking


def channel_snr_matrix(
    X: np.ndarray,
    y: np.ndarray,
    fs: float,
    stim_freqs: Iterable[float],
) -> np.ndarray:
    """Per-class per-channel SNR matrix.

    Parameters
    ----------
    X          : ndarray, shape (n_trials, n_channels, n_samples)
    y          : ndarray, shape (n_trials,)
    fs         : float
    stim_freqs : sequence of floats, length n_classes

    Returns
    -------
    snr_matrix : ndarray, shape (n_classes, n_channels)
        snr_matrix[c, ch] = mean SNR for channel ch when class c is the target.
    """
    stim_freqs_arr = np.asarray(list(stim_freqs), dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_classes = len(stim_freqs_arr)
    n_channels = X.shape[1]

    matrix = np.full((n_classes, n_channels), np.nan, dtype=np.float64)
    for c, freq in enumerate(stim_freqs_arr):
        mask = y == c
        if not mask.any():
            continue
        X_class = X[mask]  # (n_class_trials, n_channels, n_samples)
        snr = snr_at_freq(X_class, fs, freq)  # (n_class_trials, n_channels)
        matrix[c] = np.nanmean(snr, axis=0)

    return matrix


def sweep_channel_subsets(
    X: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    fs: float,
    stim_freqs: Iterable[float],
    channel_ranking: np.ndarray,
    n_harmonics: int = 2,
) -> dict[int, dict]:
    """LOBO-CV with CCA for each top-N channel subset.

    Sweeps n_channels from 1 to len(channel_ranking). Each iteration uses
    the top-N channels from ``channel_ranking`` (best SNR first).

    Parameters
    ----------
    X               : ndarray, shape (n_trials, n_channels, n_samples)
    y               : ndarray, shape (n_trials,)
    blocks          : ndarray, shape (n_trials,) — block IDs for LOBO-CV
    fs              : float
    stim_freqs      : sequence of floats
    channel_ranking : ndarray — channel indices sorted best→worst
    n_harmonics     : int, default 2

    Returns
    -------
    dict mapping n_channels (int) -> LOBO-CV result dict
        {'mean': float, 'std': float, 'per_block': dict[int, float]}
    """
    from .classifiers.cca import CCAClassifier

    stim_freqs_list = list(stim_freqs)
    X = np.asarray(X, dtype=np.float64)
    results: dict[int, dict] = {}

    for n_ch in range(1, len(channel_ranking) + 1):
        top_ch = channel_ranking[:n_ch]
        X_sub = X[:, top_ch, :]

        def _factory(sf=stim_freqs_list, f=fs, nh=n_harmonics):
            return CCAClassifier(stim_freqs=sf, fs=f, n_harmonics=nh)

        results[n_ch] = leave_one_block_out_cv(X_sub, y, blocks, _factory)

    return results
