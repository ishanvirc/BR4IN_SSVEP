"""Preprocessing wrappers around MNE primitives.

All functions accept ``X`` with shape ``(n_trials, n_channels, n_samples)``
and return the same shape. MNE expects ``(n_channels, n_samples)`` per
trial, so we iterate trials internally.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

import mne


def notch_filter(
    X: np.ndarray,
    fs: float,
    freqs: Iterable[float] = (50.0, 60.0),
    verbose: bool = False,
) -> np.ndarray:
    """Apply notch filter at each frequency in ``freqs`` (Hz)."""
    freqs_arr = np.asarray(list(freqs), dtype=np.float64)
    out = np.empty_like(X, dtype=np.float64)
    for i, trial in enumerate(X):
        out[i] = mne.filter.notch_filter(
            trial.astype(np.float64), Fs=fs, freqs=freqs_arr, verbose=verbose
        )
    return out


def bandpass(
    X: np.ndarray,
    fs: float,
    l_freq: float | None,
    h_freq: float | None,
    verbose: bool = False,
) -> np.ndarray:
    """Bandpass filter via MNE. Pass None for either edge to make it open."""
    out = np.empty_like(X, dtype=np.float64)
    for i, trial in enumerate(X):
        out[i] = mne.filter.filter_data(
            trial.astype(np.float64),
            sfreq=fs,
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=verbose,
        )
    return out


def epoch(
    continuous: np.ndarray,
    events: np.ndarray,
    fs: float,
    tmin: float,
    tmax: float,
) -> np.ndarray:
    """Slice a continuous (n_channels, n_samples) recording into trials.

    Parameters
    ----------
    continuous : ndarray, shape (n_channels, n_samples)
    events     : ndarray, shape (n_events,) of sample indices marking stim onset
    fs         : sampling rate (Hz)
    tmin       : seconds relative to event (negative for pre-stim)
    tmax       : seconds relative to event (positive for post-stim)

    Returns
    -------
    epochs : ndarray, shape (n_events, n_channels, n_samples_per_epoch)
    """
    if continuous.ndim != 2:
        raise ValueError(f"continuous must be 2D (n_channels, n_samples), got {continuous.shape}")
    n_pre = int(round(tmin * fs))
    n_post = int(round(tmax * fs))
    width = n_post - n_pre
    n_ch = continuous.shape[0]
    n_total = continuous.shape[1]
    out = np.empty((len(events), n_ch, width), dtype=np.float64)
    for i, ev in enumerate(events):
        start = int(ev) + n_pre
        end = start + width
        if start < 0 or end > n_total:
            raise ValueError(f"Event {i} at sample {ev} produces out-of-range epoch [{start}, {end}].")
        out[i] = continuous[:, start:end]
    return out


def baseline_correct(
    X: np.ndarray,
    fs: float,
    baseline: tuple[float | None, float | None] = (None, 0.0),
) -> np.ndarray:
    """Subtract the per-channel mean over the baseline window from each trial.

    ``baseline = (tmin, tmax)`` in seconds relative to epoch start. ``None``
    on either side means "epoch edge". For a default of (None, 0.0) and an
    epoch starting at t=0, this is a no-op; callers epoching with
    ``tmin < 0`` get the conventional pre-stim correction.
    """
    n_samples = X.shape[-1]
    tmin, tmax = baseline
    s_min = 0 if tmin is None else int(round(tmin * fs))
    s_max = n_samples if tmax is None else int(round(tmax * fs))
    s_min = max(0, s_min)
    s_max = min(n_samples, s_max)
    if s_max <= s_min:
        return X.astype(np.float64, copy=True)
    mean = X[..., s_min:s_max].mean(axis=-1, keepdims=True)
    return X - mean
