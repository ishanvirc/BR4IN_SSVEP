"""Plotting helpers for SSVEP analyses."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .features import psd as _psd


def plot_psd(
    X: np.ndarray,
    fs: float,
    ch: int | None = None,
    fmin: float = 0.0,
    fmax: float | None = None,
    ax=None,
):
    """Plot Welch PSD averaged over trials. Single channel or all channels.

    Parameters
    ----------
    X    : ndarray, shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
    fs   : sampling rate
    ch   : channel index. If None, plots the trial-averaged mean across channels.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if X.ndim == 2:
        X = X[np.newaxis]
    f, p = _psd(X, fs=fs, fmin=fmin, fmax=fmax)
    p_mean = p.mean(axis=0)  # average over trials → (n_channels, n_freq_bins)
    if ch is None:
        ax.plot(f, p_mean.mean(axis=0), lw=1.5)
        ax.set_title("Trial- and channel-averaged PSD")
    else:
        ax.plot(f, p_mean[ch], lw=1.5)
        ax.set_title(f"Trial-averaged PSD — channel {ch}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(alpha=0.3)
    return ax


def plot_topomap_at_freq(
    X: np.ndarray,
    fs: float,
    ch_names: list[str] | None,
    freq: float,
    ax=None,
):
    """Per-channel power at ``freq``, shown as a bar plot.

    A real topomap requires an MNE montage; until we know the g.tec channel
    layout, we fall back to a labeled bar plot which is informative enough
    for a hackathon. Swap to ``mne.viz.plot_topomap`` once the montage is set.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if X.ndim == 2:
        X = X[np.newaxis]
    f, p = _psd(X, fs=fs)
    idx = int(np.argmin(np.abs(f - freq)))
    power_per_ch = p.mean(axis=0)[:, idx]
    n_ch = power_per_ch.shape[0]
    labels = ch_names if (ch_names and len(ch_names) == n_ch) else [str(i) for i in range(n_ch)]
    ax.bar(range(n_ch), power_per_ch)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(f"Power at {freq:.2f} Hz")
    ax.set_title(f"Per-channel power at {freq:.2f} Hz")
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_spectrogram(x: np.ndarray, fs: float, ax=None):
    """Spectrogram of a single 1-D signal."""
    from scipy.signal import spectrogram

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    f, t, Sxx = spectrogram(x, fs=fs)
    ax.pcolormesh(t, f, 10.0 * np.log10(Sxx + 1e-12), shading="auto")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (dB)")
    return ax


def plot_confusion(cm: np.ndarray, labels: list, ax=None):
    """Confusion-matrix heatmap (seaborn if available, else matplotlib)."""
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    if sns is not None:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax, cbar=False,
        )
    else:
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    return ax
