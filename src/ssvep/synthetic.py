"""Synthetic SSVEP dataset generator.

Mirrors the dict shape returned by :func:`ssvep.io.load_mat` so the rest of
the pipeline is unaware which source produced the data. Used by
``scripts/run_baseline.py --synthetic`` and the smoke test suite to
exercise the full chain before real data arrives.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_FREQS = (7.5, 8.57, 10.0, 12.0)


def make_synthetic_dataset(
    stim_freqs: Iterable[float] = DEFAULT_FREQS,
    n_channels: int = 8,
    fs: float = 256.0,
    n_samples: int = 1024,
    n_trials_per_class: int = 10,
    n_blocks: int = 4,
    n_harmonics: int = 2,
    snr_db: float = -5.0,
    seed: int = 42,
) -> dict:
    """Generate a fake SSVEP dataset matching the ``load_mat`` dict schema.

    For each trial of class k, the signal is a sum over harmonics of
    ``cos(2π · h · f_k · t)`` with random per-channel amplitude (uniform
    [0.5, 1.5]) and phase (uniform [0, 2π)). White Gaussian noise is added
    at the requested SNR (target signal RMS vs. noise RMS).

    Block ids are interleaved trial-by-trial so leave-one-block-out CV
    folds receive a balanced class distribution.

    Returns
    -------
    dict with keys: ``X, y, fs, stim_freqs, ch_names, blocks, raw``
    (matches :func:`ssvep.io.load_mat`).
    """
    rng = np.random.default_rng(seed)
    stim_freqs_arr = np.asarray(list(stim_freqs), dtype=np.float64)
    n_classes = stim_freqs_arr.size
    n_trials = n_classes * n_trials_per_class

    t = np.arange(n_samples, dtype=np.float64) / float(fs)

    X = np.empty((n_trials, n_channels, n_samples), dtype=np.float64)
    y = np.empty(n_trials, dtype=np.int64)
    blocks = np.empty(n_trials, dtype=np.int64)

    snr_linear = 10.0 ** (snr_db / 10.0)

    trial_idx = 0
    for class_id, f in enumerate(stim_freqs_arr):
        for rep in range(n_trials_per_class):
            amps = rng.uniform(0.5, 1.5, size=(n_channels, n_harmonics))
            phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_channels, n_harmonics))
            sig = np.zeros((n_channels, n_samples), dtype=np.float64)
            for h in range(1, n_harmonics + 1):
                a = amps[:, h - 1][:, None]
                p = phases[:, h - 1][:, None]
                sig += a * np.cos(2.0 * np.pi * h * f * t[None, :] + p)
            sig_rms = np.sqrt(np.mean(sig ** 2))
            noise_rms = sig_rms / np.sqrt(snr_linear) if snr_linear > 0 else sig_rms
            noise = rng.normal(0.0, noise_rms, size=(n_channels, n_samples))
            X[trial_idx] = sig + noise
            y[trial_idx] = class_id
            blocks[trial_idx] = rep % n_blocks
            trial_idx += 1

    # Interleave so blocks see all classes (balanced LOBO folds).
    order = np.argsort(blocks, kind="stable")
    X = X[order]
    y = y[order]
    blocks = blocks[order]

    ch_names = [f"CH{i+1}" for i in range(n_channels)]

    return {
        "X": X,
        "y": y,
        "fs": float(fs),
        "stim_freqs": stim_freqs_arr,
        "ch_names": ch_names,
        "blocks": blocks,
        "raw": {"source": "synthetic", "snr_db": snr_db, "seed": seed},
    }
