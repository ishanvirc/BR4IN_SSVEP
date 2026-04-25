"""Loaders for hackathon SSVEP `.mat` recordings.

The exact key names used by g.tec MATLAB exports are not known until the
dataset drops at kickoff. `load_mat` therefore tries a list of candidate
keys for each canonical field. **Edit the candidate lists below once the
real dataset structure is confirmed** — there is one tuple per field at
module top so the change is a one-liner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


_X_KEYS = ("X", "data", "EEG", "trials", "epochs", "signal")
_LABEL_KEYS = ("y", "labels", "Y", "targets", "label", "class", "classes")
_FS_KEYS = ("fs", "Fs", "sfreq", "srate", "sampling_rate", "SamplingRate")
_FREQ_KEYS = (
    "stim_freqs",
    "freqs",
    "frequencies",
    "stimulation_frequencies",
    "f_stim",
    "target_freqs",
)
_CH_KEYS = ("ch_names", "channels", "channel_names", "chanlocs", "electrodes")
_BLOCK_KEYS = ("blocks", "block", "session", "run", "fold")


def _first_present(d: dict, candidates: tuple[str, ...]) -> tuple[str, Any] | None:
    """Return (key, value) for the first candidate key found in d, else None."""
    for k in candidates:
        if k in d:
            return k, d[k]
    return None


def _strip_meta(d: dict) -> dict:
    """Drop scipy.io's ``__header__`` / ``__version__`` / ``__globals__`` entries."""
    return {k: v for k, v in d.items() if not (isinstance(k, str) and k.startswith("__"))}


def _to_3d(arr: np.ndarray) -> np.ndarray:
    """Normalize EEG array to (n_trials, n_channels, n_samples).

    g.tec exports often come as (n_channels, n_samples) for continuous data
    or (n_channels, n_samples, n_trials) for epoched data — the channel
    axis is usually the smallest. We sort axes by size, smallest first,
    then promote 2D inputs to a single trial.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        # (n_channels, n_samples) → single-trial view
        return arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D array for X, got shape {arr.shape}. "
            "Reshape upstream or adjust io.load_mat."
        )
    # Heuristic: channel axis is the smallest; trial axis is one of the two
    # remaining. We assume samples > trials (typical for SSVEP windows >= 1s).
    sizes = arr.shape
    ch_axis = int(np.argmin(sizes))
    other = [i for i in range(3) if i != ch_axis]
    samp_axis = max(other, key=lambda i: sizes[i])
    trial_axis = [i for i in other if i != samp_axis][0]
    return np.transpose(arr, (trial_axis, ch_axis, samp_axis))


def load_mat(path: str | Path) -> dict:
    """Load a g.tec-style SSVEP `.mat` file into a typed dict.

    Returns a dict with these keys (None if the source did not provide it):

        X           : ndarray, shape (n_trials, n_channels, n_samples)
        y           : ndarray, shape (n_trials,), int labels
        fs          : float, sampling rate in Hz
        stim_freqs  : ndarray, shape (n_classes,), candidate frequencies (Hz)
        ch_names    : list[str] | None
        blocks      : ndarray | None, shape (n_trials,) block id for LOBO CV
        raw         : the underlying scipy.io.loadmat dict for fallback access

    Robust to both flat-dict and struct-style `.mat` files via
    ``simplify_cells=True``. If a required field cannot be located, raises
    ``KeyError`` listing the actual top-level keys present in the file —
    edit the ``_X_KEYS``/``_LABEL_KEYS``/etc. tuples at the top of this
    module to teach the loader new key names.
    """
    path = Path(path)
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    flat = _strip_meta(mat)

    # If the file is a single struct under one top-level name, descend into it.
    if len(flat) == 1:
        only_val = next(iter(flat.values()))
        if isinstance(only_val, dict):
            flat = only_val

    found_X = _first_present(flat, _X_KEYS)
    found_y = _first_present(flat, _LABEL_KEYS)
    found_fs = _first_present(flat, _FS_KEYS)
    found_freqs = _first_present(flat, _FREQ_KEYS)

    missing = [
        name for name, found in (
            ("X", found_X), ("y", found_y), ("fs", found_fs), ("stim_freqs", found_freqs)
        ) if found is None
    ]
    if missing:
        raise KeyError(
            f"Could not locate required field(s) {missing} in {path.name}. "
            f"Top-level keys present: {sorted(flat.keys())}. "
            f"Edit candidate-key tuples at the top of src/ssvep/io.py to teach the loader."
        )

    _, X_raw = found_X
    _, y_raw = found_y
    _, fs_raw = found_fs
    _, freqs_raw = found_freqs

    X = _to_3d(np.asarray(X_raw, dtype=np.float64))
    y = np.asarray(y_raw).astype(np.int64).reshape(-1)

    if y.size != X.shape[0]:
        # Sometimes labels come per-channel or per-block — try squeezing first dim.
        y = np.squeeze(np.asarray(y_raw))
        if y.size != X.shape[0]:
            raise ValueError(
                f"Label count ({y.size}) does not match trial count ({X.shape[0]}). "
                f"Adjust io.load_mat to reconcile."
            )

    fs = float(np.asarray(fs_raw).reshape(-1)[0])
    stim_freqs = np.asarray(freqs_raw, dtype=np.float64).reshape(-1)

    found_ch = _first_present(flat, _CH_KEYS)
    ch_names: list[str] | None = None
    if found_ch is not None:
        _, ch_raw = found_ch
        try:
            ch_names = [str(c) for c in np.asarray(ch_raw).reshape(-1)]
        except Exception:
            ch_names = None

    found_blocks = _first_present(flat, _BLOCK_KEYS)
    blocks: np.ndarray | None = None
    if found_blocks is not None:
        _, b_raw = found_blocks
        b = np.asarray(b_raw).reshape(-1)
        if b.size == X.shape[0]:
            blocks = b.astype(np.int64)

    return {
        "X": X,
        "y": y,
        "fs": fs,
        "stim_freqs": stim_freqs,
        "ch_names": ch_names,
        "blocks": blocks,
        "raw": flat,
    }
