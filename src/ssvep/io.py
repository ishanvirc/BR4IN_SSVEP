"""Loaders for hackathon SSVEP `.mat` recordings.

The hackathon files are **continuous** g.tec recordings: each `.mat`
contains a single ``y`` key holding a ``(11, n_samples)`` float64 matrix
plus a scalar ``fs`` key. Channels are layered as:

    CH1   (idx 0): sample time (seconds since recording start)
    CH2-9 (idx 1-8): EEG — PO7, PO3, POz, PO4, PO8, O1, Oz, O2
    CH10  (idx 9):  trigger — 0 when stim off, else stim freq in Hz
    CH11  (idx 10): g.tec live LDA classifier output (descending mapping;
                    see ``CH11_DESCENDING_MAP``)

`load_mat` epochs a single file via CH10 transitions; `load_all` does the
same across every matching file in ``data/raw/`` and assigns block IDs
per file for cross-file LOBO-CV.

The previous epoched-with-labels schema is preserved as
:func:`load_mat_legacy` (deprecated) for any caller that still needs it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


STIM_FREQS: tuple[float, ...] = (9.0, 10.0, 12.0, 15.0)
CH_NAMES: tuple[str, ...] = (
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
)
CH11_DESCENDING_MAP: dict[float, float] = {1.0: 15.0, 2.0: 12.0, 3.0: 10.0, 4.0: 9.0}
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_GLOB = "subject_*_fvep_led_training_*.mat"

# Channel indices (0-based) into the (11, n_samples) continuous matrix.
_EEG_SLICE = slice(1, 9)   # CH2..CH9
_TRIGGER_IDX = 9            # CH10
_LDA_IDX = 10               # CH11


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_continuous(path: str | Path) -> dict:
    """Load a single hackathon `.mat` file as a raw continuous matrix.

    No filtering, no epoching. Useful when you want full control over the
    pipeline (e.g. composing your own filter step, or analysing the
    continuous signal directly).

    Returns
    -------
    dict with keys:
        continuous : (11, N) float64 — raw matrix from the .mat file
        fs         : float — sampling rate
        source     : str — filename (no path)
    """
    path = Path(path)
    cont = _load_continuous(path)
    fs = _read_fs(path)
    return {"continuous": cont, "fs": fs, "source": path.name}


def load_continuous_all(
    directory: str | Path = DEFAULT_DATA_DIR,
    glob: str = DEFAULT_GLOB,
) -> list[dict]:
    """Load every matching `.mat` file as raw continuous dicts (sorted).

    Returns a list of per-file dicts in sorted order. Each dict has the
    same shape as :func:`load_continuous`'s return.

    Raises
    ------
    FileNotFoundError
        If no files match ``glob`` under ``directory``.
    """
    directory = Path(directory)
    files = sorted(directory.glob(glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob}' under {directory.resolve()}"
        )
    return [load_continuous(p) for p in files]


def epoch_trials(
    continuous: np.ndarray,
    fs: float,
    *,
    window_s: float = 3.0,
    latency_s: float = 0.14,
    ch11_window_samples: int = 100,
    source: str = "",
    block_id: int = 0,
) -> dict:
    """Epoch a (11, N) continuous matrix into per-trial EEG + labels + CH11 preds.

    Operates on whatever matrix is passed — filtered or not, this function
    doesn't care. Returns the canonical dataset dict shape (same as
    :func:`load_mat`).

    Parameters
    ----------
    continuous : ndarray, shape (11, N)
    fs : float
    window_s : float, default 3.0
        Length of per-trial analysis window (s).
    latency_s : float, default 0.14
        Offset from stim onset to window start (Chen 2015 convention).
    ch11_window_samples : int, default 100
        Last-N-samples window over which CH11 is majority-voted.
    source : str
        Filename to record under ``raw['source']``. Defaults to "".
    block_id : int
        Value to fill the ``blocks`` array with. Defaults to 0;
        :func:`load_all` overrides per file.
    """
    if continuous.ndim != 2 or continuous.shape[0] != 11:
        raise ValueError(
            f"epoch_trials expects a (11, N) matrix; got {continuous.shape}."
        )
    trigger = continuous[_TRIGGER_IDX]
    onsets_offsets = _find_trial_onsets_and_offsets(trigger)
    if not onsets_offsets:
        raise ValueError(
            f"No trials found{f' in {source}' if source else ''}: "
            "CH10 has no nonzero transitions."
        )

    X, kept = _epoch_eeg(continuous, onsets_offsets, fs, window_s, latency_s)
    y = np.array(
        [STIM_FREQS.index(float(onsets_offsets[i][2])) for i in kept],
        dtype=np.int64,
    )
    ch11_pred = _ch11_predictions(
        continuous,
        [onsets_offsets[i] for i in kept],
        ch11_window_samples,
    )
    blocks = np.full(len(kept), block_id, dtype=np.int64)
    return {
        "X": X,
        "y": y,
        "fs": float(fs),
        "stim_freqs": np.array(STIM_FREQS, dtype=np.float64),
        "ch_names": list(CH_NAMES),
        "blocks": blocks,
        "ch11_pred": ch11_pred,
        "raw": {"continuous": continuous, "source": source},
    }


def load_mat(
    path: str | Path,
    window_s: float = 3.0,
    latency_s: float = 0.14,
    ch11_window_samples: int = 100,
    *,
    l_freq: float | None = 3.0,
    h_freq: float | None = 45.0,
    notch_hz: float | None = 50.0,
    filter: bool = True,
) -> dict:
    """Load a g.tec continuous SSVEP `.mat` file and return canonical-preprocessed epochs.

    By default returns 3-45 Hz bandpass + 50 Hz notch filtering applied
    to the continuous signal **before** epoching, per VT's scan in
    `notebooks/02_preprocessing.ipynb`. Continuous-first filtering avoids
    per-epoch FIR edge artifacts at the start and end of every trial.

    Pass ``filter=False`` for raw epoched output — useful as a sanity
    baseline or for debugging, not for analysis.

    Parameters
    ----------
    path : str | Path
        Path to a `.mat` file with ``{'fs': int, 'y': (11, N) float}``.
    window_s : float, default 3.0
    latency_s : float, default 0.14
    ch11_window_samples : int, default 100
    l_freq, h_freq : float | None, default (3.0, 45.0)
        Bandpass edges (Hz). Ignored when ``filter=False``.
    notch_hz : float | None, default 50.0
        Notch frequency (Hz). Ignored when ``filter=False``.
    filter : bool, default True
        Master switch — short-circuits all 3 filter params when False.

    Returns
    -------
    dict with keys:
        X           : (n_trials, 8, round(window_s * fs)) float64 — EEG
        y           : (n_trials,) int64 — class index 0..3 into ``STIM_FREQS``
        fs          : float
        stim_freqs  : (4,) float64 — np.array(STIM_FREQS)
        ch_names    : list[str] — list(CH_NAMES)
        blocks      : (n_trials,) int64 — all zeros for a single-file load
        ch11_pred   : (n_trials,) int64 — class index 0..3, or -1 if CH11
                      did not fire in the trial-end window
        raw         : {'continuous': (11, N) ndarray (post-filter if applied),
                       'source': filename}
    """
    ds = load_continuous(path)
    cont = ds["continuous"]
    if filter:
        from .preprocessing import filter_continuous
        cont = filter_continuous(
            cont, ds["fs"], l_freq=l_freq, h_freq=h_freq, notch_hz=notch_hz
        )
    return epoch_trials(
        cont,
        ds["fs"],
        window_s=window_s,
        latency_s=latency_s,
        ch11_window_samples=ch11_window_samples,
        source=ds["source"],
        block_id=0,
    )


def load_all(
    directory: str | Path = DEFAULT_DATA_DIR,
    window_s: float = 3.0,
    latency_s: float = 0.14,
    ch11_window_samples: int = 100,
    glob: str = DEFAULT_GLOB,
    *,
    l_freq: float | None = 3.0,
    h_freq: float | None = 45.0,
    notch_hz: float | None = 50.0,
    filter: bool = True,
) -> dict:
    """Load every matching `.mat` file under ``directory`` and concatenate.

    Defaults match :func:`load_mat`: continuous-first canonical
    preprocessing applied per file. Pass ``filter=False`` for raw
    epoched output.

    Block IDs are assigned per file (sorted) so leave-one-block-out CV
    folds across files.

    Raises
    ------
    FileNotFoundError
        If no files match ``glob`` under ``directory``.
    """
    directory = Path(directory)
    files = sorted(directory.glob(glob))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob}' under {directory.resolve()}"
        )

    Xs, ys, blocks_list, ch11s = [], [], [], []
    sources: list[str] = []
    continuous_per_file: list[np.ndarray] = []
    fs_seen: float | None = None

    for block_id, path in enumerate(files):
        ds = load_mat(
            path,
            window_s=window_s,
            latency_s=latency_s,
            ch11_window_samples=ch11_window_samples,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_hz=notch_hz,
            filter=filter,
        )
        if fs_seen is None:
            fs_seen = ds["fs"]
        elif ds["fs"] != fs_seen:
            raise ValueError(
                f"Sample rate mismatch: {path.name} fs={ds['fs']} vs prior fs={fs_seen}"
            )

        Xs.append(ds["X"])
        ys.append(ds["y"])
        ch11s.append(ds["ch11_pred"])
        # Override block_id (load_mat returns 0); use file index here.
        blocks_list.append(np.full(ds["y"].size, block_id, dtype=np.int64))
        sources.append(path.name)
        continuous_per_file.append(ds["raw"]["continuous"])

    return {
        "X": np.concatenate(Xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "fs": float(fs_seen),
        "stim_freqs": np.array(STIM_FREQS, dtype=np.float64),
        "ch_names": list(CH_NAMES),
        "blocks": np.concatenate(blocks_list, axis=0),
        "ch11_pred": np.concatenate(ch11s, axis=0),
        "raw": {"sources": sources, "continuous_per_file": continuous_per_file},
    }


def load_mat_legacy(path: str | Path) -> dict:
    """Deprecated: the pre-hackathon loader for epoched-with-labels `.mat` files.

    Use :func:`load_mat` for the actual hackathon files (continuous, no
    label key). This function is preserved only for callers that depended
    on the old key-search behavior.
    """
    warnings.warn(
        "load_mat_legacy is deprecated; use load_mat for the hackathon "
        "continuous schema or call scipy.io.loadmat directly for arbitrary files.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_mat_legacy_impl(path)


# ---------------------------------------------------------------------------
# Internal helpers (continuous-schema path)
# ---------------------------------------------------------------------------


def _load_continuous(path: Path) -> np.ndarray:
    """Load the (11, N) continuous matrix from a hackathon `.mat` file."""
    mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
    flat = _strip_meta(mat)
    if "y" not in flat:
        raise KeyError(
            f"{path.name}: expected key 'y' holding (11, N) continuous data; "
            f"found {sorted(flat.keys())}. If this is a different schema, "
            f"see load_mat_legacy."
        )
    arr = np.asarray(flat["y"], dtype=np.float64)
    if arr.ndim != 2 or 11 not in arr.shape:
        raise ValueError(
            f"{path.name}: expected 2D array with one axis of length 11; got shape {arr.shape}."
        )
    if arr.shape[0] != 11:
        arr = arr.T
    return arr


def _read_fs(path: Path) -> float:
    mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
    flat = _strip_meta(mat)
    if "fs" not in flat:
        raise KeyError(f"{path.name}: missing 'fs' key.")
    return float(np.asarray(flat["fs"]).reshape(-1)[0])


def _find_trial_onsets_and_offsets(
    trigger: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Return ``[(onset_idx, offset_idx_exclusive, freq_at_onset), ...]``.

    Uses the same trick as ``trials_from_trigger`` in notebook 01: diff
    the (trigger != 0) mask with prepend/append 0, locate ±1 transitions.
    """
    diff = np.diff((trigger != 0).astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [
        (int(s), int(e), float(trigger[s])) for s, e in zip(starts, ends)
    ]


def _epoch_eeg(
    continuous: np.ndarray,
    onsets_offsets: list[tuple[int, int, float]],
    fs: float,
    window_s: float,
    latency_s: float,
) -> tuple[np.ndarray, list[int]]:
    """Slice EEG into trials. Returns (X, kept_indices).

    ``kept_indices`` indexes back into ``onsets_offsets`` so labels and
    CH11 predictions stay aligned with the trials we actually emitted.
    """
    n_window = int(round(window_s * fs))
    latency_n = int(round(latency_s * fs))
    epochs: list[np.ndarray] = []
    kept: list[int] = []
    for i, (onset, offset, _freq) in enumerate(onsets_offsets):
        start = onset + latency_n
        end = start + n_window
        if end > offset:
            # Trial is shorter than (latency + window) — skip rather than pad.
            continue
        epochs.append(continuous[_EEG_SLICE, start:end])
        kept.append(i)
    if not epochs:
        return (
            np.zeros((0, _EEG_SLICE.stop - _EEG_SLICE.start, n_window), dtype=np.float64),
            [],
        )
    return np.stack(epochs, axis=0).astype(np.float64, copy=False), kept


def _ch11_predictions(
    continuous: np.ndarray,
    onsets_offsets: list[tuple[int, int, float]],
    ch11_window_samples: int,
) -> np.ndarray:
    """Per-trial CH11 majority-vote prediction in ``[0..3]`` or ``-1``."""
    lda = continuous[_LDA_IDX]
    out = np.full(len(onsets_offsets), -1, dtype=np.int64)
    for i, (onset, offset, _freq) in enumerate(onsets_offsets):
        trial_len = offset - onset
        w = min(ch11_window_samples, trial_len)
        if w <= 0:
            continue
        window = lda[offset - w:offset]
        nonzero = window[window != 0]
        if nonzero.size == 0:
            continue
        vals, counts = np.unique(nonzero, return_counts=True)
        majority_class = float(vals[np.argmax(counts)])
        freq = CH11_DESCENDING_MAP.get(majority_class, -1.0)
        if freq < 0:
            continue
        try:
            out[i] = STIM_FREQS.index(float(freq))
        except ValueError:
            # CH11 produced a value we can't map; leave as -1.
            continue
    return out


# ---------------------------------------------------------------------------
# Legacy (epoched-with-labels) loader — preserved for back-compat
# ---------------------------------------------------------------------------


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
    for k in candidates:
        if k in d:
            return k, d[k]
    return None


def _strip_meta(d: dict) -> dict:
    return {k: v for k, v in d.items() if not (isinstance(k, str) and k.startswith("__"))}


def _to_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D array for X, got shape {arr.shape}. "
            "Reshape upstream or adjust io.load_mat_legacy."
        )
    sizes = arr.shape
    ch_axis = int(np.argmin(sizes))
    other = [i for i in range(3) if i != ch_axis]
    samp_axis = max(other, key=lambda i: sizes[i])
    trial_axis = [i for i in other if i != samp_axis][0]
    return np.transpose(arr, (trial_axis, ch_axis, samp_axis))


def _load_mat_legacy_impl(path: str | Path) -> dict:
    path = Path(path)
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    flat = _strip_meta(mat)

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
            f"Edit candidate-key tuples in src/ssvep/io.py to teach the legacy loader."
        )

    _, X_raw = found_X
    _, y_raw = found_y
    _, fs_raw = found_fs
    _, freqs_raw = found_freqs

    X = _to_3d(np.asarray(X_raw, dtype=np.float64))
    y = np.asarray(y_raw).astype(np.int64).reshape(-1)

    if y.size != X.shape[0]:
        y = np.squeeze(np.asarray(y_raw))
        if y.size != X.shape[0]:
            raise ValueError(
                f"Label count ({y.size}) does not match trial count ({X.shape[0]}). "
                f"Adjust io.load_mat_legacy to reconcile."
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
