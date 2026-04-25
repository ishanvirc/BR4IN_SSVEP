"""Evaluation metrics: accuracy, confusion matrix, ITR, leave-one-block-out CV."""

from __future__ import annotations

from math import log2
from typing import Callable

import numpy as np
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list | None = None,
) -> np.ndarray:
    """Thin wrapper over sklearn.metrics.confusion_matrix."""
    return _sk_confusion_matrix(y_true, y_pred, labels=labels)


def itr(accuracy: float, n_classes: int, window_s: float) -> float:
    """Information Transfer Rate in bits per minute (Wolpaw 1998).

    ::

        if P >= 1:        B = log2(N)
        elif P <= 1/N:    B = 0
        else:             B = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))

        ITR = B * (60 / window_s)

    where ``N = n_classes``, ``P = accuracy``, ``window_s`` is the per-trial
    decision window in seconds (selection time, including any inter-trial gap).
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    P = float(accuracy)
    N = int(n_classes)
    chance = 1.0 / N

    if P >= 1.0:
        bits_per_trial = log2(N)
    elif P <= chance:
        bits_per_trial = 0.0
    else:
        bits_per_trial = (
            log2(N)
            + P * log2(P)
            + (1.0 - P) * log2((1.0 - P) / (N - 1))
        )
    return bits_per_trial * (60.0 / window_s)


def leave_one_block_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    classifier_factory: Callable[[], object],
) -> dict:
    """Leave-one-block-out cross-validation.

    For each unique block id ``b``, train a fresh classifier on trials with
    block != b and evaluate on trials with block == b.

    Parameters
    ----------
    X                  : ndarray, shape (n_trials, n_channels, n_samples)
    y                  : ndarray, shape (n_trials,)
    blocks             : ndarray, shape (n_trials,) integer block ids
    classifier_factory : zero-arg callable returning a fresh classifier with
                         ``fit(X, y)`` / ``score(X, y)`` methods. CCA-style
                         classifiers ignore the labels in ``fit``.

    Returns
    -------
    dict with keys ``per_block`` (dict[block_id -> acc]), ``mean``, ``std``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    blocks = np.asarray(blocks)
    if not (len(X) == len(y) == len(blocks)):
        raise ValueError(
            f"length mismatch: X={len(X)} y={len(y)} blocks={len(blocks)}"
        )

    unique_blocks = np.unique(blocks)
    if unique_blocks.size < 2:
        raise ValueError("Need at least 2 distinct blocks for LOBO CV")

    per_block: dict = {}
    for b in unique_blocks:
        train_mask = blocks != b
        test_mask = blocks == b
        clf = classifier_factory()
        clf.fit(X[train_mask], y[train_mask])
        per_block[int(b)] = float(clf.score(X[test_mask], y[test_mask]))

    accs = np.array(list(per_block.values()), dtype=np.float64)
    return {
        "per_block": per_block,
        "mean": float(accs.mean()),
        "std": float(accs.std()),
    }
