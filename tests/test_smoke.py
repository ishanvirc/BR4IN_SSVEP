"""Smoke tests — fast, deterministic, no flakiness.

Hard assertions: imports, reference-signal shape, ITR edge cases.
Soft checks: CCA + LOBO-CV run on synthetic data without raising. We
deliberately do **not** assert accuracy thresholds — at SNR=-5 dB the
result swings widely with the noise realization, and a flaky test during
the hackathon is worse than no test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def test_imports():
    import ssvep
    from ssvep import io, preprocessing, features, evaluation, viz, synthetic
    from ssvep.classifiers import CCAClassifier, FBCCAClassifier, TRCAClassifier
    assert ssvep.RANDOM_SEED == 42


def test_cca_reference_shape():
    from ssvep.features import cca_reference_signals
    refs = cca_reference_signals(
        freqs=[7.5, 8.57, 10.0, 12.0],
        n_harmonics=2,
        fs=256,
        n_samples=1024,
    )
    assert refs.shape == (4, 4, 1024)
    # First row should be sin(2π·f·t), starting at 0.
    assert refs[0, 0, 0] == pytest.approx(0.0, abs=1e-12)
    # Second row should be cos(2π·f·t), starting at 1.
    assert refs[0, 1, 0] == pytest.approx(1.0, abs=1e-12)


def test_itr_edge_cases():
    from ssvep.evaluation import itr

    # Perfect accuracy: B = log2(N) bits/trial → log2(4) = 2 bits/trial → 30 bits/min @ 4s.
    assert itr(1.0, 4, 4.0) == pytest.approx(30.0, rel=1e-9)

    # Chance-level accuracy: 0 bits/min.
    assert itr(0.25, 4, 4.0) == 0.0
    # Below chance: clamped to 0.
    assert itr(0.10, 4, 4.0) == 0.0

    # Mid-range hand-computed against Wolpaw formula: P=0.85, N=4, T=4s.
    P, N, T = 0.85, 4, 4.0
    B = math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))
    expected = B * (60.0 / T)
    assert itr(P, N, T) == pytest.approx(expected, rel=1e-9)
    # Should be a sensible positive number (≈ 17.3 bits/min for these inputs).
    assert 0 < itr(P, N, T) < itr(1.0, N, T)


def test_cca_runs_on_synthetic():
    """Pipeline runs end-to-end and returns a sanely-shaped int prediction array."""
    from ssvep.synthetic import make_synthetic_dataset
    from ssvep.classifiers import CCAClassifier

    ds = make_synthetic_dataset(n_trials_per_class=2)  # small: keep tests fast
    clf = CCAClassifier(stim_freqs=ds["stim_freqs"], fs=ds["fs"])
    clf.fit(ds["X"], ds["y"])
    y_pred = clf.predict(ds["X"])

    assert y_pred.shape == (ds["X"].shape[0],)
    assert np.issubdtype(y_pred.dtype, np.integer)
    assert y_pred.min() >= 0
    assert y_pred.max() < ds["stim_freqs"].size


def test_lobo_cv_runs():
    """LOBO-CV runs, returns finite per-block accuracies + a mean."""
    from ssvep.synthetic import make_synthetic_dataset
    from ssvep.classifiers import CCAClassifier
    from ssvep.evaluation import leave_one_block_out_cv

    ds = make_synthetic_dataset(n_trials_per_class=2)
    factory = lambda: CCAClassifier(stim_freqs=ds["stim_freqs"], fs=ds["fs"])
    result = leave_one_block_out_cv(ds["X"], ds["y"], ds["blocks"], factory)

    assert "per_block" in result
    assert "mean" in result
    assert "std" in result
    assert len(result["per_block"]) == np.unique(ds["blocks"]).size
    for acc in result["per_block"].values():
        assert math.isfinite(acc)
        assert 0.0 <= acc <= 1.0
    assert math.isfinite(result["mean"])
