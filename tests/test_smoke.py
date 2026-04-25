"""Smoke tests — fast, deterministic, no flakiness.

Hard assertions: imports, reference-signal shape, ITR edge cases.
Soft checks: CCA + LOBO-CV run on synthetic data without raising. We
deliberately do **not** assert accuracy thresholds — at SNR=-5 dB the
result swings widely with the noise realization, and a flaky test during
the hackathon is worse than no test.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_SAMPLE_FILE = _DATA_DIR / "subject_1_fvep_led_training_1.mat"


@pytest.fixture
def real_data_available():
    if not _SAMPLE_FILE.exists():
        pytest.skip(f"{_SAMPLE_FILE} not present; skipping real-data tests")


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


# --- Real-data tests (skipped on machines without data/raw/) ---------------


def test_load_mat_real_file(real_data_available):
    """load_mat handles the continuous (11, N) hackathon schema."""
    from ssvep.io import load_mat

    ds = load_mat(_SAMPLE_FILE)
    assert ds["X"].ndim == 3 and ds["X"].shape[1] == 8
    assert ds["X"].shape[0] == ds["y"].size == ds["ch11_pred"].size
    assert ds["y"].dtype == np.int64
    assert set(np.unique(ds["y"]).tolist()) <= {0, 1, 2, 3}
    assert ds["fs"] == 256.0
    assert ds["ch11_pred"].dtype == np.int64
    # CH11 sentinel allowed.
    assert ds["ch11_pred"].min() >= -1
    assert ds["ch11_pred"].max() <= 3


def test_load_all_blocks(real_data_available):
    """load_all() concatenates the 4 standard files with per-file block IDs."""
    from ssvep.io import load_all

    ds = load_all()
    assert ds["X"].shape[0] == ds["y"].size == 80
    assert set(np.unique(ds["blocks"]).tolist()) == {0, 1, 2, 3}


def test_ch11_per_trial_accuracy_matches_notebook(real_data_available):
    """Per-trial CH11 accuracy reproduces the notebook's 0.872 finding."""
    from ssvep.io import load_all

    ds = load_all()
    fired = ds["ch11_pred"] >= 0
    if fired.sum() == 0:
        pytest.skip("CH11 never fired in any trial window; nothing to score")
    acc = float((ds["ch11_pred"][fired] == ds["y"][fired]).mean())
    assert 0.80 <= acc <= 0.95, (
        f"per-trial CH11 accuracy {acc:.3f} outside expected band [0.80, 0.95]"
    )
