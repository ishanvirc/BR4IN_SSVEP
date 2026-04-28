"""SSVEP classifiers.

- :class:`CCAClassifier` — Lin et al. 2007.
- :class:`FBCCAClassifier` — Chen et al. 2015 filter-bank CCA.
- :class:`TRCAClassifier` — Nakanishi et al. 2018 ensemble TRCA
  (port of meegkit BSD-3; attribution in ``LICENSES/meegkit-BSD-3.txt``).

All three expose the sklearn-style ``fit / predict / score`` interface
on ``(n_trials, n_channels, n_samples)`` arrays, so they are
interchangeable in :func:`ssvep.evaluation.leave_one_block_out_cv`.
For CCA and FBCCA, ``fit`` is a no-op (the reference signals are
data-agnostic). TRCA learns one spatial filter per class.
"""

from .cca import CCAClassifier
from .fbcca import FBCCAClassifier
from .trca import TRCAClassifier

__all__ = ["CCAClassifier", "FBCCAClassifier", "TRCAClassifier"]
