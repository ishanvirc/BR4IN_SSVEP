"""SSVEP classifiers: CCA (implemented), FBCCA + TRCA (stubs)."""

from .cca import CCAClassifier
from .fbcca import FBCCAClassifier
from .trca import TRCAClassifier

__all__ = ["CCAClassifier", "FBCCAClassifier", "TRCAClassifier"]
