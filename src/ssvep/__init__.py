"""br41n-ssvep — hackathon SSVEP analysis pipeline.

Shared random seed for any stochastic operation across the team.
"""

RANDOM_SEED = 42

from . import io, preprocessing, features, evaluation, viz, synthetic, classifiers

__all__ = [
    "RANDOM_SEED",
    "io",
    "preprocessing",
    "features",
    "evaluation",
    "viz",
    "synthetic",
    "classifiers",
]
