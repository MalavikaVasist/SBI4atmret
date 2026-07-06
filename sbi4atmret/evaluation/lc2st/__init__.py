"""
Local Classifier Two-Sample Test (l-C2ST-NF).

Tests posterior calibration by training classifiers in the flow's
latent space to detect mismatches between q(θ|x) and p(θ|x).
"""

from .lc2st import (
    LC2STEvaluator,
    LC2STResult,
)
