"""
Log-Likelihood Ratio (LLR) test for nested model comparison.

Computes the LLR between two competing atmospheric models by finding
the MAP of posterior/prior ratio for each model and taking the log-ratio.

Can also build a null distribution of T-statistics by simulating
observations under the simpler model and computing LLR for each.
"""

from .llr import (
    LLREvaluator,
    LLRResult,
)
