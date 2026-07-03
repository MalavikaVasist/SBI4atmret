"""
Bayes Factor estimation via Learned Classifiers.

Uses the embedding from the trained NPE estimator (frozen) and trains
a lightweight classification head to distinguish N competing models.
"""

from .bayesfactor import (
    BayesFactorClassifier,
    BayesFactorResult,
    prepare_classification_data,
    train_bayes_classifier,
    compute_bayes_factor,
)
