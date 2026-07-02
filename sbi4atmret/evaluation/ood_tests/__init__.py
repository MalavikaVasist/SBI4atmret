"""
OOD Detection Tests.

Investigates whether an observation is outside the training distribution using:
- Posterior variability (D_v): disagreement between independently trained models
- Embedding analysis: Mahalanobis distance, k-NN, PCA in embedding space
- Meta-learner uncertainty (optional): predicted std as indirect OOD signal
"""

from .posterior_variability import (
    pairwise_kl_divergence,
    pairwise_kl_per_param,
    posterior_variability_score,
    compute_ood_score,
    plot_variability,
    VariabilityResult,
)

from .embedding_analysis import (
    extract_embeddings,
    extract_obs_embedding,
    compute_pca,
    mahalanobis_distance,
    knn_distance,
    analyze_embeddings,
    plot_embedding_pca,
    EmbeddingAnalysisResult,
)
