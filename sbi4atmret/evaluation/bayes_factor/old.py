"""
Bayes Factor estimation via Learned Binary Classifiers.

Train a neural classifier to distinguish spectra generated under two
competing atmospheric models (e.g., cloudfree vs patchy). At test time,
the classifier's log-odds on the real observation approximates the
log Bayes factor:

    log B_12 ≈ logit(p(M1 | x_obs)) = log p(M1|x) - log p(M2|x)

This is the "Likelihood-Free Model Comparison" approach.

Workflow:
    1. Generate training data from both models (with noise applied).
    2. Label: model_A → 0, model_B → 1.
    3. Train a binary classifier (embedding + head).
    4. Evaluate on the real observation.
    5. The sigmoid output gives the probability of each model.

References:
    - Hermans et al. (2020): Likelihood-free MCMC with amortized ratio estimators
    - Vasist et al. (2023): Classifier-based Bayes factors for exoplanet atmospheres
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass(frozen=True)
class BayesFactorResult:
    """Container for classifier-based Bayes factor results."""

    # Raw logit output on observation
    logit: float

    # Probability of model A given observation
    prob_model_a: float

    # Probability of model B given observation
    prob_model_b: float

    # Log Bayes factor: log(p(M_A|x) / p(M_B|x))
    log_bayes_factor: float

    # Interpretation on Jeffreys' scale
    interpretation: str

    # Trained classifier model
    classifier: Optional[Any] = None

    # Training history
    history: Optional[dict] = None


# =========================================================
# CLASSIFIER MODEL
# =========================================================

class BayesFactorClassifier(nn.Module):
    """
    Binary classifier for model comparison.

    Architecture: SoftClip → Embedding (ResMLP) → Classification Head

    Outputs a single logit: positive → favors model_B, negative → favors model_A.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_features: List[int] = None,
        head_type: str = "mlp",
        head_depth: int = 4,
        head_factor: int = 16,
        softclip_bound: float = 100.0,
    ):
        super().__init__()

        if hidden_features is None:
            hidden_features = [512] * 3 + [256] * 5 + [128] * 7

        from lampe.nn import ResMLP

        self.embedding = nn.Sequential(
            SoftClip(softclip_bound),
            ResMLP(
                input_dim,
                embedding_dim,
                hidden_features=hidden_features,
                activation=nn.ELU,
            ),
        )

        # Classification head
        if head_type == "mlp":
            self.head = self._build_mlp_head(embedding_dim, head_depth, head_factor)
        else:
            self.head = nn.Linear(embedding_dim, 1)

    def _build_mlp_head(self, input_dim, depth, factor):
        layers = []
        dim = input_dim
        for _ in range(depth):
            next_dim = max(dim // 2, factor)
            layers.extend([nn.Linear(dim, next_dim), nn.ELU()])
            dim = next_dim
        layers.append(nn.Linear(dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, input_dim) spectra

        Returns:
            logits: (B, 1)
        """
        emb = self.embedding(x)
        return self.head(emb)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return probability of model B."""
        return torch.sigmoid(self.forward(x))


class SoftClip(nn.Module):
    """Soft clipping activation."""

    def __init__(self, bound: float = 100.0):
        super().__init__()
        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        return self.bound * torch.tanh(x / self.bound)


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_classification_data(
    spectra_a: Tensor,
    spectra_b: Tensor,
    shuffle: bool = True,
) -> tuple:
    """
    Prepare labeled data for binary classification.

    Args:
        spectra_a: (N, D) spectra from model A (label=0)
        spectra_b: (N, D) spectra from model B (label=1)
        shuffle: whether to shuffle

    Returns:
        (x, labels): x is (2N, D), labels is (2N, 1)
    """
    labels_a = torch.zeros(len(spectra_a), 1)
    labels_b = torch.ones(len(spectra_b), 1)

    x = torch.cat([spectra_a, spectra_b], dim=0)
    labels = torch.cat([labels_a, labels_b], dim=0)

    if shuffle:
        perm = torch.randperm(len(x))
        x = x[perm]
        labels = labels[perm]

    return x, labels


# =========================================================
# TRAINING
# =========================================================

def train_bayes_classifier(
    classifier: BayesFactorClassifier,
    train_x: Tensor,
    train_labels: Tensor,
    val_x: Tensor,
    val_labels: Tensor,
    n_epochs: int = 1024,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 32,
    min_lr: float = 1e-9,
    device: str = "cuda",
    save_path: Optional[Path] = None,
) -> dict:
    """
    Train the Bayes factor classifier.

    Args:
        classifier: BayesFactorClassifier instance
        train_x, train_labels: training data
        val_x, val_labels: validation data
        n_epochs: max epochs
        batch_size: batch size
        lr: initial learning rate
        weight_decay: weight decay
        patience: LR scheduler patience
        min_lr: minimum LR (early stopping trigger)
        device: computation device
        save_path: path to save best model

    Returns:
        dict with training history
    """
    classifier = classifier.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=patience, min_lr=min_lr
    )

    train_ds = TensorDataset(train_x, train_labels)
    val_ds = TensorDataset(val_x, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")

    for epoch in tqdm(range(n_epochs), desc="Training classifier"):
        # Train
        classifier.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = classifier(x_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        classifier.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = classifier(x_batch)
                loss = loss_fn(logits, y_batch)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path is not None:
                torch.save(classifier.state_dict(), save_path)

        # Early stopping on min LR
        if optimizer.param_groups[0]["lr"] <= min_lr:
            break

    if save_path is not None and save_path.exists():
        classifier.load_state_dict(torch.load(save_path))

    return history


# =========================================================
# INFERENCE: COMPUTE BAYES FACTOR
# =========================================================

def compute_bayes_factor(
    classifier: BayesFactorClassifier,
    x_obs: Tensor,
    device: str = "cuda",
) -> BayesFactorResult:
    """
    Compute the Bayes factor for a real observation.

    The classifier's logit output directly approximates log(p(M_B|x) / p(M_A|x)).

    Args:
        classifier: trained BayesFactorClassifier
        x_obs: (1, D) observation tensor
        device: computation device

    Returns:
        BayesFactorResult
    """
    classifier = classifier.to(device).eval()

    with torch.no_grad():
        logit = classifier(x_obs.to(device)).item()

    prob_b = torch.sigmoid(torch.tensor(logit)).item()
    prob_a = 1.0 - prob_b

    # log BF: positive → favors B, negative → favors A
    # We define BF as A/B convention, so negate
    log_bf = -logit  # log(p(A|x) / p(B|x))

    # Jeffreys' scale (in natural log, convert to log10 for standard scale)
    log10_bf = log_bf / np.log(10)
    abs_log10 = abs(log10_bf)

    if abs_log10 < 0.5:
        interpretation = "Not worth more than a bare mention"
    elif abs_log10 < 1.0:
        interpretation = "Substantial"
    elif abs_log10 < 1.5:
        interpretation = "Strong"
    elif abs_log10 < 2.0:
        interpretation = "Very strong"
    else:
        interpretation = "Decisive"

    favored = "Model A" if log_bf > 0 else "Model B"
    interpretation = f"{interpretation} evidence for {favored} (log10 BF = {log10_bf:.2f})"

    return BayesFactorResult(
        logit=logit,
        prob_model_a=prob_a,
        prob_model_b=prob_b,
        log_bayes_factor=log_bf,
        interpretation=interpretation,
        classifier=classifier,
    )
