"""
Bayes Factor estimation via Learned Classifiers.

Train a classifier on embedded spectra to distinguish N competing
atmospheric models. At test time, the classifier's softmax output
on the real observation gives model probabilities. Bayes factors
follow from pairwise probability ratios.

The embedding is reused from the trained NPE estimator (self.net.embedding),
so the classifier sees the same representation the posterior network learned.

Supports:
- Binary: 2 models (e.g., cloudfree vs cloudy)
- Multi-class: N >= 2 models (e.g., cloudfree vs cloudy vs patchy)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, List, Dict

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

    # Model names
    model_names: List[str]

    # Softmax probabilities for each model
    probabilities: Dict[str, float]

    # Pairwise log Bayes factors: {("A","B"): log(p_A/p_B)}
    log_bayes_factors: Dict[tuple, float]

    # Interpretation per pair
    interpretations: Dict[tuple, str]

    # Raw logits
    logits: np.ndarray

    # Trained classifier
    classifier: Optional[Any] = None

    # Training history
    history: Optional[dict] = None


# =========================================================
# CLASSIFIER HEAD (sits on top of frozen embedding)
# =========================================================

class BayesFactorClassifier(nn.Module):
    """
    Classification head for model comparison.

    Takes embedded spectra (from self.net.embedding) and classifies
    them into N model classes.

    Architecture: Frozen Embedding → Trainable MLP Head → N logits

    Args:
        embedding: the embedding module from the trained NPE (frozen)
        embedding_dim: output dimension of the embedding
        n_classes: number of competing models (>= 2)
        head_hidden: list of hidden layer sizes for the classification head
        dropout: dropout rate
    """

    def __init__(
        self,
        embedding: nn.Module,
        embedding_dim: int,
        n_classes: int = 2,
        head_hidden: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if head_hidden is None:
            head_hidden = [128, 64]

        self.embedding = embedding
        self.n_classes = n_classes

        # Freeze the embedding — we don't train it
        for param in self.embedding.parameters():
            param.requires_grad = False

        # Build classification head
        layers = []
        in_dim = embedding_dim
        for h_dim in head_hidden:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, D_obs) raw spectra (same format as NPE training input)

        Returns:
            logits: (B, n_classes)
        """
        with torch.no_grad():
            emb = self.embedding(x)
        return self.head(emb)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_classification_data(
    spectra_dict: Dict[str, Tensor],
    shuffle: bool = True,
) -> tuple:
    """
    Prepare labeled data for multi-class classification.

    Args:
        spectra_dict: {model_name: (N, D) spectra tensor}
                      e.g., {"cloudfree": tensor, "cloudy": tensor, "patchy": tensor}
        shuffle: whether to shuffle

    Returns:
        (x, labels, model_names)
            x: (total_N, D) concatenated spectra
            labels: (total_N,) integer class labels
            model_names: list of model names in label order
    """
    model_names = list(spectra_dict.keys())
    all_x = []
    all_labels = []

    for class_idx, name in enumerate(model_names):
        spectra = spectra_dict[name]
        all_x.append(spectra)
        all_labels.append(torch.full((len(spectra),), class_idx, dtype=torch.long))

    x = torch.cat(all_x, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if shuffle:
        perm = torch.randperm(len(x))
        x = x[perm]
        labels = labels[perm]

    return x, labels, model_names


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

    Uses CrossEntropyLoss (works for both binary and multi-class).

    Args:
        classifier: BayesFactorClassifier instance
        train_x, train_labels: training data
        val_x, val_labels: validation data
        n_epochs: max epochs
        batch_size: batch size
        lr: initial learning rate
        weight_decay: AdamW weight decay
        patience: LR scheduler patience
        min_lr: minimum LR (triggers early stopping)
        device: computation device
        save_path: path to save best model

    Returns:
        dict with train_loss, val_loss, val_accuracy per epoch
    """
    classifier = classifier.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Only optimize the head (embedding is frozen)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=patience, min_lr=min_lr,
    )

    train_ds = TensorDataset(train_x, train_labels)
    val_ds = TensorDataset(val_x, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")

    for epoch in tqdm(range(n_epochs), desc="Training BF classifier"):
        # Train
        classifier.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = classifier(x_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.head.parameters(), 1.0)
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
                preds = logits.argmax(dim=-1)
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

        # Early stop on min LR
        if optimizer.param_groups[0]["lr"] <= min_lr:
            break

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | Acc: {val_acc:.3f}")

    if save_path is not None and save_path.exists():
        classifier.load_state_dict(torch.load(save_path))

    return history


# =========================================================
# INFERENCE: COMPUTE BAYES FACTOR
# =========================================================

def compute_bayes_factor(
    classifier: BayesFactorClassifier,
    x_obs: Tensor,
    model_names: List[str],
    device: str = "cuda",
) -> BayesFactorResult:
    """
    Compute Bayes factors for the real observation.

    Args:
        classifier: trained BayesFactorClassifier
        x_obs: (1, D) observation tensor
        model_names: list of model names corresponding to class indices
        device: computation device

    Returns:
        BayesFactorResult with probabilities and pairwise log Bayes factors
    """
    classifier = classifier.to(device).eval()

    with torch.no_grad():
        logits = classifier(x_obs.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

    logits_np = logits.cpu().numpy().flatten()

    # Probabilities per model
    probabilities = {name: float(probs[i]) for i, name in enumerate(model_names)}

    # Pairwise log Bayes factors
    log_bfs = {}
    interpretations = {}

    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i >= j:
                continue

            # log BF(i vs j) = log(p_i / p_j)
            if probs[j] > 0 and probs[i] > 0:
                log_bf = float(np.log(probs[i]) - np.log(probs[j]))
            else:
                log_bf = float("inf") if probs[i] > probs[j] else float("-inf")

            log10_bf = log_bf / np.log(10)
            abs_log10 = abs(log10_bf)

            if abs_log10 < 0.5:
                strength = "Inconclusive"
            elif abs_log10 < 1.0:
                strength = "Substantial"
            elif abs_log10 < 1.5:
                strength = "Strong"
            elif abs_log10 < 2.0:
                strength = "Very strong"
            else:
                strength = "Decisive"

            favored = name_i if log_bf > 0 else name_j
            interp = f"{strength} evidence for {favored} (log10 BF = {log10_bf:.2f})"

            log_bfs[(name_i, name_j)] = log_bf
            interpretations[(name_i, name_j)] = interp

    return BayesFactorResult(
        model_names=model_names,
        probabilities=probabilities,
        log_bayes_factors=log_bfs,
        interpretations=interpretations,
        logits=logits_np,
        classifier=classifier,
    )
