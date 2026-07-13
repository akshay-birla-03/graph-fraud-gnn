"""Full-batch (transductive) training for node classification.

The same routine trains both the GCN and the MLP: full-batch forward passes,
a class-weighted cross-entropy computed only over the training-mask nodes, and
early model selection on validation ROC-AUC.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .data import GraphData
from .evaluate import Metrics, evaluate_model, fraud_probabilities
from .layers import normalize_adjacency


def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and torch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict] = field(default_factory=list)
    val_metrics: Metrics | None = None
    test_metrics: Metrics | None = None


def class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights computed on the training nodes."""
    y_tr = y[mask]
    counts = torch.bincount(y_tr, minlength=2).float()
    counts = torch.clamp(counts, min=1.0)
    inv = counts.sum() / (2.0 * counts)
    return inv


def train_node_classifier(
    model: nn.Module,
    data: GraphData,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 0,
    threshold: float = 0.5,
    verbose: bool = False,
) -> TrainResult:
    """Train ``model`` on ``data`` with masked, class-weighted cross-entropy.

    Selects the epoch with the best validation ROC-AUC and restores those
    weights before computing final test metrics.
    """
    set_seed(seed)

    a_hat = normalize_adjacency(data.adj)
    x, y = data.x, data.y
    weights = class_weights(y, data.train_mask)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best_val_auc = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, a_hat)
        loss = criterion(logits[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation metric for model selection.
        val_metrics = evaluate_model(model, x, a_hat, y, data.val_mask, threshold)
        record = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "val_roc_auc": val_metrics.roc_auc,
        }
        history.append(record)
        if verbose and epoch % 20 == 0:
            print(f"epoch {epoch:3d} loss {record['train_loss']:.4f} "
                  f"val_auc {record['val_roc_auc']:.4f}")

        if not np.isnan(val_metrics.roc_auc) and val_metrics.roc_auc > best_val_auc:
            best_val_auc = val_metrics.roc_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_metrics = evaluate_model(model, x, a_hat, y, data.val_mask, threshold)
    test_metrics = evaluate_model(model, x, a_hat, y, data.test_mask, threshold)
    return TrainResult(
        model=model,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


@torch.no_grad()
def predict_proba(model: nn.Module, data: GraphData) -> torch.Tensor:
    """Fraud probabilities for every node."""
    model.eval()
    a_hat = normalize_adjacency(data.adj)
    logits = model(data.x, a_hat)
    return fraud_probabilities(logits)
