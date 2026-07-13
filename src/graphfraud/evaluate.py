"""Evaluation metrics for node classification under class imbalance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def fraud_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax probability of the fraud (positive) class from 2-class logits."""
    return torch.softmax(logits, dim=1)[:, 1]


def compute_metrics(
    y_true, y_score, threshold: float = 0.5
) -> Metrics:
    """Compute ROC-AUC, PR-AUC, F1, precision, recall.

    ``y_score`` is the predicted probability of the positive (fraud) class.
    """
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    # ROC/PR-AUC require both classes present; guard for degenerate inputs.
    if len(np.unique(y_true)) < 2:
        roc = float("nan")
        pr = float("nan")
    else:
        roc = float(roc_auc_score(y_true, y_score))
        pr = float(average_precision_score(y_true, y_score))

    return Metrics(
        roc_auc=roc,
        pr_auc=pr,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
    )


@torch.no_grad()
def evaluate_model(model, x, a_hat, y, mask, threshold: float = 0.5) -> Metrics:
    """Run a model and compute metrics on the masked nodes."""
    model.eval()
    logits = model(x, a_hat)
    scores = fraud_probabilities(logits)
    return compute_metrics(y[mask], scores[mask], threshold=threshold)


@dataclass
class Comparison:
    gcn: Metrics
    mlp: Metrics

    @property
    def roc_auc_gain(self) -> float:
        return self.gcn.roc_auc - self.mlp.roc_auc

    @property
    def pr_auc_gain(self) -> float:
        return self.gcn.pr_auc - self.mlp.pr_auc


def compare_models(gcn_metrics: Metrics, mlp_metrics: Metrics) -> Comparison:
    """Bundle two metric sets and expose GCN-over-MLP gains."""
    return Comparison(gcn=gcn_metrics, mlp=mlp_metrics)
