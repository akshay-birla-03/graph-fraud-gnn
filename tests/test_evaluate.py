import numpy as np
import torch

from graphfraud.evaluate import (
    compare_models,
    compute_metrics,
    fraud_probabilities,
)


def test_compute_metrics_perfect():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.8]
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert m.roc_auc == 1.0
    assert m.pr_auc == 1.0
    assert m.f1 == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0


def test_compute_metrics_worst_ranking():
    # Perfectly wrong ranking -> ROC-AUC 0.
    y_true = [0, 0, 1, 1]
    y_score = [0.9, 0.8, 0.1, 0.2]
    m = compute_metrics(y_true, y_score, threshold=0.5)
    assert m.roc_auc == 0.0


def test_compute_metrics_threshold_effect():
    y_true = [0, 1, 1]
    y_score = [0.4, 0.6, 0.45]
    m_low = compute_metrics(y_true, y_score, threshold=0.3)
    m_high = compute_metrics(y_true, y_score, threshold=0.5)
    # Lower threshold -> higher recall.
    assert m_low.recall >= m_high.recall


def test_fraud_probabilities_softmax():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    p = fraud_probabilities(logits)
    assert p.shape == (2,)
    assert p[1] > p[0]
    assert torch.all((p >= 0) & (p <= 1))


def test_compute_metrics_single_class_nan_auc():
    m = compute_metrics([1, 1, 1], [0.2, 0.8, 0.5])
    assert np.isnan(m.roc_auc)
    assert np.isnan(m.pr_auc)


def test_compare_models_gain():
    a = compute_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])  # perfect
    b = compute_metrics([0, 1, 0, 1], [0.6, 0.4, 0.55, 0.45])  # bad
    cmp = compare_models(a, b)
    assert cmp.roc_auc_gain > 0
    assert cmp.gcn is a
    assert cmp.mlp is b
