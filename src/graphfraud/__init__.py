"""graph-fraud-gnn: a from-scratch GCN for fraud-ring detection.

Public API for building transaction graphs, constructing the from-scratch GCN
and MLP baseline, training them transductively, and evaluating / comparing.
"""

from __future__ import annotations

from .data import (
    FEATURE_NAMES,
    GraphData,
    generate_transaction_graph,
    global_edge_density,
    intra_ring_edge_density,
)
from .evaluate import (
    Comparison,
    Metrics,
    compare_models,
    compute_metrics,
    evaluate_model,
    fraud_probabilities,
)
from .layers import GCNLayer, normalize_adjacency
from .model import GCN, MLP
from .train import TrainResult, predict_proba, set_seed, train_node_classifier

__version__ = "0.1.0"

__all__ = [
    "FEATURE_NAMES",
    "GraphData",
    "generate_transaction_graph",
    "global_edge_density",
    "intra_ring_edge_density",
    "Comparison",
    "Metrics",
    "compare_models",
    "compute_metrics",
    "evaluate_model",
    "fraud_probabilities",
    "GCNLayer",
    "normalize_adjacency",
    "GCN",
    "MLP",
    "TrainResult",
    "predict_proba",
    "set_seed",
    "train_node_classifier",
    "__version__",
]
