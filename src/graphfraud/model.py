"""Node-classification models: a from-scratch GCN and a graph-blind MLP baseline.

Both networks share the same hidden width and depth so that the *only* material
difference is whether the model is allowed to use the graph. That makes the
GCN-vs-MLP comparison a fair test of whether graph structure helps.
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import GCNLayer


class GCN(nn.Module):
    """Stacked GCN for transductive node classification.

    Architecture (default 2 hidden layers -> 3 GCN layers total is overkill for
    these graphs; 2 layers already reaches 2-hop neighbourhoods which covers a
    ring). We expose ``num_layers`` for flexibility.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden_features))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_features, hidden_features))
        self.layers.append(GCNLayer(hidden_features, num_classes))

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, a_hat)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
                h = torch.dropout(h, p=self.dropout, train=self.training)
        return h  # logits [N, num_classes]


class MLP(nn.Module):
    """Graph-blind baseline: a plain feed-forward net over node features only.

    Same width / depth as the GCN. It never sees the adjacency, so it can only
    exploit per-node tabular signal -- which is exactly what we want to beat.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
        self.layers.append(nn.Linear(hidden_features, num_classes))

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor | None = None) -> torch.Tensor:
        # a_hat accepted (and ignored) so train/eval code can treat both models
        # with an identical call signature.
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
                h = torch.dropout(h, p=self.dropout, train=self.training)
        return h  # logits [N, num_classes]
