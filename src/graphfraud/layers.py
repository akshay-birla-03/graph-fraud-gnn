"""From-scratch Graph Convolutional Network building blocks.

We implement the GCN propagation rule of Kipf & Welling (2017):

    H' = sigma( A_hat @ H @ W )

where ``A_hat = D^{-1/2} (A + I) D^{-1/2}`` is the symmetrically normalised
adjacency with self-loops added. No torch-geometric -- just dense matmuls, which
is perfectly fine for the modest graph sizes used here.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Compute the symmetric-normalised adjacency with self-loops.

    Returns ``A_hat = D^{-1/2} (A + I) D^{-1/2}``.

    Adding ``I`` (self-loops) lets a node keep its own signal during message
    passing; the symmetric ``D^{-1/2} . D^{-1/2}`` scaling keeps the spectral
    radius bounded so stacked layers do not blow up or vanish, and weights each
    neighbour by ``1 / sqrt(deg(i) * deg(j))``.
    """
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be a square 2-D matrix")
    n = adj.shape[0]
    identity = torch.eye(n, dtype=adj.dtype, device=adj.device)
    a_tilde = adj + identity  # add self-loops
    deg = a_tilde.sum(dim=1)  # degree including self-loop
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ a_tilde @ d_mat


class GCNLayer(nn.Module):
    """A single graph-convolution layer: ``A_hat @ H @ W (+ b)``.

    The normalised adjacency ``A_hat`` is passed in at ``forward`` time (it is
    computed once per graph and reused across layers).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Glorot / Xavier uniform initialisation.
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # support = H @ W  ;  out = A_hat @ support
        support = x @ self.weight
        out = a_hat @ support
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _sanity_glorot_scale(in_f: int, out_f: int) -> float:
    """Expected std of Glorot-uniform init (used only in tests/docs)."""
    return math.sqrt(2.0 / (in_f + out_f))
