"""Synthetic transaction-graph generator with planted fraud rings.

The generator produces a transductive node-classification problem: accounts are
nodes, transactions are (undirected) edges, and a small minority of accounts are
fraudulent. Fraudsters are organised into *rings* -- small communities that
transact densely among themselves -- which is exactly the kind of collusive
structure that a graph model can exploit but a tabular model cannot see.

Everything is deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Node feature layout (column index -> meaning).
FEATURE_NAMES = (
    "account_age",
    "avg_txn_amount",
    "txn_count",
    "in_degree_tendency",
    "out_degree_tendency",
    "country_risk",
)
NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class GraphData:
    """Container for a generated transaction graph."""

    x: torch.Tensor  # [N, F] node features
    adj: torch.Tensor  # [N, N] dense symmetric adjacency (0/1, no self-loops)
    edge_index: torch.Tensor  # [2, E] undirected edges (both directions)
    y: torch.Tensor  # [N] int64 labels (1 = fraud, 0 = legit)
    train_mask: torch.Tensor  # [N] bool
    val_mask: torch.Tensor  # [N] bool
    test_mask: torch.Tensor  # [N] bool
    ring_id: torch.Tensor  # [N] int64, ring index or -1 for non-ring nodes

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def num_edges(self) -> int:
        # edge_index stores both directions; count undirected edges.
        return self.edge_index.shape[1] // 2


def _add_undirected_edge(edges: set, u: int, v: int) -> None:
    if u == v:
        return
    a, b = (u, v) if u < v else (v, u)
    edges.add((a, b))


def generate_transaction_graph(
    n_legit: int = 460,
    n_rings: int = 8,
    ring_size_range: tuple[int, int] = (5, 12),
    background_attach: int = 2,
    ring_internal_prob: float = 0.7,
    ring_external_edges: int = 2,
    seed: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> GraphData:
    """Generate a transaction graph with planted fraud rings.

    Legit accounts form a preferential-attachment (Barabasi-Albert style)
    background graph. Fraud rings are then planted: each ring is a small, densely
    connected clique-like community of fraudulent accounts with only a few edges
    reaching into the background (mirroring how fraud rings shuffle money among
    themselves while maintaining a thin veneer of normal activity).

    Fraudulent nodes also have subtly shifted features (younger accounts, higher
    average transaction amounts, higher country risk) but with heavy overlap, so
    that features alone are weak and the graph structure carries real signal.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Build the legit background graph via preferential attachment. ---
    edges: set = set()
    m = max(1, background_attach)
    # Seed clique.
    degree = np.zeros(n_legit, dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            _add_undirected_edge(edges, i, j)
            degree[i] += 1
            degree[j] += 1
    for new in range(m, n_legit):
        # Attach to m existing nodes with probability proportional to degree.
        weights = degree[:new] + 1.0
        probs = weights / weights.sum()
        targets = rng.choice(new, size=min(m, new), replace=False, p=probs)
        for t in targets:
            _add_undirected_edge(edges, new, int(t))
            degree[new] += 1
            degree[int(t)] += 1

    # --- 2. Plant fraud rings. ---
    ring_id = np.full(0, -1)  # placeholder, resized below
    node_ring = {}
    next_node = n_legit
    ring_members: list[list[int]] = []
    for r in range(n_rings):
        size = int(rng.integers(ring_size_range[0], ring_size_range[1] + 1))
        members = list(range(next_node, next_node + size))
        next_node += size
        ring_members.append(members)
        for idx in members:
            node_ring[idx] = r
        # Dense internal connectivity.
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if rng.random() < ring_internal_prob:
                    _add_undirected_edge(edges, members[i], members[j])
        # Guarantee the ring is connected (chain fallback).
        for i in range(len(members) - 1):
            _add_undirected_edge(edges, members[i], members[i + 1])
        # A few thin links into the legit background (mules / cash-out points).
        for _ in range(ring_external_edges):
            u = int(rng.choice(members))
            v = int(rng.integers(0, n_legit))
            _add_undirected_edge(edges, u, v)

    n_total = next_node

    # --- 3. Labels and ring ids. ---
    y = np.zeros(n_total, dtype=np.int64)
    ring_id = np.full(n_total, -1, dtype=np.int64)
    for idx, r in node_ring.items():
        y[idx] = 1
        ring_id[idx] = r

    # --- 4. Node features. ---
    x = np.zeros((n_total, NUM_FEATURES), dtype=np.float32)
    # Legit distributions.
    for i in range(n_total):
        if y[i] == 0:
            account_age = rng.normal(5.0, 2.0)
            avg_amt = rng.normal(200.0, 80.0)
            txn_count = rng.normal(40.0, 20.0)
            in_tend = rng.normal(0.5, 0.2)
            out_tend = rng.normal(0.5, 0.2)
            country_risk = rng.normal(0.3, 0.15)
        else:
            # Fraud: subtly shifted, heavy overlap.
            account_age = rng.normal(3.5, 2.0)
            avg_amt = rng.normal(260.0, 90.0)
            txn_count = rng.normal(55.0, 25.0)
            in_tend = rng.normal(0.6, 0.2)
            out_tend = rng.normal(0.55, 0.2)
            country_risk = rng.normal(0.45, 0.18)
        x[i] = [account_age, avg_amt, txn_count, in_tend, out_tend, country_risk]

    # Standardise features (helps optimisation; done globally, transductive).
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)

    # --- 5. Adjacency tensors. ---
    adj = np.zeros((n_total, n_total), dtype=np.float32)
    src, dst = [], []
    for (a, b) in edges:
        adj[a, b] = 1.0
        adj[b, a] = 1.0
        src += [a, b]
        dst += [b, a]
    edge_index = np.stack([np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)])

    # --- 6. Stratified train/val/test masks. ---
    train_mask, val_mask, test_mask = _make_masks(
        y, train_frac, val_frac, rng
    )

    return GraphData(
        x=torch.from_numpy(x),
        adj=torch.from_numpy(adj),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(y),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        ring_id=torch.from_numpy(ring_id),
    )


def _make_masks(
    y: np.ndarray, train_frac: float, val_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split so each class appears in every split."""
    n = y.shape[0]
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * train_frac))
        n_va = int(round(len(idx) * val_frac))
        # Guarantee at least one node per split for the minority class.
        n_tr = max(1, min(n_tr, len(idx) - 2)) if len(idx) >= 3 else n_tr
        n_va = max(1, n_va) if len(idx) >= 3 else n_va
        train_mask[idx[:n_tr]] = True
        val_mask[idx[n_tr:n_tr + n_va]] = True
        test_mask[idx[n_tr + n_va:]] = True
    return train_mask, val_mask, test_mask


def intra_ring_edge_density(data: GraphData) -> float:
    """Average internal edge density across rings (edges / possible edges)."""
    ring_id = data.ring_id.numpy()
    adj = data.adj.numpy()
    densities = []
    for r in np.unique(ring_id):
        if r < 0:
            continue
        members = np.where(ring_id == r)[0]
        k = len(members)
        if k < 2:
            continue
        sub = adj[np.ix_(members, members)]
        n_edges = sub.sum() / 2.0
        possible = k * (k - 1) / 2.0
        densities.append(n_edges / possible)
    return float(np.mean(densities)) if densities else 0.0


def global_edge_density(data: GraphData) -> float:
    """Overall edge density of the whole graph."""
    n = data.num_nodes
    possible = n * (n - 1) / 2.0
    return float(data.num_edges / possible)
