"""Command-line entry point: generate the graph, train GCN + MLP, compare."""

from __future__ import annotations

import argparse

from .data import (
    generate_transaction_graph,
    global_edge_density,
    intra_ring_edge_density,
)
from .evaluate import compare_models
from .model import GCN, MLP
from .train import train_node_classifier


def _fmt(m) -> str:
    return (
        f"ROC-AUC={m.roc_auc:.3f}  PR-AUC={m.pr_auc:.3f}  "
        f"F1={m.f1:.3f}  P={m.precision:.3f}  R={m.recall:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="graphfraud",
        description="GCN vs graph-blind MLP for fraud-ring detection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-legit", type=int, default=460)
    parser.add_argument("--n-rings", type=int, default=8)
    args = parser.parse_args(argv)

    data = generate_transaction_graph(
        n_legit=args.n_legit, n_rings=args.n_rings, seed=args.seed
    )
    n_fraud = int(data.y.sum())
    print("=" * 68)
    print("graph-fraud-gnn : fraud-ring detection on a transaction graph")
    print("=" * 68)
    print(f"nodes={data.num_nodes}  edges={data.num_edges}  "
          f"features={data.num_features}")
    print(f"fraud nodes={n_fraud} ({100 * n_fraud / data.num_nodes:.1f}%)  "
          f"rings={args.n_rings}")
    print(f"intra-ring edge density={intra_ring_edge_density(data):.4f}  "
          f"global edge density={global_edge_density(data):.5f}")
    print("-" * 68)

    gcn = GCN(data.num_features, hidden_features=args.hidden)
    mlp = MLP(data.num_features, hidden_features=args.hidden)

    print("training GCN (uses graph structure) ...")
    gcn_res = train_node_classifier(gcn, data, epochs=args.epochs, seed=args.seed)
    print("training MLP (graph-blind baseline) ...")
    mlp_res = train_node_classifier(mlp, data, epochs=args.epochs, seed=args.seed)

    cmp = compare_models(gcn_res.test_metrics, mlp_res.test_metrics)
    print("-" * 68)
    print("TEST RESULTS")
    print(f"  GCN : {_fmt(cmp.gcn)}")
    print(f"  MLP : {_fmt(cmp.mlp)}")
    print("-" * 68)
    print(f"GCN-over-MLP ROC-AUC gain : {cmp.roc_auc_gain:+.3f}")
    print(f"GCN-over-MLP PR-AUC  gain : {cmp.pr_auc_gain:+.3f}")
    verdict = "graph structure HELPS" if cmp.roc_auc_gain > 0.02 else "no clear gain"
    print(f"verdict : {verdict}")
    print("=" * 68)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
