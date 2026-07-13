import numpy as np

from graphfraud.data import generate_transaction_graph
from graphfraud.model import GCN, MLP
from graphfraud.train import class_weights, set_seed, train_node_classifier


def test_training_reduces_loss():
    d = generate_transaction_graph(seed=0)
    model = GCN(d.num_features, hidden_features=16)
    res = train_node_classifier(model, d, epochs=60, seed=0)
    losses = [h["train_loss"] for h in res.history]
    # Later loss should be clearly below the initial loss.
    assert np.mean(losses[-5:]) < losses[0]


def test_class_weights_favour_minority():
    d = generate_transaction_graph(seed=0)
    w = class_weights(d.y, d.train_mask)
    # Fraud (class 1) is the minority, so it gets the larger weight.
    assert w[1] > w[0]


def test_training_deterministic():
    d = generate_transaction_graph(seed=0)
    # Seed before construction so initial weights match too.
    set_seed(7)
    r1 = train_node_classifier(GCN(d.num_features), d, epochs=40, seed=7)
    set_seed(7)
    r2 = train_node_classifier(GCN(d.num_features), d, epochs=40, seed=7)
    assert abs(r1.test_metrics.roc_auc - r2.test_metrics.roc_auc) < 1e-9


def test_gcn_beats_mlp_key_quality_bar():
    # THE key evidence: on held-out test nodes the GCN must reach strong
    # ROC-AUC AND clearly beat the graph-blind MLP baseline.
    d = generate_transaction_graph(seed=0)
    gcn = train_node_classifier(GCN(d.num_features), d, epochs=200, seed=0)
    mlp = train_node_classifier(MLP(d.num_features), d, epochs=200, seed=0)

    gcn_auc = gcn.test_metrics.roc_auc
    mlp_auc = mlp.test_metrics.roc_auc

    assert gcn_auc >= 0.80, f"GCN ROC-AUC too low: {gcn_auc:.3f}"
    assert gcn_auc > mlp_auc + 0.05, (
        f"GCN ({gcn_auc:.3f}) did not clearly beat MLP ({mlp_auc:.3f})"
    )
    # Graph model should also win on PR-AUC (matters under imbalance).
    assert gcn.test_metrics.pr_auc > mlp.test_metrics.pr_auc
