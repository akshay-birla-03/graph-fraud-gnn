import torch

from graphfraud.data import (
    NUM_FEATURES,
    generate_transaction_graph,
    global_edge_density,
    intra_ring_edge_density,
)


def test_generator_deterministic():
    d1 = generate_transaction_graph(seed=42)
    d2 = generate_transaction_graph(seed=42)
    assert torch.equal(d1.x, d2.x)
    assert torch.equal(d1.adj, d2.adj)
    assert torch.equal(d1.y, d2.y)
    assert torch.equal(d1.train_mask, d2.train_mask)


def test_generator_seed_changes_graph():
    d1 = generate_transaction_graph(seed=1)
    d2 = generate_transaction_graph(seed=2)
    assert not torch.equal(d1.y, d2.y) or not torch.equal(d1.adj, d2.adj)


def test_feature_dim():
    d = generate_transaction_graph(seed=0)
    assert d.num_features == NUM_FEATURES
    assert d.x.shape[0] == d.num_nodes


def test_adjacency_symmetric_no_self_loops():
    d = generate_transaction_graph(seed=0)
    assert torch.equal(d.adj, d.adj.t())
    assert torch.diagonal(d.adj).sum() == 0


def test_class_imbalance_present():
    d = generate_transaction_graph(seed=0)
    n_fraud = int(d.y.sum())
    frac = n_fraud / d.num_nodes
    assert 0 < frac < 0.25  # fraud is a clear minority


def test_fraud_rings_denser_than_background():
    d = generate_transaction_graph(seed=0)
    intra = intra_ring_edge_density(d)
    glob = global_edge_density(d)
    # Rings should be *dramatically* denser internally than the whole graph.
    assert intra > glob
    assert intra > 10 * glob


def test_ring_membership_matches_labels():
    d = generate_transaction_graph(seed=0)
    ring = d.ring_id.numpy()
    y = d.y.numpy()
    # Every ring member is fraud; every non-ring node is legit.
    assert (y[ring >= 0] == 1).all()
    assert (y[ring < 0] == 0).all()


def test_masks_partition_and_stratified():
    d = generate_transaction_graph(seed=0)
    tr, va, te = d.train_mask, d.val_mask, d.test_mask
    # Disjoint.
    assert (tr & va).sum() == 0
    assert (tr & te).sum() == 0
    assert (va & te).sum() == 0
    # Cover all nodes.
    assert int((tr | va | te).sum()) == d.num_nodes
    # Each split contains at least one fraud node (stratified).
    for m in (tr, va, te):
        assert int(d.y[m].sum()) >= 1


def test_rings_connected_internally():
    d = generate_transaction_graph(seed=0)
    ring = d.ring_id.numpy()
    adj = d.adj.numpy()
    for r in set(ring[ring >= 0].tolist()):
        members = (ring == r).nonzero()[0]
        # Each ring member has at least one intra-ring edge.
        sub = adj[members][:, members]
        assert (sub.sum(axis=1) > 0).all()
