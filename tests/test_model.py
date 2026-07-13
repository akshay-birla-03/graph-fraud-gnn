import torch

from graphfraud.data import generate_transaction_graph
from graphfraud.layers import normalize_adjacency
from graphfraud.model import GCN, MLP


def test_gcn_forward_shape():
    d = generate_transaction_graph(seed=0)
    a_hat = normalize_adjacency(d.adj)
    model = GCN(d.num_features, hidden_features=16, num_classes=2)
    out = model(d.x, a_hat)
    assert out.shape == (d.num_nodes, 2)


def test_gcn_num_layers():
    model = GCN(6, hidden_features=8, num_layers=3)
    assert len(model.layers) == 3
    a_hat = normalize_adjacency(torch.eye(10))
    out = model(torch.randn(10, 6), a_hat)
    assert out.shape == (10, 2)


def test_mlp_ignores_graph():
    # MLP output must be identical regardless of adjacency passed in.
    torch.manual_seed(0)
    model = MLP(6, hidden_features=8)
    model.eval()
    x = torch.randn(12, 6)
    a1 = normalize_adjacency(torch.eye(12))
    a2 = normalize_adjacency((torch.rand(12, 12) > 0.5).float())
    out1 = model(x, a1)
    out2 = model(x, a2)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_gcn_uses_graph():
    # GCN output SHOULD change if adjacency changes (it uses structure).
    torch.manual_seed(0)
    model = GCN(6, hidden_features=8)
    model.eval()
    x = torch.randn(12, 6)
    a1 = normalize_adjacency(torch.eye(12))  # no edges -> only self
    dense = ((torch.rand(12, 12) > 0.3).float())
    dense = ((dense + dense.t()) > 0).float()
    dense.fill_diagonal_(0)
    a2 = normalize_adjacency(dense)
    out1 = model(x, a1)
    out2 = model(x, a2)
    assert not torch.allclose(out1, out2, atol=1e-4)


def test_gcn_backward():
    d = generate_transaction_graph(seed=0)
    a_hat = normalize_adjacency(d.adj)
    model = GCN(d.num_features, hidden_features=16)
    out = model(d.x, a_hat)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
