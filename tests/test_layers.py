import math

import torch

from graphfraud.layers import GCNLayer, normalize_adjacency


def test_normalize_adjacency_self_loops_and_symmetry():
    # Simple path graph 0-1-2 (undirected).
    a = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    a_hat = normalize_adjacency(a)
    # Symmetric.
    assert torch.allclose(a_hat, a_hat.t(), atol=1e-6)
    # Self-loops added: diagonal must be strictly positive.
    assert (torch.diagonal(a_hat) > 0).all()


def test_normalize_adjacency_hand_computed():
    # Two connected nodes 0-1. With self-loops, degrees are 2 and 2.
    # A_hat = D^-1/2 (A+I) D^-1/2. Every nonzero entry = 1/sqrt(2*2) = 0.5.
    a = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    a_hat = normalize_adjacency(a)
    expected = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert torch.allclose(a_hat, expected, atol=1e-6)


def test_normalize_adjacency_row_scaling_star_graph():
    # Star: centre 0 connected to 1,2,3. Degrees w/ self loop: c=4, leaves=2.
    a = torch.zeros(4, 4)
    for j in (1, 2, 3):
        a[0, j] = 1.0
        a[j, 0] = 1.0
    a_hat = normalize_adjacency(a)
    # Edge (0, j): 1 / sqrt(4 * 2) = 1/sqrt(8).
    assert math.isclose(a_hat[0, 1].item(), 1.0 / math.sqrt(8), abs_tol=1e-6)
    # Centre self-loop: 1 / sqrt(4 * 4) = 0.25.
    assert math.isclose(a_hat[0, 0].item(), 0.25, abs_tol=1e-6)
    # Leaf self-loop: 1 / sqrt(2 * 2) = 0.5.
    assert math.isclose(a_hat[1, 1].item(), 0.5, abs_tol=1e-6)


def test_normalize_adjacency_isolated_node_no_nan():
    # Node 2 isolated -> degree (with self loop) = 1, no inf/nan.
    a = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    a_hat = normalize_adjacency(a)
    assert torch.isfinite(a_hat).all()
    assert math.isclose(a_hat[2, 2].item(), 1.0, abs_tol=1e-6)


def test_gcn_layer_output_shape():
    layer = GCNLayer(5, 3)
    x = torch.randn(7, 5)
    a_hat = normalize_adjacency(torch.eye(7))
    out = layer(x, a_hat)
    assert out.shape == (7, 3)


def test_gcn_layer_differentiable():
    layer = GCNLayer(4, 2)
    x = torch.randn(6, 4, requires_grad=True)
    a = (torch.rand(6, 6) > 0.5).float()
    a = ((a + a.t()) > 0).float()
    a_hat = normalize_adjacency(a)
    out = layer(x, a_hat)
    out.sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_message_passing_mixes_neighbours():
    # A node's output must depend on its neighbours' features. Perturb a
    # neighbour and confirm the target node's embedding changes; perturb a
    # non-neighbour (in a 1-layer sense) and confirm it does NOT.
    torch.manual_seed(0)
    # Graph: 0-1, 2 isolated from 0.
    a = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    a_hat = normalize_adjacency(a)
    layer = GCNLayer(2, 2)
    x = torch.zeros(3, 2)
    base = layer(x, a_hat)[0].clone()

    # Perturb neighbour (node 1) -> node 0 output should change.
    x1 = x.clone()
    x1[1] = torch.tensor([1.0, -1.0])
    changed = layer(x1, a_hat)[0]
    assert not torch.allclose(base, changed, atol=1e-6)

    # Perturb non-neighbour (node 2) -> node 0 output unchanged.
    x2 = x.clone()
    x2[2] = torch.tensor([5.0, 5.0])
    same = layer(x2, a_hat)[0]
    assert torch.allclose(base, same, atol=1e-6)
