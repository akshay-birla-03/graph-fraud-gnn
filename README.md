# graph-fraud-gnn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akshay-birla-03/graph-fraud-gnn/blob/main/notebooks/Run_in_Colab.ipynb)

A **Graph Convolutional Network (GCN), implemented from scratch in PyTorch**, for
**fraud-ring detection** on a financial transaction graph. Accounts are nodes,
transactions are edges, and a small minority of accounts are fraudulent. The
fraudsters are organised into **rings** — dense little communities that shuffle
money among themselves — and the whole point of this project is to show that a
model which can *see the graph* catches them, while an otherwise-identical model
that only sees per-account features does not.

No `torch-geometric`. The GCN layer, the symmetric adjacency normalisation, and
the message passing are all written by hand with dense matrix multiplies.

> The graph is **synthetic**, but the setup mirrors how real anti-money-laundering
> (AML) systems work: fraud is collusive, so the *structure* of who-pays-whom
> carries signal that a purely tabular model is blind to.

---

## Why fraud rings are a graph problem

A single fraudulent account can look almost identical to a legit one — features
overlap heavily (a slightly younger account, a slightly higher average transfer,
a slightly riskier country). Taken one row at a time, the signal is weak.

But fraudsters **collude**. They open a cluster of accounts and cycle funds
between them to layer and launder money. That collusion shows up as a dense
subgraph: a *ring*. A model that aggregates information over graph neighbourhoods
sees "this account sits inside an unusually tight cluster of look-alike accounts"
— a feature that literally does not exist for a row-wise classifier.

```
        legit background (preferential attachment)
   o---o---o---o---o---o---o---o---o---o---o
    \ /   \   /       \     /       \   /
     o     o           o             o
     |                                |    thin "mule" links
   [ o=o=o=o ]      [ o=o=o=o=o ]   into the legit graph
     \_____/          \________/
      FRAUD RING        FRAUD RING
   (dense internal edges, ~0.75 density)
```

Measured on the default graph: **intra-ring edge density ≈ 0.75** vs a **global
edge density ≈ 0.008** — rings are ~90x denser internally than the graph at
large. That is the structure the GCN exploits.

---

## The GCN propagation rule

Each graph-convolution layer computes

```
H' = σ( Â · H · W )
```

where `H` are the current node embeddings, `W` is a learnable weight matrix, `σ`
is ReLU, and `Â` is the **symmetric-normalised adjacency with self-loops**:

```
Â = D̃^(-1/2) · (A + I) · D̃^(-1/2) ,   D̃ = diag(row-sums of A + I)
```

Two design choices matter, and both are implemented in
[`layers.normalize_adjacency`](src/graphfraud/layers.py):

- **`A + I` (self-loops).** Without them, `Â · H` replaces each node's embedding
  purely with a blend of its *neighbours* and throws away the node's own signal.
  Adding the identity keeps the node in its own receptive field.
- **`D^(-1/2) · … · D^(-1/2)` (symmetric normalisation).** Raw `A · H` sums over
  neighbours, so high-degree nodes get huge activations and stacked layers
  explode. Symmetric normalisation weights each edge `(i, j)` by
  `1 / sqrt(deg(i) · deg(j))`, keeping the operator's spectral radius bounded so
  you can stack layers stably.

Stacking `L` layers gives every node a receptive field of its `L`-hop
neighbourhood. Two layers already cover a whole ring, which is why the default
`GCN` uses two.

The **MLP baseline** ([`model.MLP`](src/graphfraud/model.py)) has the *same width
and depth* but simply drops `Â`: it is `H' = σ(H · W)`. It is graph-blind by
construction, so the GCN-vs-MLP gap isolates exactly the value of graph structure.

---

## Results (measured, seed 0)

Run `graphfraud` yourself — these are the real numbers from this repo:

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|--------:|-------:|----:|----------:|-------:|
| **GCN** (graph)      | **0.998** | **0.990** | **0.917** | 1.000 | 0.846 |
| MLP (graph-blind)    | 0.772 | 0.575 | 0.450 | 0.333 | 0.692 |
| **GCN − MLP gain**   | **+0.227** | **+0.415** | +0.467 | — | — |

Graph: 525 nodes, 1118 edges, 6 features, 65 fraud accounts (12.4%), 8 rings.

The GCN clears the quality bar (test ROC-AUC ≥ 0.80) with room to spare **and**
beats the tabular baseline by a wide margin on both ROC-AUC and — the metric that
matters under class imbalance — PR-AUC. That gap *is* the evidence that graph
structure helps. The test suite asserts it on real training runs (see
`tests/test_train.py::test_gcn_beats_mlp_key_quality_bar`), so it cannot silently
regress.

**Honesty note.** These numbers are strong partly because the data is synthetic
and the planted rings are cleanly separable in structure (even while their
*features* overlap). On real AML data the absolute AUCs would be lower and noisier.
The *qualitative* result — structure-aware models beat structure-blind ones on
collusive fraud — is the robust, transferable takeaway.

---

## Install

```bash
pip install -e ".[dev]"          # torch, numpy, scikit-learn, pytest, ruff
# CPU-only torch, if needed:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

CLI (generate graph → train GCN + MLP → compare):

```bash
graphfraud                       # defaults, seed 0
graphfraud --seed 1 --epochs 300 --n-rings 12
```

Library:

```python
from graphfraud import (
    generate_transaction_graph, GCN, MLP,
    train_node_classifier, compare_models,
)

data = generate_transaction_graph(seed=0)
gcn = train_node_classifier(GCN(data.num_features), data, epochs=200, seed=0)
mlp = train_node_classifier(MLP(data.num_features), data, epochs=200, seed=0)

cmp = compare_models(gcn.test_metrics, mlp.test_metrics)
print(gcn.test_metrics.as_dict())
print("ROC-AUC gain:", cmp.roc_auc_gain)
```

## Project layout

```
src/graphfraud/
  data.py       synthetic transaction graph + planted fraud rings, masks
  layers.py     normalize_adjacency(A) and from-scratch GCNLayer (Â H W)
  model.py      GCN (stacked layers) and the graph-blind MLP baseline
  train.py      full-batch transductive training, class-weighted CE
  evaluate.py   ROC-AUC / PR-AUC / F1 / precision / recall, model comparison
  cli.py        end-to-end GCN-vs-MLP comparison
tests/          34 tests (layers, data, model, train, evaluate, cli)
```

## Testing

```bash
python -m pytest        # 34 tests, ~50s on CPU
ruff check src tests
```

The tests verify, among other things: `normalize_adjacency` on hand-computed
tiny graphs (self-loops, symmetry, `1/sqrt(deg·deg)` scaling); that a GCN layer
actually mixes neighbour information (perturbing a neighbour changes a node's
output, perturbing a non-neighbour does not); that the generator is deterministic
and its rings are denser than the background; that training reduces the masked
loss; and the headline bar — **GCN test ROC-AUC ≥ 0.80 and clearly above the MLP**.

## License

MIT — see [LICENSE](LICENSE).
