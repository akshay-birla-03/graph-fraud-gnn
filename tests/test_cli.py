import graphfraud
from graphfraud.cli import main


def test_version_exposed():
    assert isinstance(graphfraud.__version__, str)
    assert graphfraud.__version__.count(".") >= 2


def test_public_api_importable():
    for name in ("GCN", "MLP", "GCNLayer", "normalize_adjacency",
                 "generate_transaction_graph", "train_node_classifier"):
        assert hasattr(graphfraud, name)


def test_cli_runs(capsys):
    # Small/fast config so this stays quick.
    rc = main(["--epochs", "40", "--n-legit", "160", "--n-rings", "4", "--hidden", "16"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TEST RESULTS" in out
    assert "GCN" in out and "MLP" in out
    assert "GCN-over-MLP ROC-AUC gain" in out
