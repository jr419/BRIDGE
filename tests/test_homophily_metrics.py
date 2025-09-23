import torch
import dgl

from bridge.utils import (
    compute_label_matrix,
    class_bottlenecking_score,
    self_bottlenecking_score,
    total_bottlenecking_score,
)


def test_compute_label_matrix_basic():
    labels = torch.tensor([2, 2, 1, 0, 1])
    M, classes = compute_label_matrix(labels)
    assert M.shape == (5, 3)  # three unique classes {0,1,2}
    # Check one-hot rows align to classes order
    idx_2 = (classes == 2).nonzero().item()
    assert M[0, idx_2] == 1 and M[1, idx_2] == 1


def make_complete_graph(n: int):
    src, dst = torch.where(torch.ones(n, n) - torch.eye(n) > 0)
    return dgl.graph((src, dst), num_nodes=n)


def test_bottlenecking_scores_shapes_and_nonneg(tiny_graph):
    g = tiny_graph(n=6, undirected=True)
    p = 1
    cb = class_bottlenecking_score(p, g)
    sb = self_bottlenecking_score(p, g)
    tb = total_bottlenecking_score(p, g)
    for x in (cb, sb, tb):
        assert x.shape[0] == g.num_nodes()
        assert torch.all(x >= 0)


def test_class_bottlenecking_higher_when_labels_uniform():
    n = 6
    g = make_complete_graph(n)
    g = dgl.to_bidirected(g)
    g.ndata['feat'] = torch.eye(n)

    # All same label -> strongest class concentration
    g.ndata['label'] = torch.zeros(n, dtype=torch.long)
    cb_uniform = class_bottlenecking_score(1, g).mean().item()

    # Alternating labels -> more mixed classes
    g.ndata['label'] = torch.tensor([0, 1, 2, 0, 1, 2])
    cb_mixed = class_bottlenecking_score(1, g).mean().item()

    assert cb_uniform >= cb_mixed - 1e-6

