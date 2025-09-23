import torch
import dgl
import numpy as np

from bridge.rewiring.operations import create_rewired_graph


def make_toy_graph(n=6, undirected=True):
    src = torch.tensor([0,1,2,3,4,5])
    dst = torch.tensor([1,2,3,4,5,0])
    g = dgl.graph((src, dst), num_nodes=n)
    if undirected:
        g = dgl.to_bidirected(g, copy_ndata=True)
    # minimal node data
    g.ndata['feat'] = torch.eye(n)
    g.ndata['label'] = torch.tensor([0,0,1,1,2,2])
    g.ndata['train_mask'] = torch.ones(n, dtype=torch.bool)
    g.ndata['val_mask'] = torch.ones(n, dtype=torch.bool)
    g.ndata['test_mask'] = torch.ones(n, dtype=torch.bool)
    return g


def test_full_resample_symmetry_and_bounds():
    g = make_toy_graph(n=6, undirected=True)
    k = 3
    # one-hot hard predictions Z
    pred = g.ndata['label'].clone()
    Z_pred = torch.zeros(g.num_nodes(), k)
    Z_pred.scatter_(1, pred.unsqueeze(1), 1.0)
    # simple B_opt that induces more within-class edges
    B = torch.tensor([[0.9, 0.1, 0.1],
                      [0.1, 0.9, 0.1],
                      [0.1, 0.1, 0.9]], dtype=torch.float32)

    g2 = create_rewired_graph(g, B, pred, Z_pred, sym_type='upper', device='cpu')

    # Check graph type
    assert isinstance(g2, dgl.DGLGraph)

    # Check no self-loops added implicitly
    u, v = g2.edges()
    assert not torch.any(u == v)

    # Check symmetry by comparing adjacency to its transpose
    A = g2.adj().to_dense()
    assert torch.allclose(A, A.T)


def test_probabilities_respected_monotonicity():
    g = make_toy_graph(n=12, undirected=True)
    k = 3
    pred = g.ndata['label'].clone()
    Z_pred = torch.zeros(g.num_nodes(), k)
    Z_pred.scatter_(1, pred.unsqueeze(1), 1.0)

    # Two different B matrices: one with higher within-class prob
    B_high = torch.tensor([[0.8, 0.05, 0.05],
                           [0.05, 0.8, 0.05],
                           [0.05, 0.05, 0.8]], dtype=torch.float32)
    B_low = torch.tensor([[0.4, 0.3, 0.3],
                          [0.3, 0.4, 0.3],
                          [0.3, 0.3, 0.4]], dtype=torch.float32)

    # Sample multiple times to estimate average within-class density
    def sample_within_density(B, trials=10):
        total_within = 0
        total_edges = 0
        for _ in range(trials):
            g2 = create_rewired_graph(g, B, pred, Z_pred, sym_type='upper', device='cpu')
            A = g2.adj().to_dense()
            mask_same = pred.unsqueeze(1) == pred.unsqueeze(0)
            total_within += (A[mask_same].sum().item() / 2)  # undirected
            total_edges += (A.sum().item() / 2)
        return total_within / max(total_edges, 1e-6)

    within_high = sample_within_density(B_high)
    within_low = sample_within_density(B_low)
    assert within_high >= within_low - 0.05
