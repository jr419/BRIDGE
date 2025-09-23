import torch
import dgl

from bridge.rewiring.digl import digl_rewired


def test_digl_topk_runs(tiny_graph):
    g = tiny_graph(n=8, undirected=True, k=3)
    # Should run and return a graph with weights
    g2 = digl_rewired(g, method='ppr', alpha=0.2, k=3)
    assert isinstance(g2, dgl.DGLGraph)
    assert 'weight' in g2.edata
    # No NaNs in weights
    assert not torch.isnan(g2.edata['weight']).any()


def test_digl_eps_runs(tiny_graph):
    g = tiny_graph(n=8, undirected=True, k=3)
    g2 = digl_rewired(g, method='heat', t=3.0, eps=1e-3)
    assert isinstance(g2, dgl.DGLGraph)
    assert 'weight' in g2.edata
