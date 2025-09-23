import dgl
import torch

from bridge.rewiring.sdrf import sdrf_rewire


def test_sdrf_rewire_interface(tiny_graph):
    g = tiny_graph(n=10, undirected=True)
    # Force CPU-friendly implementation
    g2 = sdrf_rewire(g, tau=0.1, n_iterations=1, c_plus=0.0, symmetric=True, device='cpu', force_implementation='pytorch')
    assert isinstance(g2, dgl.DGLGraph)
    assert g2.num_nodes() == g.num_nodes()
    # Features preserved
    for k in g.ndata.keys():
        assert k in g2.ndata
        assert g2.ndata[k].shape == g.ndata[k].shape
