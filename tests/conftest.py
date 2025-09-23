import torch
import pytest


@pytest.fixture
def tiny_graph():
    def _make(n=6, undirected=True, k=3):
        try:
            import dgl
        except Exception as e:
            pytest.skip(f"DGL unavailable: {e}")
        src = torch.arange(n)
        dst = torch.roll(src, shifts=-1)
        g = dgl.graph((src, dst), num_nodes=n)
        if undirected:
            g = dgl.to_bidirected(g, copy_ndata=True)
        g.ndata['feat'] = torch.eye(n)
        labels = torch.tensor([(i % k) for i in range(n)])
        g.ndata['label'] = labels
        g.ndata['train_mask'] = torch.ones(n, dtype=torch.bool)
        g.ndata['val_mask'] = torch.ones(n, dtype=torch.bool)
        g.ndata['test_mask'] = torch.ones(n, dtype=torch.bool)
        return g
    return _make
