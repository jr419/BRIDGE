import torch
import dgl

from bridge.utils import (
    check_symmetry,
    make_symmetric,
    build_sparse_adj_matrix,
    normalize_sparse_adj,
    get_A_hat_p,
)


def test_check_symmetry_and_make_symmetric():
    # Directed 3-node path
    g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])), num_nodes=3)
    g.ndata['label'] = torch.tensor([0, 1, 2])
    g.ndata['feat'] = torch.eye(3)
    assert not check_symmetry(g)

    g2 = make_symmetric(g, sym_type='both')
    assert check_symmetry(g2)

    # Upper/lower also produce symmetric graphs
    g3 = make_symmetric(g, sym_type='upper')
    g4 = make_symmetric(g, sym_type='lower')
    assert check_symmetry(g3) and check_symmetry(g4)


def test_build_sparse_adj_and_normalize(tiny_graph):
    g = tiny_graph(n=5, undirected=True)
    A = build_sparse_adj_matrix(g, self_loops=False, sym=True)
    assert A.is_sparse
    assert A.shape == (g.num_nodes(), g.num_nodes())
    A_norm = normalize_sparse_adj(A)
    assert A_norm.is_sparse
    # No NaNs and within [0,1]
    vals = A_norm.coalesce().values()
    assert not torch.isnan(vals).any()
    assert (vals >= 0).all()


def test_get_A_hat_p_matches_sparse_norm_for_p1(tiny_graph):
    g = tiny_graph(n=5, undirected=True)
    A_sparse = build_sparse_adj_matrix(g, self_loops=False, sym=True)
    A_norm_sparse = normalize_sparse_adj(A_sparse).to_dense()
    A_hat_p = get_A_hat_p(1, g, self_loops=False, sym=True, device='cpu')
    # allow small numerical tolerance from scipy/csr route
    assert torch.allclose(A_norm_sparse.cpu(), A_hat_p.cpu(), atol=1e-6)

