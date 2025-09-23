import numpy as np
import torch
import dgl

from bridge.utils import (
    compute_confidence_interval,
    infer_B,
    generate_all_symmetric_permutation_matrices,
)


def test_compute_confidence_interval_basic():
    data = [1.0, 2.0, 3.0, 4.0]
    mean, lo, hi = compute_confidence_interval(data, confidence=0.95)
    assert abs(mean - 2.5) < 1e-6
    assert lo < mean < hi

    # n < 2 path
    mean1, lo1, hi1 = compute_confidence_interval([5.0])
    assert np.isnan(lo1) and np.isnan(hi1)


def test_infer_B_dimensions(tiny_graph):
    g = tiny_graph(n=6, undirected=True, k=3)
    n = g.num_nodes()
    k = int(g.ndata['label'].max().item()) + 1
    Z = torch.zeros(n, k)
    Z[torch.arange(n), g.ndata['label']] = 1
    B = infer_B(g, Z)
    assert B.shape == (k, k)


def test_generate_all_symmetric_permutation_matrices_small():
    k = 2
    Pi = np.eye(k)
    mats = generate_all_symmetric_permutation_matrices(k, Pi)
    assert len(mats) >= 1
    # All are permutation and symmetric
    for P in mats:
        assert P.shape == (k, k)
        assert (P.T == P).all()
        assert (P.sum(axis=0) == 1).all() and (P.sum(axis=1) == 1).all()

