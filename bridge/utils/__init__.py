"""
Utility functions for graph operations and computations.

This module provides various utility functions for working with graphs,
including homophily metrics, graph operations, and matrix operations.
"""

from .homophily import (local_homophily, local_autophily, local_total_connectivity,
                        compute_label_matrix)
from .graph_utils import (set_seed, check_symmetry, make_symmetric, homophily,
                          build_sparse_adj_matrix, normalize_sparse_adj,
                          get_A_hat_p, get_A_p)
from .matrix_utils import (infer_B, closest_symmetric_permutation_matrix,
                           generate_all_symmetric_permutation_matrices,
                           optimal_B, compute_confidence_interval)

from .dataset_processing import add_train_val_test_splits

__all__ = [
    'local_homophily', 'local_autophily', 'local_total_connectivity',
    'compute_label_matrix', 'set_seed', 'check_symmetry', 'make_symmetric',
    'homophily', 'build_sparse_adj_matrix', 'normalize_sparse_adj',
    'get_A_hat_p', 'get_A_p', 'infer_B', 'closest_symmetric_permutation_matrix',
    'generate_all_symmetric_permutation_matrices', 'optimal_B',
    'compute_confidence_interval', 'add_train_val_test_splits'
]
