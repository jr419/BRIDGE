"""
Operations for rewiring graph structures.

This module provides functions for modifying graph structures through
edge addition and removal based on optimization criteria.
"""

import torch
import dgl
from typing import Tuple, List, Dict, Union, Optional, Any


def create_rewired_graph(
    g: dgl.DGLGraph,
    B_opt_tensor: torch.Tensor,
    pred: torch.Tensor,
    Z_pred: torch.Tensor,
    p_remove: float,
    p_add: float,
    sym_type: str = 'upper',
    device: Union[str, torch.device] = 'cpu'
) -> dgl.DGLGraph:
    """
    Create a rewired version of a graph using predicted class probabilities
    and an optimal block matrix.
    
    The rewiring process:
    1. Computes optimal edge probabilities based on the provided block matrix and node embeddings
    2. Removes existing edges with probability p_remove * (1 - optimal_prob)
    3. Adds new edges with probability p_add * optimal_prob
    4. Ensures the resulting graph maintains desired symmetry properties
    
    Args:
        g: Original graph to rewire
        B_opt_tensor: Optimal block matrix
        pred: Predicted class labels for each node
        Z_pred: Predicted class probabilities for each node (softmax outputs)
        p_remove: Probability of removing existing edges
        p_add: Probability of adding new edges
        sym_type: Type of symmetry to enforce ('upper', 'lower', or 'asymetric')
        device: Device to perform computations on
        
    Returns:
        dgl.DGLGraph: The rewired graph
    """
    # Get original graph attributes
    n_nodes = g.num_nodes()
    
    # ========== Compute Edge Probabilities ==========
    A_opt_p = (Z_pred.cpu() @ B_opt_tensor.cpu() @ Z_pred.cpu().T) / n_nodes

    # Get current adjacency matrix
    A_old = g.cpu().adj().to_dense()

    # ========== Likelihood-based Rewiring ==========
    # Compute new edge probabilities:
    # - For existing edges: keep with prob (1 - p_remove * (1 - A_opt_p))
    # - For non-existing edges: add with prob (p_add * A_opt_p)
    A_p = A_old * (1 - p_remove * (1 - A_opt_p)) + (1 - A_old) * A_opt_p * p_add
    
    # Clamp probabilities to avoid numerical errors
    A_p = torch.clamp(A_p, 0, 1)

    # Handle NaN values by setting them to 0
    A_p[torch.isnan(A_p)] = 0

    # Sample new adjacency matrix
    A = torch.bernoulli(A_p)
    
    # Ensure symmetry if required
    if sym_type == 'upper':
        A = torch.triu(A) + torch.triu(A, 1).T  # Ensure symmetry using upper triangular
    elif sym_type == 'lower':
        A = torch.tril(A) + torch.tril(A, -1).T  # Ensure symmetry using lower triangular

    # ========== Build Rewired Graph ==========
    g_rewired = g.clone().cpu()
    g_rewired.remove_edges(torch.arange(g_rewired.num_edges()))
    u, v = torch.where(A > 0)
    g_rewired = dgl.add_edges(g_rewired, u, v)

    return g_rewired
