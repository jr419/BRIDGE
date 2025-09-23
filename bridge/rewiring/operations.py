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
    sym_type: str = 'upper',
    device: Union[str, torch.device] = 'cpu'
) -> dgl.DGLGraph:
    """
    Create a rewired version of a graph using predicted class assignments
    and an optimal block matrix.
    
    This implementation always fully resamples the adjacency according to the
    optimal probabilities (i.e., full resampling). We use hard predictions
    (one-hot Z_pred). No temperature, p_add, or p_remove are used.
    
    The rewiring process:
    1. Computes optimal edge probabilities based on the provided block matrix and node assignments
    2. Samples every potential edge independently from Bernoulli(A_opt_p)
    3. Ensures the resulting graph maintains desired symmetry properties
    
    Args:
        g: Original graph to rewire
        B_opt_tensor: Optimal block matrix
        pred: Predicted class labels for each node (hard assignments)
        Z_pred: One-hot class assignments for each node
        sym_type: Type of symmetry to enforce ('upper', 'lower', or 'asymetric')
        device: Device to perform computations on
        
    Returns:
        dgl.DGLGraph: The rewired graph
    """
    # Get original graph attributes
    n_nodes = g.num_nodes()
    
    # ========== Compute Edge Probabilities ==========
    A_opt_p = (Z_pred.cpu() @ B_opt_tensor.cpu() @ Z_pred.cpu().T) / n_nodes

    # Clamp probabilities to valid range and handle NaNs
    A_opt_p = torch.clamp(A_opt_p, 0, 1)
    A_opt_p[torch.isnan(A_opt_p)] = 0
    # Do not allow self-loop probabilities
    A_opt_p.fill_diagonal_(0)

    # ========== Full Resampling ==========
    A = torch.bernoulli(A_opt_p)
    
    # Ensure symmetry if required
    if sym_type == 'upper':
        A = torch.triu(A) + torch.triu(A, 1).T  # Ensure symmetry using upper triangular
    elif sym_type == 'lower':
        A = torch.tril(A) + torch.tril(A, -1).T  # Ensure symmetry using lower triangular

    # Remove any self-loops that may remain
    A.fill_diagonal_(0)

    # ========== Build Rewired Graph ==========
    g_rewired = g.clone().cpu()
    g_rewired.remove_edges(torch.arange(g_rewired.num_edges()))
    u, v = torch.where(A > 0)
    g_rewired = dgl.add_edges(g_rewired, u, v)

    return g_rewired
