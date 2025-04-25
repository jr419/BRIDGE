"""
Homophily metrics for graph analysis.

This module provides functions for computing various homophily metrics
for graph neural networks, which measure the similarity of connected nodes.
"""

import torch
import dgl
import numpy as np
from typing import Tuple, List, Dict, Union, Optional, Any
from .graph_utils import get_A_hat_p, build_sparse_adj_matrix, normalize_sparse_adj


def compute_label_matrix(
    y: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a one-hot label matrix from a label vector.
    
    Args:
        y: Label tensor of shape (n_nodes,)
        device: Device to place the output tensor
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - One-hot label matrix of shape (n_nodes, n_classes)
            - Unique class labels
    """
    y = y.long()
    classes = torch.unique(y)
    C = classes.numel()
    n = y.shape[0]
    
    # Map label values to a [0..C-1] range if needed
    # (Necessary if labels are not 0..C-1 already.)
    # A quick way is to create a mapping from unique labels to [0..C-1].
    class2idx = {int(c.item()): i for i, c in enumerate(classes)}
    idx = torch.tensor([class2idx[int(lbl.item())] for lbl in y], device=device)

    # Create one-hot label matrix
    M = torch.zeros(n, C, device=device)
    M[torch.arange(n), idx] = 1.0
    return M, classes


def power_adj_times_matrix(A: torch.Tensor, M: torch.Tensor, p: int) -> torch.Tensor:
    """
    Compute (A^p) M using repeated multiplication in sparse form.
    
    Args:
        A: Sparse adjacency matrix
        M: Dense matrix to multiply with
        p: Power to raise the adjacency matrix to
        
    Returns:
        torch.Tensor: Result of (A^p) M
    """
    if p < 1:
        raise ValueError("Power p must be a positive integer.")
    if p == 1:
        return sparse_mm(A, M)
    
    # Repeated multiplication p times
    result = M
    for _ in range(p):
        result = sparse_mm(A, result)
    return result


def sparse_mm(sparse_A: torch.Tensor, dense_B: torch.Tensor) -> torch.Tensor:
    """
    Perform sparse-dense matrix multiplication.
    
    Args:
        sparse_A: Sparse matrix of shape (m, n)
        dense_B: Dense matrix of shape (n, k)
        
    Returns:
        torch.Tensor: Result of sparse_A @ dense_B of shape (m, k)
    """
    return torch.sparse.mm(sparse_A, dense_B)


def local_homophily(
    p: int, 
    g: dgl.DGLGraph, 
    y: Optional[torch.Tensor] = None,
    self_loops: bool = False,
    do_hp: bool = False,
    fix_d: bool = True, 
    sym: bool = False, 
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Compute the local p-homophily for each node in the graph.
    
    Local homophily measures how similar a node's features are to its p-hop neighbors
    with respect to class labels.
    
    Args:
        p: The 'order' of homophily (number of hops)
        g: Input graph
        y: Node labels of shape (n_nodes,), if None will use g.ndata['label']
        self_loops: Whether to include self-loops in adjacency
        do_hp: Whether to compute higher-order polynomial version (I - A)
        fix_d: If True, row-normalize adjacency (D^{-1}A)
        sym: Whether to symmetrize adjacency (A <- A + A^T)
        device: Device to perform computation on
        
    Returns:
        torch.Tensor: The local homophily scores for each node
    """
    device = torch.device(device)

    # Get labels
    if y is None:
        y = g.ndata['label']
    y = y.to(device)

    # 1) Build adjacency (sparse)
    A = build_sparse_adj_matrix(g, self_loops=self_loops, sym=sym, device=device)

    # 2) Normalize adjacency
    A = normalize_sparse_adj(A)

    if do_hp:
        # Create sparse identity matrix
        I = torch.sparse_coo_tensor(
            indices=torch.arange(A.size(0)).repeat(2, 1),
            values=torch.ones(A.size(0)),
            size=A.size()
        )
        # Low pass filter: I - A
        A = I - A

    # 3) Build label matrix M
    if len(y.shape) == 1:
        M, classes = compute_label_matrix(y, device=device)
    else:
        M = y.to(device)

    # 4) Compute S = (A^p) M  (shape: n x C)
    S = power_adj_times_matrix(A, M, p)

    # 5) local_homophily(i) = sum_{c} S[i,c]^2
    homophily_scores = (S**2).sum(dim=1)

    return homophily_scores.detach().cpu()


def local_autophily(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> np.ndarray:
    """
    Compute the local autophily for each node in the graph.
    
    Autophily measures how similar a node is to itself through its neighborhood,
    regardless of class labels.
    
    Args:
        p: The order of the local autophily
        g: Input graph
        self_loops: Whether to include self-loops in the adjacency matrix
        fix_d: Whether to fix the degree distribution by normalizing
        sym: Whether to symmetrize the adjacency matrix
        device: Device to perform computations on
        
    Returns:
        np.ndarray: An array containing the local autophily scores for each node
    """
    device = torch.device(device)
    # Compute A_hat^p
    A_hat_p = get_A_hat_p(p, g, self_loops=self_loops, fix_d=fix_d, sym=sym, device=device)
    
    # Compute sum_j A_hat_p[i,j]^2
    local_autophily = (A_hat_p ** 2).sum(dim=1)

    return local_autophily.cpu().numpy()


def local_total_connectivity(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> np.ndarray:
    """
    Compute the local total connectivity for each node in the graph.
    
    Total connectivity measures how well connected a node is to its p-hop neighborhood.
    
    Args:
        p: The order of the local connectivity
        g: Input graph
        self_loops: Whether to include self-loops in the adjacency matrix
        fix_d: Whether to fix the degree distribution by normalizing
        sym: Whether to symmetrize the adjacency matrix
        device: Device to perform computations on
        
    Returns:
        np.ndarray: An array containing the local total connectivity scores for each node
    """
    device = torch.device(device)
    # Compute A_hat^p
    A_hat_p = get_A_hat_p(p, g, self_loops=self_loops, fix_d=fix_d, sym=sym, device=device)
    
    # Compute (sum_j A_hat_p[i,j])^2
    sum_A = A_hat_p.sum(dim=1)
    local_total_connectivity = sum_A ** 2

    return local_total_connectivity.cpu().numpy()
