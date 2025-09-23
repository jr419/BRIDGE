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


def class_bottlenecking_score(
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
    Compute the local class-bottlenecking score for each node in the graph over p hops.

    The class-bottlenecking score measures how strongly a node connects to
    same-class nodes within p hops (with respect to class labels).

    Args:
        p: The hop order (number of hops)
        g: Input graph
        y: Node labels of shape (n_nodes,), if None will use g.ndata['label']
        self_loops: Whether to include self-loops in adjacency
        do_hp: Whether to compute higher-order polynomial version (I - A)
        fix_d: If True, row-normalize adjacency (D^{-1}A)
        sym: Whether to symmetrize adjacency (A <- A + A^T)
        device: Device to perform computation on

    Returns:
        torch.Tensor: The class-bottlenecking scores for each node
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

    # 5) class-bottlenecking score(i) = sum_{c} S[i,c]^2
    homophily_scores = (S**2).sum(dim=1)

    return homophily_scores.detach().cpu()


def self_bottlenecking_score(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    do_hp: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> np.ndarray:
    """
    Compute the self bottlenecking score (local self-connectivity) for each node.

    Self-connectivity measures how strongly a node reconnects to itself through
    its neighborhood, regardless of class labels.

    Args:
        p: The order for the self bottlenecking score
        g: Input graph
        self_loops: Whether to include self-loops in the adjacency matrix
        do_hp : Whether to compute higher-order polynomial version (I - A)
        fix_d: Whether to fix the degree distribution by normalizing
        sym: Whether to symmetrize the adjacency matrix
        device: Device to perform computations on

    Returns:
        np.ndarray: An array containing the self bottlenecking scores (self-connectivity) for each node
    """
    device = torch.device(device)

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
    M = torch.eye(A.size(0), device=device)  # Identity matrix for self-connectivity

    # 4) Compute S = (A^p)  (shape: n x C)
    S = power_adj_times_matrix(A, M, p)

    # 5) self bottlenecking score(i) =  S[i,i]^2
    autophily_scores = S.diag()**2

    return autophily_scores.detach().cpu()


def total_bottlenecking_score(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    do_hp: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> np.ndarray:
    """
    Compute the total bottlenecking score for each node in the graph.

    The total bottlenecking score measures how well a node connects within its
    p-hop neighborhood overall.

    Args:
        p: The order for the total bottlenecking score
        g: Input graph
        self_loops: Whether to include self-loops in the adjacency matrix
        do_hp: Whether to compute higher-order polynomial version (I - A)
        fix_d: Whether to fix the degree distribution by normalizing
        sym: Whether to symmetrize the adjacency matrix
        device: Device to perform computations on

    Returns:
        np.ndarray: An array containing the total bottlenecking scores for each node
    """
    device = torch.device(device)

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
    M = torch.eye(A.size(0), device=device)  # Identity matrix for total bottlenecking

    # 4) Compute S = (A^p)  (shape: n x C)
    S = power_adj_times_matrix(A, M, p)

    # 5) total bottlenecking score(i) = sum_{j} S[i,j]^2
    total_connectivity_scores = (S**2).sum(dim=1)

    return total_connectivity_scores.detach().cpu()

# Backward-compatibility aliases
local_homophily = class_bottlenecking_score
local_autophily = self_bottlenecking_score
local_total_connectivity = total_bottlenecking_score
