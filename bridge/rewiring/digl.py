"""
DIGL (Diffusion Improves Graph Learning) rewiring implementation.

Based on: Klicpera, J., Weißenberger, S., & Günnemann, S. (2019). 
Diffusion improves graph learning. NeurIPS.
"""

import torch
import dgl
import numpy as np
from typing import Tuple, Optional, Union


def compute_ppr_matrix(g: dgl.DGLGraph, alpha: float = 0.15, eps: float = 1e-6, max_iter: int = 100) -> torch.Tensor:
    """
    Compute the Personalized PageRank (PPR) matrix using power iteration.
    
    Args:
        g: Input DGL graph
        alpha: Teleport probability (restart probability)
        eps: Convergence threshold
        max_iter: Maximum number of iterations
        
    Returns:
        torch.Tensor: PPR matrix of shape (n_nodes, n_nodes)
    """
    n = g.number_of_nodes()
    device = g.device
    
    # Get adjacency matrix and compute transition matrix
    A = g.adjacency_matrix().to_dense().float().to(device)
    degrees = g.out_degrees().float().to(device)
    
    # Avoid division by zero
    degrees[degrees == 0] = 1.0
    
    # Row-normalized adjacency matrix (transition matrix)
    P = A / degrees.unsqueeze(1)
    
    # Initialize PPR matrix (each node's PPR vector)
    PPR = torch.eye(n, device=device)
    
    # Power iteration for each source node
    for _ in range(max_iter):
        PPR_old = PPR.clone()
        PPR = alpha * torch.eye(n, device=device) + (1 - alpha) * torch.matmul(P.T, PPR)
        
        # Check convergence
        if torch.max(torch.abs(PPR - PPR_old)) < eps:
            break
    
    return PPR


def compute_heat_kernel(g: dgl.DGLGraph, t: float = 5.0, method: str = 'exact') -> torch.Tensor:
    """
    Compute the heat kernel matrix.
    
    Args:
        g: Input DGL graph
        t: Diffusion time parameter
        method: 'exact' for eigendecomposition, 'approx' for approximation
        
    Returns:
        torch.Tensor: Heat kernel matrix of shape (n_nodes, n_nodes)
    """
    n = g.number_of_nodes()
    device = g.device
    
    # Get normalized Laplacian
    A = g.adjacency_matrix().to_dense().float().to(device)
    degrees = g.out_degrees().float().to(device)
    degrees[degrees == 0] = 1.0
    
    # Compute normalized Laplacian L = I - D^(-1/2) * A * D^(-1/2)
    D_sqrt_inv = torch.diag(1.0 / torch.sqrt(degrees))
    L = torch.eye(n, device=device) - torch.matmul(torch.matmul(D_sqrt_inv, A), D_sqrt_inv)
    
    if method == 'exact' and n < 1000:  # Use exact method for small graphs
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # Heat kernel: H = U * exp(-t * Lambda) * U^T
        H = torch.matmul(
            torch.matmul(eigenvectors, torch.diag(torch.exp(-t * eigenvalues))),
            eigenvectors.T
        )
    else:  # Use approximation for large graphs
        # Taylor approximation: exp(-tL) ≈ sum_{k=0}^K (-t)^k L^k / k!
        K = 10  # Number of terms
        H = torch.eye(n, device=device)
        L_power = torch.eye(n, device=device)
        
        for k in range(1, K + 1):
            L_power = torch.matmul(L_power, L)
            H = H + ((-t) ** k / np.math.factorial(k)) * L_power
    
    return H


def digl_rewired(
    g: dgl.DGLGraph,
    diffusion_type: str = 'ppr',
    alpha: float = 0.15,
    t: float = 5.0,
    epsilon: float = 0.01,
    add_ratio: float = 1.0,
    remove_ratio: float = 1.0,
    self_loops: bool = False,
    symmetric: bool = True
) -> dgl.DGLGraph:
    """
    Apply DIGL (Diffusion Improves Graph Learning) rewiring to a graph.
    
    Args:
        g: Input DGL graph
        diffusion_type: Type of diffusion ('ppr' for PageRank, 'heat' for heat kernel)
        alpha: PPR teleport probability (only for PPR)
        t: Heat kernel time parameter (only for heat kernel)
        epsilon: Threshold for edge addition
        add_ratio: Maximum ratio of edges to add
        remove_ratio: Maximum ratio of edges to remove
        self_loops: Whether to include self-loops
        symmetric: Whether to maintain graph symmetry
        
    Returns:
        dgl.DGLGraph: Rewired graph
    """
    n = g.number_of_nodes()
    device = g.device
    
    # Compute diffusion matrix
    if diffusion_type == 'ppr':
        S = compute_ppr_matrix(g, alpha=alpha)
    elif diffusion_type == 'heat':
        S = compute_heat_kernel(g, t=t)
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")
    
    # Get current adjacency matrix
    A = g.adjacency_matrix().to_dense().float().to(device)
    
    # Remove self-loops from consideration if not wanted
    if not self_loops:
        S.fill_diagonal_(0)
        A.fill_diagonal_(0)
    
    # Make symmetric if required
    if symmetric:
        S = (S + S.T) / 2
    
    # Identify candidate edges to add (high diffusion score, not in graph)
    non_edges_mask = (A == 0)
    if not self_loops:
        non_edges_mask.fill_diagonal_(False)
    
    candidate_scores = S * non_edges_mask.float()
    
    # Select top edges to add based on diffusion scores
    num_edges_to_add = int(add_ratio * g.number_of_edges())
    if num_edges_to_add > 0:
        # Flatten and get top-k
        flat_scores = candidate_scores.flatten()
        top_k_values, top_k_indices = torch.topk(flat_scores, min(num_edges_to_add, flat_scores.numel()))
        
        # Filter by epsilon threshold
        valid_mask = top_k_values > epsilon
        top_k_indices = top_k_indices[valid_mask]
        
        # Convert back to 2D indices
        add_edges_i = top_k_indices // n
        add_edges_j = top_k_indices % n
    else:
        add_edges_i = torch.tensor([], dtype=torch.long, device=device)
        add_edges_j = torch.tensor([], dtype=torch.long, device=device)
    
    # Identify edges to remove (low diffusion score, in graph)
    edges_mask = (A > 0)
    if not self_loops:
        edges_mask.fill_diagonal_(False)
    
    edge_scores = S * edges_mask.float()
    
    # Select bottom edges to remove based on diffusion scores
    num_edges_to_remove = int(remove_ratio * g.number_of_edges())
    if num_edges_to_remove > 0:
        # Get edges and their scores
        edge_indices = edges_mask.nonzero()
        edge_score_values = edge_scores[edge_indices[:, 0], edge_indices[:, 1]]
        
        # Get bottom-k
        bottom_k_values, bottom_k_indices = torch.topk(
            edge_score_values, 
            min(num_edges_to_remove, edge_score_values.numel()), 
            largest=False
        )
        
        # Filter by epsilon threshold
        valid_mask = bottom_k_values < epsilon
        bottom_k_indices = bottom_k_indices[valid_mask]
        
        # Get edges to remove
        remove_edges = edge_indices[bottom_k_indices]
        remove_edges_i = remove_edges[:, 0]
        remove_edges_j = remove_edges[:, 1]
    else:
        remove_edges_i = torch.tensor([], dtype=torch.long, device=device)
        remove_edges_j = torch.tensor([], dtype=torch.long, device=device)
    
    # Build new adjacency matrix
    A_new = A.clone()
    
    # Remove edges
    if remove_edges_i.numel() > 0:
        A_new[remove_edges_i, remove_edges_j] = 0
        if symmetric:
            A_new[remove_edges_j, remove_edges_i] = 0
    
    # Add edges
    if add_edges_i.numel() > 0:
        A_new[add_edges_i, add_edges_j] = 1
        if symmetric:
            A_new[add_edges_j, add_edges_i] = 1
    
    # Create new graph
    edges = A_new.nonzero()
    new_g = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=n, device=device)
    
    # Copy node features
    for key in g.ndata.keys():
        new_g.ndata[key] = g.ndata[key]
    
    return new_g