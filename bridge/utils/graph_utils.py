"""
Utility functions for graph operations.

This module provides functions for manipulating and analyzing graphs.
"""

import torch
import dgl
import numpy as np
import random
from scipy.sparse import csr_matrix
from typing import Tuple, List, Dict, Union, Optional, Any


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all relevant libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def check_symmetry(g: dgl.DGLGraph) -> bool:
    """
    Check if a DGL graph is symmetric (undirected).
    
    Args:
        g: DGL graph
        
    Returns:
        bool: True if graph is symmetric, False otherwise
    """
    # Get adjacency matrix
    A = g.adjacency_matrix().to_dense()
    # Check if A equals its transpose
    return torch.allclose(A, A.t(), atol=1e-8)


def make_symmetric(g: dgl.DGLGraph, sym_type: str = 'both') -> dgl.DGLGraph:
    """
    Make a DGL graph symmetric by adding reverse edges.
    
    Args:
        g: DGL graph to symmetrize
        sym_type: Symmetrization type:
            'both': Add all reverse edges
            'upper': Reflect upper triangular part
            'lower': Reflect lower triangular part
    
    Returns:
        dgl.DGLGraph: New symmetric DGL graph
    """
    # Get all edges
    u, v = g.edges()
    
    if sym_type == 'both':
        # Create bi-directional edges by adding all reverse edges
        u_new = torch.cat([u, v])
        v_new = torch.cat([v, u])
    
    else:
        # Create masks for upper and lower triangular parts
        upper_mask = u < v
        lower_mask = u > v
        
        if sym_type == 'upper':
            # Keep upper triangular edges and reflect them
            u_keep = u[upper_mask]
            v_keep = v[upper_mask]
            # Add reverse edges for upper triangular part
            u_new = torch.cat([u_keep, v_keep])
            v_new = torch.cat([v_keep, u_keep])
            
        elif sym_type == 'lower':
            # Keep lower triangular edges and reflect them
            u_keep = u[lower_mask]
            v_keep = v[lower_mask]
            # Add reverse edges for lower triangular part
            u_new = torch.cat([u_keep, v_keep])
            v_new = torch.cat([v_keep, u_keep])
            
        else:
            raise ValueError("sym_type must be one of: 'both', 'upper', 'lower'")
    
    # Create new graph with symmetrized edges
    new_g = dgl.graph((u_new, v_new), num_nodes=g.num_nodes())
    
    # Copy node features
    for key in g.ndata.keys():
        new_g.ndata[key] = g.ndata[key]
    
    # Copy edge features if they exist
    for key in g.edata.keys():
        if sym_type == 'both':
            # Duplicate all edge features
            new_g.edata[key] = torch.cat([g.edata[key], g.edata[key]])
        else:
            # Only duplicate features for the kept edges
            if sym_type == 'upper':
                kept_features = g.edata[key][upper_mask]
            else:  # lower
                kept_features = g.edata[key][lower_mask]
            new_g.edata[key] = torch.cat([kept_features, kept_features])
    
    return new_g


def homophily(g: dgl.DGLGraph) -> float:
    """
    Compute the homophily score of a graph.
    
    Homophily is defined as the fraction of edges that connect nodes of the same class.
    
    Args:
        g: Input graph with node labels
        
    Returns:
        float: Homophily score between 0 and 1
    """
    A = g.adjacency_matrix().to_dense()
    y = g.ndata['label']
    h = sum([(A[y==i][:,y==i]).sum() for i in range(g.ndata['label'].max()+1)])/(A.sum())
    return h


def build_sparse_adj_matrix(
    g: dgl.DGLGraph,
    self_loops: bool = False,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Build a sparse adjacency matrix from a DGL graph.
    
    Args:
        g: Input DGL graph
        self_loops: Whether to include self-loops in the adjacency matrix
        sym: Whether to symmetrize the adjacency matrix
        device: Device to place the output tensor
        
    Returns:
        torch.Tensor: Sparse adjacency matrix of shape (n_nodes, n_nodes)
    """
    device = torch.device(device)
    n = g.num_nodes()
    
    # Get edges
    src, dst = g.edges()
    indices = torch.stack([src, dst], dim=0).to(device)

    # Build initial adjacency in sparse format
    values = torch.ones(indices.shape[1], device=device)
    A = torch.sparse_coo_tensor(indices, values, (n, n), device=device)

    # Add self-loops
    if self_loops:
        loop_index = torch.arange(n, device=device)
        loop_index = torch.stack([loop_index, loop_index], dim=0)
        loop_vals = torch.ones(n, device=device)
        A = add_sparse(A, loop_index, loop_vals)

    # Symmetrize if desired: A <- A + A^T (or average)
    if sym:
        A_T = torch.sparse_coo_tensor(
            torch.stack([dst, src], dim=0).to(device), 
            values, (n, n), device=device
        )
        A = sparse_add(A, A_T)
        # If you want to average them: A = 0.5 * A

    A = A.coalesce()  # Make sure it's in a coalesced form
    return A


def add_sparse(
    A: torch.Tensor,
    new_indices: torch.Tensor,
    new_values: torch.Tensor
) -> torch.Tensor:
    """
    Add new indices and values to a sparse COO tensor.
    
    Args:
        A: Input sparse tensor
        new_indices: New indices to add
        new_values: New values to add
        
    Returns:
        torch.Tensor: Updated sparse tensor
    """
    # Combine old and new
    indices = torch.cat([A.indices(), new_indices], dim=1)
    values = torch.cat([A.values(), new_values])
    # Build new coalesced tensor
    A_new = torch.sparse_coo_tensor(indices, values, A.shape, device=A.device)
    return A_new.coalesce()


def sparse_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Add two sparse COO tensors.
    
    Args:
        A: First sparse tensor
        B: Second sparse tensor
        
    Returns:
        torch.Tensor: Sum of the two sparse tensors
    """
    indices = torch.cat([A.indices(), B.indices()], dim=1)
    values = torch.cat([A.values(), B.values()])
    C = torch.sparse_coo_tensor(indices, values, A.shape, device=A.device).coalesce()
    return C


def normalize_sparse_adj(A: torch.Tensor) -> torch.Tensor:
    """
    Normalize a sparse adjacency matrix.
    
    This performs D^{-1/2}AD^{-1/2} normalization where D is the degree matrix.
    
    Args:
        A: Input sparse adjacency matrix
        
    Returns:
        torch.Tensor: Normalized sparse adjacency matrix
    """
    # Convert to COO format for easy manipulation
    if not A.is_coalesced():
        A = A.coalesce()
    
    # Get indices and values
    indices = A.indices()
    values = A.values()
    n = A.shape[0]
    
    # Get row and column indices
    row_indices = indices[0]
    col_indices = indices[1]
    
    # Compute out-degrees (row sums)
    out_degree = torch.zeros(n, device=A.device)
    out_degree.index_add_(0, row_indices, values)
    
    # Compute in-degrees (column sums)
    in_degree = torch.zeros(n, device=A.device)
    in_degree.index_add_(0, col_indices, values)
    
    # Handle zero degrees (isolated nodes)
    out_degree[out_degree == 0] = 1.0
    in_degree[in_degree == 0] = 1.0
    
    # D_out^-1/2 A D_in^-1/2
    d_out_inv_sqrt = torch.pow(out_degree, -0.5)
    d_in_inv_sqrt = torch.pow(in_degree, -0.5)
    
    # Vectorized normalization using both in and out degrees
    new_values = values * d_out_inv_sqrt[row_indices] * d_in_inv_sqrt[col_indices]
    
    # Create new sparse tensor with normalized values
    return torch.sparse_coo_tensor(indices, new_values, A.shape)


def get_A_hat_p(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Compute the normalized adjacency matrix raised to the power p.

    Args:
        p: The power to which the normalized adjacency matrix is raised
        g: Input graph
        self_loops: Whether to include self-loops in the adjacency matrix
        fix_d: Whether to fix the degree distribution by normalizing
        sym: Whether to symmetrize the adjacency matrix
        device: Device to perform computations on

    Returns:
        torch.Tensor: The normalized adjacency matrix raised to the power p
    """
    device = torch.device(device)
    n = g.number_of_nodes()

    # Obtain adjacency matrix as a dense tensor
    A = g.cpu().adjacency_matrix().to_dense()

    if self_loops:
        A = A + torch.eye(n, device='cpu')

    if sym:
        A = torch.triu(A, diagonal=1)
        A = A + A.t()

    d_arr = A.sum(1)
    A[d_arr==0,:][:,d_arr==0]=torch.eye(n)[d_arr==0,:][:,d_arr==0]
    d_arr[d_arr==0] = 1
    d_inv_arr = (1/d_arr**(1/2))
    A_hat = csr_matrix(d_inv_arr[:,None]*A*d_inv_arr[None,:])
    A_hat_p = torch.tensor((A_hat**p).todense(),device=device)

    return A_hat_p.to(device)


def get_A_p(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Compute the adjacency matrix raised to the power p.

    Args:
        p: The power to which the adjacency matrix is raised
        g: Input graph
        self_loops: Whether to include self-loops in the adjacency matrix
        fix_d: Placeholder for compatibility; not used here
        sym: Whether to symmetrize the adjacency matrix
        device: Device to perform computations on

    Returns:
        torch.Tensor: The adjacency matrix raised to the power p
    """
    device = torch.device(device)
    n = g.number_of_nodes()

    # Obtain adjacency matrix as a dense tensor
    A = g.adjacency_matrix().to_dense().to(device)

    if self_loops:
        A = A + torch.eye(n, device=device)

    if sym:
        A = torch.triu(A, diagonal=1)
        A = A + A.t()

    if p == 1:
        A_p = A
    else:
        A_p = torch.matrix_power(A, p)

    return A_p
