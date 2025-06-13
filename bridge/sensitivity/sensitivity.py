"""
Sensitivity analysis for Graph Neural Networks.

This module provides functions for computing the sensitivity of graph neural networks
to various types of input perturbations, including signal, noise, and global variations.
These sensitivity measures are key to understanding the Signal-to-Noise Ratio of GNNs.
"""

import torch
import torch.nn as nn
import dgl
from typing import Dict, Tuple, List, Union, Literal, Optional
from torch.autograd.functional import jacobian


def estimate_sensitivity_analytic(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    labels: torch.Tensor, 
    sensitivity_type: Literal["signal", "noise", "global"]
) -> torch.Tensor:
    """
    Estimate the sensitivity for a *Linear* GCN analytically (no autograd).
    
    This function computes the sensitivity matrix for a linear GCN model analytically,
    based on the model weights and graph structure, without requiring autograd.
    
    For a linear GCN, we derive that for h[i,p] = sum_{j,q} A[i,j] X[j,q] W[q,p],
    the partial derivative d h[i,p]/ d X[m,n] = A[i,m] * W[n,p].
    
    Args:
        model: A linear GCN model with a weight attribute
        graph: The input graph
        labels: Node labels (used for signal sensitivity)
        sensitivity_type: Type of sensitivity to compute:
            - "signal": Sensitivity to coherent class-specific changes
            - "noise": Sensitivity to unstructured, IID noise
            - "global": Sensitivity to global shifts in the input features
            
    Returns:
        A sensitivity tensor of shape [num_nodes, num_classes, in_feats, in_feats]
    """
    # Extract the learned weight matrix W: shape [in_feats, out_feats]
    W = model.weight.detach()  # [in_feats, out_feats]
    in_feats, num_classes = W.shape
    n = graph.number_of_nodes()
    
    # Prepare normalized adjacency
    A = graph.adj().to_dense()
    d_arr = A.sum(1)
    A[d_arr==0,:][:,d_arr==0] = torch.eye(n)[d_arr==0,:][:,d_arr==0]
    d_arr[d_arr==0] = 1
    d_inv_arr = (1/d_arr**(1/2))
    A = d_inv_arr[:,None]*A*d_inv_arr[None,:]
    num_nodes = A.shape[0]
    
    # Precompute row sums, row sums of squares
    row_sums = A.sum(dim=1)         # shape [num_nodes]
    row_sums_sq = (A ** 2).sum(1)   # shape [num_nodes]
    
    # For "signal": sum_{j,k : labels[j]==labels[k]} A[i,j]*A[i,k].
    # We'll group node sets by label. For each label L:
    #   v_i[L] = sum_{j in L} A[i,j]
    # Then C_signal[i] = sum_{L} (v_i[L])^2
    unique_labels = labels.unique()
    v_per_label = []
    for lbl in unique_labels:
        mask_lbl = (labels == lbl).float()  # shape [num_nodes]
        # v_lbl[i] = sum_{j in L} A[i,j]
        v_lbl = (A * mask_lbl.unsqueeze(0)).sum(dim=1)
        v_per_label.append(v_lbl)
    # => shape [num_labels, num_nodes], transpose => [num_nodes, num_labels]
    v_per_label = torch.stack(v_per_label, dim=0).T
    # C_signal[i] = sum_{L} v_per_label[i, L]^2
    C_signal = (v_per_label ** 2).sum(dim=1)

    # We'll build a result [num_nodes, num_classes, in_feats, in_feats].
    sensitivity_matrix = torch.zeros(num_nodes, num_classes, in_feats, in_feats, dtype=torch.double)

    if sensitivity_type == "signal":
        # sum_{i} sum_{labels[j]==labels[k]} (A[i,j]*W[q,p])*(A[i,k]*W[r,p])
        # => sum_{i} W[q,p]*W[r,p]*C_signal[i]
        for p in range(num_classes):
            for q in range(in_feats):
                for r in range(in_feats):
                    sensitivity_matrix[:,p,q,r] = W[q,p] * W[r,p] * C_signal

    elif sensitivity_type == "noise":
        # sum_{i} sum_{j} A[i,j]^2 * W[q,p]*W[r,p]
        for p in range(num_classes):
            for q in range(in_feats):
                for r in range(in_feats):
                    sensitivity_matrix[:,p,q,r] = W[q,p] * W[r,p] * row_sums_sq

    elif sensitivity_type == "global":
        # sum_{i} ( sum_j A[i,j] )^2 * W[q,p]*W[r,p]
        for p in range(num_classes):
            for q in range(in_feats):
                for r in range(in_feats):
                    sensitivity_matrix[:,p,q,r] = W[q,p] * W[r,p] * (row_sums**2)

    else:
        raise ValueError(f"Unknown sensitivity_type={sensitivity_type}")

    return sensitivity_matrix


def compute_jacobian(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    x: torch.Tensor, 
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the model with respect to the input.
    
    The Jacobian J has the form:
        J[i, j, k, l] = d (model(graph, x)[i, j]) / d x[k, l]
    where:
      - i ranges over number of nodes       (N)
      - j ranges over output features       (out_feats)
      - k ranges over number of nodes       (N)
      - l ranges over input features        (in_feats)
    
    Args:
        model: The neural network model
        graph: The input graph
        x: Input features of shape (N, in_feats)
        device: Device to compute on ("cuda" or "cpu")
        
    Returns:
        A tensor of shape (N, out_feats, N, in_feats) containing
        the Jacobian for all outputs w.r.t. all inputs
    """
    x = x.to(device)
    graph = graph.to(device)
    model = model.to(device)

    # Define a 'forward function' that takes only the input tensor x
    # and returns the model output of shape (N, out_feats).
    def forward_func(x_):
        return model(graph, x_)

    # Use PyTorch's autograd.functional.jacobian to compute the full Jacobian
    x.requires_grad_(True)
    jac = jacobian(forward_func, x, create_graph=False, strict=False)

    # Detach the Jacobian to save memory
    jac = jac.detach()

    # Reset requires_grad if needed
    x.requires_grad_(False)

    return jac


def estimate_sensitivity_autograd(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    in_feats: int,
    labels: torch.Tensor, 
    sensitivity_type: Literal["signal", "noise", "global"], 
    device: str = "cuda"
) -> torch.Tensor:
    """
    Estimate sensitivity using autograd-computed Jacobian.
    
    This function computes the sensitivity matrix for any differentiable model
    using PyTorch's automatic differentiation. It handles different sensitivity
    types based on the paper's definitions.
    
    Args:
        model: The neural network model
        graph: The input graph
        in_feats: Number of input features
        labels: Node labels (used for signal sensitivity)
        sensitivity_type: Type of sensitivity to compute:
            - "signal": Sensitivity to coherent class-specific changes
            - "noise": Sensitivity to unstructured, IID noise
            - "global": Sensitivity to global shifts in the input features
        device: Device to compute on ("cuda" or "cpu")
        
    Returns:
        A sensitivity tensor of shape [num_nodes, num_classes, in_feats, in_feats]
    """
    model.eval()
    N = graph.number_of_nodes()
    out_feats = labels.unique().shape[0]
    
    # Compute Jacobian at zero input
    x = torch.zeros(N, in_feats, device=device)
    jac = compute_jacobian(model, graph, x, device)  # shape: [N, out_feats, N, in_feats]
    
    # Initialize sensitivity matrix
    sensitivity = torch.zeros(N, out_feats, in_feats, in_feats, device=device)
    
    # Compute different types of sensitivity
    if sensitivity_type == "signal":
        # Signal sensitivity: measures response to coherent class-specific changes
        for i in range(N):
            for label in torch.unique(labels):
                mask = (labels == label)
                jac_filtered = jac[i, :, mask, :]  # shape: [out_feats, num_nodes_in_class, in_feats]
                
                # Take outer product for each output feature
                for out_idx in range(out_feats):
                    jac_out = jac_filtered[out_idx]  # shape: [num_nodes_in_class, in_feats]
                    jac_sum = jac_out.sum(dim=0)    # shape: [in_feats]
                    sensitivity[i, out_idx] += torch.outer(jac_sum, jac_sum)
    
    elif sensitivity_type == "noise":
        # Noise sensitivity: measures response to unstructured, IID noise
        for i in range(N):
            for out_idx in range(out_feats):
                jac_out = jac[i, out_idx]  # shape: [N, in_feats]
                sensitivity[i, out_idx] = jac_out.T @ jac_out

    elif sensitivity_type == "global":
        # Global sensitivity: measures response to global shifts in the input
        for i in range(N):
            for out_idx in range(out_feats):
                jac_out = jac[i, out_idx]  # shape: [N, in_feats]
                jac_sum = jac_out.sum(dim=0)  # shape: [in_feats]
                sensitivity[i, out_idx] = torch.outer(jac_sum, jac_sum)
    
    else:
        raise ValueError(f"Unknown sensitivity_type={sensitivity_type}")
    
    return sensitivity
