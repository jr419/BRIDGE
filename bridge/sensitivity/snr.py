"""
Signal-to-Noise Ratio (SNR) estimation for Graph Neural Networks.

This module provides methods for estimating the Signal-to-Noise Ratio (SNR)
of Graph Neural Networks through Monte Carlo simulations and theoretical approaches.
The SNR is a crucial metric for understanding when and how well graph neural networks
can discriminate between different classes.
"""

import torch
import torch.nn as nn
import dgl
from typing import Callable, Dict, Tuple, List, Union, Optional
import numpy as np
from tqdm import tqdm

from .sensitivity import estimate_sensitivity_analytic, estimate_sensitivity_autograd


def estimate_snr_monte_carlo(
    model: nn.Module,
    graph: dgl.DGLGraph,
    in_feats: int,
    labels: torch.Tensor,
    num_montecarlo_simulations: int,
    feature_generator: Callable,
    device: str = "cpu",
    inner_samples: int = 100,
    split_model_input_size: int = 100
) -> torch.Tensor:
    """
    Estimate the Signal-to-Noise Ratio (SNR) of an MPNN's outputs via Monte Carlo simulation.
    
    This function estimates SNR by:
    1. Generating multiple realizations of class means (mu)
    2. For each mu, generating multiple feature samples with the same mu but different noise/global shifts
    3. Computing empirical mean and variance across these samples
    4. Computing SNR = var_across_mu(mean) / mean_across_mu(var)
    
    Args:
        model: The neural network model
        graph: The input graph
        in_feats: Number of input features
        labels: Node labels
        num_montecarlo_simulations: Number of outer loop iterations (different mu samples)
        feature_generator: Function to generate features with signature:
                        feature_generator(num_nodes, in_feats, labels, num_mu_samples)
        device: Device to compute on
        inner_samples: Number of feature samples for each mu
        split_model_input_size: Maximum batch size for processing samples
        
    Returns:
        A tensor of shape [num_nodes, out_feats] containing the estimated SNR
        for each node and output feature
    """
    model.eval().to(device)
    
    num_nodes = labels.shape[0]
    out_feats = labels.unique().shape[0]

    # Allocate arrays for storing conditional means & variances
    # shape => [num_nodes, out_feats, num_montecarlo_simulations]
    H_mean = torch.zeros(num_nodes, out_feats, num_montecarlo_simulations, device=device)
    H_var = torch.zeros(num_nodes, out_feats, num_montecarlo_simulations, device=device)

    # Generate features for all simulations at once
    feats_4d_stacked_feats = feature_generator(
        num_nodes=num_nodes,
        in_feats=in_feats*num_montecarlo_simulations,
        labels=labels,
        num_mu_samples=inner_samples
    ).to(device)
    
    for outer_idx in tqdm(range(num_montecarlo_simulations), desc="MC SNR Estimation"):
        # Extract features for this simulation
        feats_3d = feats_4d_stacked_feats[:, outer_idx*in_feats:(1+outer_idx)*in_feats, :].double()

        with torch.no_grad():
            # model(...) => [num_nodes, out_feats, inner_samples] (for LinearGCN)
            H_3d = model(graph, feats_3d)

        # Compute mean & var over inner_samples dimension (dim=2)
        H_cond_mean = H_3d.mean(dim=2)  # => [num_nodes, out_feats]
        H_cond_var = H_3d.var(dim=2)   # => [num_nodes, out_feats]

        # Store the results
        H_mean[:, :, outer_idx] = H_cond_mean
        H_var[:, :, outer_idx] = H_cond_var

    # Compute:
    # 1. Variance across mu samples for the means (numerator)
    # 2. Mean across mu samples for the variances (denominator)
    var_across_mu = H_mean.var(dim=-1)
    mean_cond_var = H_var.mean(dim=-1)
    
    # Handle potential division by zero
    eps = 1e-8
    mean_cond_var = torch.clamp(mean_cond_var, min=eps)

    # Return SNR for each node and feature
    snr_estimate = var_across_mu / mean_cond_var
    return snr_estimate


def estimate_snr_theorem(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    labels: torch.Tensor, 
    sigma_intra: torch.Tensor, 
    sigma_inter: torch.Tensor, 
    tau: torch.Tensor, 
    eta: torch.Tensor
) -> torch.Tensor:
    """
    Compute SNR estimate using the theoretical formula from the paper.
    
    This function implements Theorem 1 from the paper, which relates
    the SNR to signal, global, and noise sensitivities, weighted by
    covariance matrices of the input features.
    
    Args:
        model: The neural network model (must be LinearGCN)
        graph: The input graph
        labels: Node labels
        sigma_intra: Intra-class covariance matrix
        sigma_inter: Inter-class covariance matrix
        tau: Global shift covariance matrix
        eta: Noise covariance matrix
        
    Returns:
        A tensor of shape [num_nodes] containing the estimated SNR
        for each node (averaged across output features)
    """
    in_feats = model.weight.shape[0]
    out_size = model.weight.shape[1]
    
    # Compute sensitivities
    sigsen = estimate_sensitivity_analytic(model, graph, labels, "signal")
    noisesen = estimate_sensitivity_analytic(model, graph, labels, "noise")
    globsen = estimate_sensitivity_analytic(model, graph, labels, "global")

    # Compute numerator and denominator of the SNR formula
    numerator = torch.zeros(graph.number_of_nodes(), out_size)
    for q in range(in_feats):
        for r in range(in_feats):
            numerator += (
                (sigma_intra[q,r].item() - sigma_inter[q,r].item()) * sigsen[:,:,q,r]
                + sigma_inter[q,r].item() * globsen[:,:,q,r]
            )

    denominator = torch.zeros(graph.number_of_nodes(), out_size)
    for q in range(in_feats):
        for r in range(in_feats):
            denominator += (
                tau[q,r].item() * globsen[:,:,q,r]
                + eta[q,r].item() * noisesen[:,:,q,r]
            )
    
    # Prevent division by zero
    denominator[denominator < 1e-8] = 1e-8

    # Average SNR across output features
    return torch.mean(numerator / denominator, dim=1)


def estimate_snr_theorem_autograd(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    in_feats: int,
    labels: torch.Tensor, 
    sigma_intra: torch.Tensor, 
    sigma_inter: torch.Tensor, 
    tau: torch.Tensor, 
    eta: torch.Tensor, 
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute SNR estimate using the theoretical formula with autograd sensitivities.
    
    This is similar to estimate_snr_theorem, but uses automatic differentiation
    to compute the sensitivities, making it applicable to non-linear models.
    
    Args:
        model: The neural network model (any model that works with autograd)
        graph: The input graph
        in_feats: Number of input features
        labels: Node labels
        sigma_intra: Intra-class covariance matrix
        sigma_inter: Inter-class covariance matrix
        tau: Global shift covariance matrix
        eta: Noise covariance matrix
        device: Device to compute on
        
    Returns:
        A tensor of shape [num_nodes] containing the estimated SNR
        for each node (averaged across output features)
    """
    # Compute sensitivities using autograd
    sigsen = estimate_sensitivity_autograd(model, graph, in_feats, labels, "signal", device)
    noisesen = estimate_sensitivity_autograd(model, graph, in_feats, labels, "noise", device)
    globsen = estimate_sensitivity_autograd(model, graph, in_feats, labels, "global", device)
    
    out_feats = labels.unique().shape[0]
    
    # Move covariance matrices to device
    sigma_intra = sigma_intra.to(device)
    sigma_inter = sigma_inter.to(device)
    tau = tau.to(device)
    eta = eta.to(device)
    
    # Compute numerator and denominator of the SNR formula
    numerator = torch.zeros(graph.number_of_nodes(), out_feats, device=device)
    denominator = torch.zeros(graph.number_of_nodes(), out_feats, device=device)
    
    for q in range(in_feats):
        for r in range(in_feats):
            numerator += (
                (sigma_intra[q,r] - sigma_inter[q,r]) * sigsen[:,:,q,r]
                + sigma_inter[q,r] * globsen[:,:,q,r]
            )
            denominator += (
                tau[q,r] * globsen[:,:,q,r]
                + eta[q,r] * noisesen[:,:,q,r]
            )
    
    # Prevent division by zero
    denominator[denominator < 1e-8] = 1e-8
    
    # Average SNR across output features
    return torch.mean(numerator / denominator, dim=1)
