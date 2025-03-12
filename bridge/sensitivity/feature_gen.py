"""
Feature generation utilities for sensitivity analysis.

This module provides functions for generating synthetic node features
with controlled covariance structures between classes, global shifts,
and node-specific noise, for use in sensitivity analysis.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Union, Optional, Any


def generate_features(
    num_nodes: int, 
    num_features: int, 
    labels: np.ndarray, 
    inter_class_cov: np.ndarray, 
    intra_class_cov: np.ndarray, 
    global_cov: np.ndarray, 
    noise_cov: np.ndarray, 
    mu_repeats: int = 1
) -> np.ndarray:
    """
    Generate synthetic node features with controlled covariance structure.
    
    This function implements the feature generation model from the paper:
        X_i = μ_{y_i} + γ + ε_i
    where:
        - μ_{y_i} is the class-specific mean for the class of node i
        - γ is a global random vector (same for all nodes)
        - ε_i is node-specific noise
    
    Args:
        num_nodes: Number of nodes
        num_features: Number of feature dimensions
        labels: Node class labels (numpy array)
        inter_class_cov: Covariance matrix between different classes
        intra_class_cov: Covariance matrix within the same class
        global_cov: Covariance matrix for the global shift
        noise_cov: Covariance matrix for the node-specific noise
        mu_repeats: Number of feature realizations to generate for each class mean
        
    Returns:
        A numpy array of shape (num_nodes, num_features, mu_repeats)
        containing the generated features
    """
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Build a large covariance matrix for the class means:
    # (num_classes*num_features) x (num_classes*num_features)
    mu_covariance = np.zeros((num_classes * num_features, num_classes * num_features))
    for i in range(num_classes):
        for j in range(num_classes):
            start_i, end_i = i * num_features, (i + 1) * num_features
            start_j, end_j = j * num_features, (j + 1) * num_features
            if i == j:
                mu_covariance[start_i:end_i, start_j:end_j] = intra_class_cov
            else:
                mu_covariance[start_i:end_i, start_j:end_j] = inter_class_cov

    # Sample concatenated class means and reshape
    mu_concatenated = np.random.multivariate_normal(
        mean=np.zeros(num_classes * num_features),
        cov=mu_covariance
    )
    mu_vectors = mu_concatenated.reshape(num_classes, num_features)

    # 1) Generate global shift vectors for each repeat: shape (mu_repeats, num_features)
    gamma_all = np.random.multivariate_normal(
        mean=np.zeros(num_features),
        cov=global_cov,
        size=mu_repeats
    )

    # 2) Generate node-specific noise for all nodes and repeats: shape (mu_repeats, num_nodes, num_features)
    epsilon_all = np.random.multivariate_normal(
        mean=np.zeros(num_features),
        cov=noise_cov,
        size=(mu_repeats, num_nodes)
    )
    
    # 3) Gather node-specific means and broadcast:
    #    mu_vectors[labels] has shape (num_nodes, num_features).
    mu_all = mu_vectors[labels]              # (num_nodes, num_features)
    mu_all = mu_all[None, :, :]             # (1, num_nodes, num_features)
    gamma_all = gamma_all[:, None, :]       # (mu_repeats, 1, num_features)

    # 4) Sum them up: result has shape (mu_repeats, num_nodes, num_features)
    X_all = mu_all + gamma_all + epsilon_all

    # 5) Transpose to (num_nodes, num_features, mu_repeats)
    X_repeats = np.transpose(X_all, (1, 2, 0))

    return X_repeats


def create_feature_generator(
    sigma_intra: torch.Tensor,
    sigma_inter: torch.Tensor,
    tau: torch.Tensor,
    eta: torch.Tensor,
    dtype: torch.dtype = torch.float64
):
    """
    Create a feature generator function with fixed covariance parameters.
    
    Args:
        sigma_intra: Intra-class covariance matrix
        sigma_inter: Inter-class covariance matrix
        tau: Global shift covariance matrix
        eta: Noise covariance matrix
        dtype: Torch data type for the output tensor
        
    Returns:
        A function that generates features with signature:
        feature_generator(num_nodes, in_feats, labels, num_mu_samples)
    """
    def feature_generator_fixed(
        num_nodes: int, 
        in_feats: int, 
        labels: torch.Tensor, 
        num_mu_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate features with predefined covariance structure.
        
        Args:
            num_nodes: Number of nodes
            in_feats: Number of input features
            labels: Node labels
            num_mu_samples: Number of samples to generate per class mean
            
        Returns:
            A tensor of shape (num_nodes, in_feats, num_mu_samples)
        """
        # Extract diagonal elements for simplicity (can be extended to full matrices)
        inter_class_cov = sigma_inter.diag().numpy()[0] * np.eye(in_feats)
        intra_class_cov = sigma_intra.diag().numpy()[0] * np.eye(in_feats)
        global_cov = tau.diag().numpy()[0] * np.eye(in_feats)
        noise_cov = eta.diag().numpy()[0] * np.eye(in_feats)
        
        # Convert labels to numpy
        labels_np = labels.numpy()

        # Generate features
        features_np = generate_features(
            num_nodes, in_feats, labels_np,
            inter_class_cov, intra_class_cov, global_cov, noise_cov,
            mu_repeats=num_mu_samples
        )
        
        # Convert to torch tensor with requested dtype
        features_torch = torch.tensor(features_np, dtype=dtype)
        return features_torch
    
    return feature_generator_fixed
