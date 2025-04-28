"""
Synthetic graph dataset generation.

This module provides classes for generating synthetic graph datasets
with controllable properties, such as homophily and community structure.
"""

import os
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
from typing import Optional, Tuple, List, Dict, Union, Any

from bridge.utils.dataset_processing import add_train_val_test_splits


class SyntheticGraphDataset(DGLDataset):
    """
    A DGL Dataset that generates a synthetic graph using a stochastic block model (SBM)
    and assigns node features generated from a block-based covariance model.

    Args:
        n: Number of nodes
        k: Number of communities (classes/blocks)
        h: Homophily ratio (higher values favor intra-community edges)
        d_mean: Mean degree scaling factor
        sigma_intra_scalar: Scalar for intra-class covariance
        sigma_inter_scalar: Scalar for inter-class covariance
        tau_scalar: Scalar for the global covariance (shared across nodes)
        eta_scalar: Scalar for node-wise noise covariance
        in_feats: Dimensionality of node features
        d_in: Input dimension for feature generation (defaults to k if not provided)
        alpha: Dirichlet concentration parameter for block proportions
        sym: If True, creates an undirected (symmetric) graph
        mu: Class-specific mean matrix (if None, defaults are used)
    """
    def __init__(
        self,
        n: int = 100,
        k: int = 3,
        h: float = 0.8,
        d_mean: float = 3,
        sigma_intra_scalar: float = 0.1,
        sigma_inter_scalar: float = -0.05,
        tau_scalar: float = 1,
        eta_scalar: float = 1,
        in_feats: int = 5,
        d_in: Optional[int] = None,
        alpha: Optional[float] = None,
        sym: bool = True,
        mu: Optional[np.ndarray] = None
    ):
        self.n = n
        self.k = k
        self.h = h
        self.d_mean = d_mean
        self.sigma_intra_scalar = sigma_intra_scalar
        self.sigma_inter_scalar = sigma_inter_scalar
        self.tau_scalar = tau_scalar
        self.eta_scalar = eta_scalar
        self.in_feats = in_feats
        self.d_in = d_in if d_in is not None else k
        self.alpha = alpha
        self.sym = sym
        self.mu = mu

        super().__init__(name=f'synthetic_graph_dataset_h={h:.2f}_d={d_mean:.2f}')

    def process(self) -> None:
        """
        Generate the synthetic SBM graph and its features.
        
        This method:
        1. Builds a block connectivity matrix based on homophily parameter
        2. Generates community assignments for nodes
        3. Creates the graph adjacency matrix using the SBM model
        4. Constructs a DGL graph from the adjacency
        5. Generates node features based on community assignments
        6. Creates train/validation/test splits
        """
        n, k, h, d_mean = self.n, self.k, self.h, self.d_mean

        # -- Build the block connectivity matrix B --
        B = np.zeros((k, k))
        # Intra-community connections
        B[np.diag_indices(k)] = k * d_mean * h
        # Inter-community connections
        off_diag = np.where(np.ones((k, k)) - np.eye(k))
        B[off_diag] = k * d_mean * (1 - h) / (k - 1)

        # -- Generate block proportions (Pi) and assign nodes to blocks --
        if self.alpha is None:
            Pi = np.eye(k) / k
        else:
            Pi = np.diag(np.random.default_rng().dirichlet([self.alpha] * k))
        # Assign each node a block (label)
        y = np.random.default_rng().choice(k, n, p=np.diag(Pi))
        
        # One-hot encode block membership for later computation.
        Z = np.zeros((n, k))
        Z[np.arange(n), y] = 1

        # -- Compute edge probability matrix and sample the adjacency --
        A_p = np.dot(Z, np.dot(B, Z.T)) / n
        A_p = torch.tensor(A_p, dtype=torch.float32)
        A = torch.bernoulli(A_p).numpy()
        if self.sym:
            # Enforce symmetry to create an undirected graph.
            A[np.triu_indices(n, 1)] = A.T[np.triu_indices(n, 1)]
        # Create the DGL graph.
        u, v = np.where(A == 1)
        self.graph = dgl.graph((u, v), num_nodes=n)
        
        # -- Store node labels --
        self.graph.ndata['label'] = torch.tensor(y, dtype=torch.long)
        # Initialize placeholder for features.
        self.graph.ndata['feat'] = torch.zeros(n, self.in_feats, dtype=torch.float32)
        
        # -- Create train/validation/test masks --
        self.graph = add_train_val_test_splits(self.graph, 0.6, num_splits=100)

        # Save B for potential further inspection.
        self.B = B

        # -- Generate node features using a covariance structure --
        self._generate_features()

    def _generate_features(self, num_mu_samples: int = 1) -> None:
        """
        Generate node features using a block-based covariance model.
        
        Features are generated as a sum of:
        1. Class-specific mean vectors
        2. Global random variation
        3. Node-specific noise
        
        Args:
            num_mu_samples: Number of independent realizations to generate
        """
        n = self.n
        in_feats = self.in_feats
        labels = self.graph.ndata['label'].numpy()

        # Scale for covariance matrices.
        scale = 1e-4
        sigma_intra_np = self.sigma_intra_scalar * scale * np.eye(in_feats)
        sigma_inter_np = self.sigma_inter_scalar * scale * np.eye(in_feats)
        tau_np         = self.tau_scalar         * scale * np.eye(in_feats)
        eta_np         = self.eta_scalar         * scale * np.eye(in_feats)

        # Generate features with shape (n, in_feats, num_mu_samples)
        X_repeats = SyntheticGraphDataset.generate_features(
            num_nodes=n,
            num_features=in_feats,
            labels=labels,
            inter_class_cov=sigma_inter_np,
            intra_class_cov=sigma_intra_np,
            global_cov=tau_np,
            noise_cov=eta_np,
            mu_repeats=num_mu_samples
        )
        # Use the first realization.
        self.graph.ndata['feat'] = torch.tensor(X_repeats[:, :, 0], dtype=torch.float32)

    @staticmethod
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
        Generate node features according to a block-based model.
        
        For each node, a class-specific mean vector is drawn from a multivariate normal 
        with covariance having intra- and inter-class blocks. Then a global variation (gamma)
        and node-specific noise (epsilon) are added.

        Args:
            num_nodes: Number of nodes in the graph
            num_features: Dimensionality of node features
            labels: Node labels/community assignments
            inter_class_cov: Covariance matrix for inter-class relationships
            intra_class_cov: Covariance matrix for intra-class relationships 
            global_cov: Covariance matrix for global variations
            noise_cov: Covariance matrix for node-specific noise
            mu_repeats: Number of independent realizations to generate
            
        Returns:
            np.ndarray: Generated features of shape (num_nodes, num_features, mu_repeats)
        """
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # Build a large covariance matrix for the class-specific means.
        mu_covariance = np.zeros((num_classes * num_features, num_classes * num_features))
        for i in range(num_classes):
            for j in range(num_classes):
                start_i, end_i = i * num_features, (i + 1) * num_features
                start_j, end_j = j * num_features, (j + 1) * num_features
                if i == j:
                    mu_covariance[start_i:end_i, start_j:end_j] = intra_class_cov
                else:
                    mu_covariance[start_i:end_i, start_j:end_j] = inter_class_cov

        # Sample concatenated class means and reshape.
        mu_concatenated = np.random.multivariate_normal(
            mean=np.zeros(num_classes * num_features),
            cov=mu_covariance
        )
        mu_vectors = mu_concatenated.reshape(num_classes, num_features)
        
        # Global variation: shape (mu_repeats, num_features)
        gamma_all = np.random.multivariate_normal(
            mean=np.zeros(num_features),
            cov=global_cov,
            size=mu_repeats
        )
        # Node-specific noise: shape (mu_repeats, num_nodes, num_features)
        epsilon_all = np.random.multivariate_normal(
            mean=np.zeros(num_features),
            cov=noise_cov,
            size=(mu_repeats, num_nodes)
        )
        
        # Gather node-specific means and broadcast.
        mu_all = mu_vectors[labels]             # (num_nodes, num_features)
        mu_all = mu_all[None, :, :]             # (1, num_nodes, num_features)
        gamma_all = gamma_all[:, None, :]       # (mu_repeats, 1, num_features)
        
        # Sum the components.
        X_all = mu_all + gamma_all + epsilon_all  # (mu_repeats, num_nodes, num_features)
        # Transpose to shape: (num_nodes, num_features, mu_repeats)
        X_repeats = np.transpose(X_all, (1, 2, 0))
        return X_repeats

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        """
        Get the graph at the specified index.
        
        Args:
            idx: Index of the graph to retrieve
            
        Returns:
            dgl.DGLGraph: The graph at the specified index
            
        Raises:
            IndexError: If the index is out of bounds (only index 0 is valid)
        """
        if idx != 0:
            raise IndexError("SyntheticGraphDataset contains only one graph.")
        return self.graph

    def __len__(self) -> int:
        """
        Get the number of graphs in the dataset.
        
        Returns:
            int: The number of graphs in the dataset (always 1)
        """
        return 1
