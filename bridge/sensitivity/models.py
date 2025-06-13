"""
Graph Convolutional Network models for sensitivity analysis.

This module provides GNN model implementations specifically designed for
sensitivity analysis in the context of graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Tuple, Union, Optional


class LinearGCN(nn.Module):
    """
    A simple linear GCN layer: h = Â @ x @ W.

    The normalized adjacency matrix Â is computed as D^{-1/2}AD^{-1/2}, where
    A is the adjacency matrix with self-loops and D is the degree matrix.
    This model does not include biases or nonlinearities.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden feature dimension (unused in this implementation)
        out_feats: Output feature dimension
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # Just a linear transformation; no biases
        self.weight1 = nn.Parameter(torch.randn(in_feats, out_feats))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LinearGCN.
        
        Args:
            graph: A DGLGraph (the adjacency will be normalized)
            x: Node features, either:
              - shape (num_nodes, in_feats) for a single feature sample, OR
              - shape (num_nodes, in_feats, num_samples) for multiple samples.

        Returns:
            If x is 2D: shape (num_nodes, out_feats).
            If x is 3D: shape (num_nodes, num_samples, out_feats).
        """
        # -- 1) Obtain adjacency as a dense matrix on the same device as x
        A = graph.adj().to(x.device).to_dense().double()

        # -- 2) Normalize the adjacency matrix
        N = A.shape[0]
        d_arr = A.sum(1).double()
        # Handle isolated nodes by adding self-loops
        A[d_arr==0,:][:,d_arr==0] = torch.eye(N).to(x.device).double()[d_arr==0,:][:,d_arr==0]
        d_arr[d_arr==0] = 1
        d_inv_arr = (1/d_arr**(1/2))
        A_norm = d_inv_arr[:,None]*A*d_inv_arr[None,:]

        if x.dim() == 2:
            # x: [N, in_feats]
            # => output: [N, out_feats]
            return A_norm @ x.double() @ self.weight1

        elif x.dim() == 3:
            # x: [N, F, S]
            N, F, S = x.shape

            # Flatten out the last dimension for a single matmul with A_norm
            x_2d = x.reshape(N, F*S)   # [N, F*S]
            Ax = A_norm @ x_2d         # => [N, F*S]

            # Reshape back to 3D => [N, F, S]
            Ax_3d = Ax.view(N, F, S)

            # Multiply by weight1 [F, O], to get shape [N, S, O].
            # Using einsum 'nfs,fo->nos':
            h_3d = torch.einsum('nfs,fo->nos', Ax_3d, self.weight1)

            # shape => [N, S, O]
            return h_3d

        else:
            raise ValueError("Unsupported input dimensionality for x.")
        

class FNN(nn.Module):
    """
    A simple linear layer: h = Â @ x @ W.

    The normalized adjacency matrix Â is computed as D^{-1/2}AD^{-1/2}, where
    A is the adjacency matrix with self-loops and D is the degree matrix.
    This model does not include biases or nonlinearities.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden feature dimension (unused in this implementation)
        out_feats: Output feature dimension
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # Just a linear transformation; no biases
        self.weight1 = nn.Parameter(torch.randn(in_feats, out_feats))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LinearGCN.
        
        Args:
            graph: A DGLGraph (the adjacency will be normalized)
            x: Node features, either:
              - shape (num_nodes, in_feats) for a single feature sample, OR
              - shape (num_nodes, in_feats, num_samples) for multiple samples.

        Returns:
            If x is 2D: shape (num_nodes, out_feats).
            If x is 3D: shape (num_nodes, num_samples, out_feats).
        """

        if x.dim() == 2:
            # x: [N, in_feats]
            # => output: [N, out_feats]
            return x.double() @ self.weight1

        elif x.dim() == 3:
            # x: [N, F, S]
            N, F, S = x.shape

            # Flatten out the last dimension for a single matmul with A_norm
            x_2d = x.reshape(N, F*S)   # [N, F*S]

            # Reshape back to 3D => [N, F, S]
            x_3d = x_2d.view(N, F, S)

            # Multiply by weight1 [F, O], to get shape [N, S, O].
            # Using einsum 'nfs,fo->nos':
            h_3d = torch.einsum('nfs,fo->nos', x_3d, self.weight1)

            # shape => [N, S, O]
            return h_3d

        else:
            raise ValueError("Unsupported input dimensionality for x.")


class TwoLayerGCN(nn.Module):
    """
    A two-layer GCN with ReLU activation between layers.
    
    This model applies two graph convolution layers with a ReLU nonlinearity
    after the first layer, and includes trainable bias terms.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden feature dimension
        out_feats: Output feature dimension
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # The +1 accounts for the bias term, which is appended as a feature
        self.weight1 = nn.Parameter(torch.randn(in_feats+1, hidden_feats))
        self.weight2 = nn.Parameter(torch.randn(hidden_feats+1, out_feats))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TwoLayerGCN.
        
        Args:
            graph: A DGLGraph (the adjacency will be normalized)
            x: Node features, either:
              - shape (num_nodes, in_feats) for a single feature sample, OR
              - shape (num_nodes, in_feats, num_samples) for multiple samples.

        Returns:
            If x is 2D: shape (num_nodes, out_feats).
            If x is 3D: shape (num_nodes, num_samples, out_feats).
        """
        # -- 1) Obtain adjacency as a dense matrix on the same device as x
        A = graph.adj().to(x.device).to_dense().double()

        # -- 2) Normalize the adjacency matrix
        N = A.shape[0]
        d_arr = A.sum(1).double()
        # Handle isolated nodes by adding self-loops
        A[d_arr==0,:][:,d_arr==0] = torch.eye(N).to(x.device).double()[d_arr==0,:][:,d_arr==0]
        d_arr[d_arr==0] = 1
        d_inv_arr = (1/d_arr**(1/2))
        A_norm = d_inv_arr[:,None]*A*d_inv_arr[None,:]

        if x.dim() == 2:
            # x: [N, in_feats]
            # => output: [N, out_feats]
            # Append bias feature
            h = torch.cat((x, torch.ones((x.shape[0], 1), device=x.device)), dim=1)
            h = F.relu(A_norm @ h.double() @ self.weight1)
            h = torch.cat((h, torch.ones((h.shape[0], 1), device=h.device)), dim=1)
            return A_norm @ h @ self.weight2

        elif x.dim() == 3:
            # x: [N, F, S]
            h = x
            
            # Process through both layers
            for i, W in enumerate([self.weight1, self.weight2]):
                # Append bias feature along feature dimension
                h = torch.cat((h, torch.ones((h.shape[0], 1, h.shape[2]), device=h.device)), dim=1)
                
                N, F_num, S = h.shape
                
                # Flatten last dimension for adjacency matrix multiplication
                h_2d = h.reshape(N, F_num*S)   # [N, F_num*S]
                Ah = A_norm @ h_2d           # => [N, F_num*S]
                
                # Reshape back to 3D => [N, F_num, S]
                Ah_3d = Ah.view(N, F_num, S)
                
                # Multiply by weight [F_num, O], to get shape [N, S, O].
                h = torch.einsum('nfs,fo->nos', Ah_3d, W)
                
                # Apply ReLU after the first layer only
                if i == 0:
                    h = F.relu(h)
                    
            return h

        else:
            raise ValueError("Unsupported input dimensionality for x.")


class TwoLayerFNN(nn.Module):
    """
    A two-layer feedforward neural network with ReLU activation between layers.
    
    This model applies two linear transformations with a ReLU nonlinearity
    after the first layer, and includes trainable bias terms.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden feature dimension
        out_feats: Output feature dimension
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # The +1 accounts for the bias term, which is appended as a feature
        self.weight1 = nn.Parameter(torch.randn(in_feats+1, hidden_feats))
        self.weight2 = nn.Parameter(torch.randn(hidden_feats+1, out_feats))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TwoLayerFNN.
        
        Args:
            graph: A DGLGraph (not used but kept for consistency)
            x: Node features, either:
              - shape (num_nodes, in_feats) for a single feature sample, OR
              - shape (num_nodes, in_feats, num_samples) for multiple samples.

        Returns:
            If x is 2D: shape (num_nodes, out_feats).
            If x is 3D: shape (num_nodes, num_samples, out_feats).
        """

        if x.dim() == 2:
            # x: [N, in_feats]
            # => output: [N, out_feats]
            # Append bias feature
            h = torch.cat((x, torch.ones((x.shape[0], 1), device=x.device)), dim=1)
            h = F.relu(h.double() @ self.weight1)
            h = torch.cat((h, torch.ones((h.shape[0], 1), device=h.device)), dim=1)
            return h @ self.weight2

        elif x.dim() == 3:
            # x: [N, F, S]
            h = x
            
            # Process through both layers
            for i, W in enumerate([self.weight1, self.weight2]):
                # Append bias feature along feature dimension
                h = torch.cat((h, torch.ones((h.shape[0], 1, h.shape[2]), device=h.device)), dim=1)
                
                N, F_num, S = h.shape
                
                # No adjacency matrix multiplication for FNN - just apply weight
                # Multiply by weight [F_num, O], to get shape [N, S, O].
                h = torch.einsum('nfs,fo->nos', h, W)
                
                # Apply ReLU after the first layer only
                if i == 0:
                    h = F.relu(h)
                    
            return h

        else:
            raise ValueError("Unsupported input dimensionality for x.")