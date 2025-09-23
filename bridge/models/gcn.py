"""
Graph Convolutional Network (GCN) implementation.

This module provides a standard Graph Convolutional Network (GCN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv
from typing import Optional, Callable, Union


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN).

    Supports variable depth and dropout.

    Args:
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        n_layers: Number of hidden GCN layers
        dropout_p: Dropout probability
        activation: Activation function to use (default: F.relu)
        bias: Whether to use bias in GraphConv layers
        residual_connection: Unused placeholder for potential residual connections
    """
    def __init__(
        self, 
        in_feats: int, 
        h_feats: int, 
        out_feats: int, 
        n_layers: int, 
        dropout_p: float, 
        activation: Callable = F.relu, 
        bias: bool = True, 
        residual_connection: bool = False
    ):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        
        # Input layer
        self.layers.append(GraphConv(in_feats, h_feats, bias=bias, allow_zero_in_degree=True))
        
        # Hidden layers (if any)
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(h_feats, h_feats, bias=bias, allow_zero_in_degree=True))
     
        # Output layer
        self.layers.append(GraphConv(h_feats, out_feats, bias=bias, allow_zero_in_degree=True))

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GCN model.
        
        Args:
            g: Input graph
            features: Node feature matrix
            
        Returns:
            torch.Tensor: Node embeddings
        """
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:  # no activation & dropout on the output layer
                h = self.activation(h)
                h = self.dropout(h)
        return h
