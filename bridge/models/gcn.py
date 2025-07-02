"""
Graph Convolutional Network (GCN) implementations.

This module provides implementations of Graph Convolutional Networks (GCNs) and
their variants, including a High Graph Convolution (HPGraphConv).
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
    Graph Convolutional Network (GCN) implementation.
    
    This implementation supports variable depth and optional residual connections.
    
    Args:
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        n_layers: Number of hidden GCN layers
        dropout_p: Dropout probability
        activation: Activation function to use (default: F.relu)
        bias: Whether to use bias in GraphConv layers
        residual_connection: Deprecated flag for residual connections (not used in this implementation)
        do_hp: Depreciated flag for High Pass Graph Convolution (not used in this implementation)
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
        residual_connection: bool = False,
        do_hp: bool = False
    ):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        self.do_hp = do_hp
        
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


class HPGraphConv(nn.Module):
    """
    High Graph Convolution layer.
    
    This layer implements a High filter for graph convolution,
    represented as I - GCN, which emphasizes the difference between a node's features
    and its neighbors' features.
    
    Args:
        in_feats: Input feature dimension
        out_feats: Output feature dimension
        activation: Activation function to use (default: None)
        bias: Whether to use bias
        weight: Whether to apply a linear transformation
        allow_zero_in_degree: Whether to allow nodes with zero in-degree
    """
    def __init__(
        self, 
        in_feats: int, 
        out_feats: int, 
        activation: Optional[Callable] = None, 
        bias: bool = True,
        weight: bool = True,
        allow_zero_in_degree: bool = True
    ):
        super(HPGraphConv, self).__init__()
        self.conv = GraphConv(
            in_feats=in_feats, 
            out_feats=in_feats, 
            activation=None, 
            allow_zero_in_degree=allow_zero_in_degree,
            weight=False, 
            bias=False
        )
        
        # Linear transformation
        if weight:
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        else:
            self.linear = lambda x: x
            
        self.activation = activation
        self.in_feats = in_feats
        self.out_feats = out_feats

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor, do_hp: bool = True) -> torch.Tensor:
        """
        Forward pass for the HPGraphConv layer.
        
        Args:
            g: Input graph
            features: Node feature matrix
            do_hp: Whether to compute High (I - GCN) or just GCN
            
        Returns:
            torch.Tensor: Transformed node features
        """
        if do_hp:
            # High: I - GCN
            conv_h = features - self.conv(g, features)
        else:
            # Standard GCN
            conv_h = self.conv(g, features)

        h = self.linear(conv_h)

        if self.activation is not None:
            h = self.activation(h)
            
        return h
