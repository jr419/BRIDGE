"""
Selective Graph Neural Network models.

This module provides implementations of Graph Neural Networks that can selectively
operate on different graph structures and choose the best one for each node.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GraphConv
from typing import List, Optional, Callable, Union
from .gcn import HPGraphConv


class SelectiveGCN(nn.Module):
    """
    Selective Graph Convolutional Network that operates on multiple graph versions.
    
    This model applies the same GCN architecture to multiple versions of the input graph,
    then selects the best version for each node based on a mask provided in the graph's
    node features.
    
    Args:
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        n_layers: Number of hidden GCN layers
        dropout_p: Dropout probability
        activation: Activation function to use (default: F.relu)
        bias: Whether to use bias in GraphConv layers
        residual_connection: Whether to use residual connections
        do_hp: Whether to use HPGraphConv instead of GraphConv
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
        super(SelectiveGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        self.do_hp = do_hp
        self.out_feats = out_feats
        
        # Input layer
        if do_hp:
            self.layers.append(HPGraphConv(in_feats, h_feats, bias=bias, allow_zero_in_degree=True))
        else:
            self.layers.append(GraphConv(in_feats, h_feats, bias=bias, allow_zero_in_degree=True))
        
        # Hidden layers (if any)
        for _ in range(n_layers - 1):
            if do_hp:
                self.layers.append(HPGraphConv(h_feats, h_feats, bias=bias, allow_zero_in_degree=True))
            else:
                self.layers.append(GraphConv(h_feats, h_feats, bias=bias, allow_zero_in_degree=True))
     
        # Output layer
        if do_hp:
            self.layers.append(HPGraphConv(h_feats, out_feats, bias=bias, allow_zero_in_degree=True))
        else:
            self.layers.append(GraphConv(h_feats, out_feats, bias=bias, allow_zero_in_degree=True))
    
    def forward(self, g_list: List[dgl.DGLGraph], features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SelectiveGCN model.
        
        This method applies the GCN layers to each graph in the list, then selects
        the appropriate output for each node based on the 'mask' field in the first graph.
        
        Args:
            g_list: List of input graphs
            features: Node feature matrix
            
        Returns:
            torch.Tensor: Node embeddings selected from the best graph for each node
        """
        device = features.device
        h_out = torch.zeros((len(g_list), g_list[0].number_of_nodes(), self.out_feats), device=device)

        # Process each graph with the same GCN
        for i, g in enumerate(g_list):
            h = features
            for j, layer in enumerate(self.layers):
                h = layer(g, h)
                if j != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
            h_out[i] = h

        # Get the mask that determines which graph to use for each node
        mask = g_list[0].ndata['mask'].long()  # Shape: (n,)
        
        # Generate node indices
        n = features.size(0)
        node_indices = torch.arange(n, device=device)
        
        # Select the appropriate convolved features based on the mask
        # h_selected: Shape (n, out_feats)
        h_selected = h_out[mask, node_indices, :]
        return h_selected
