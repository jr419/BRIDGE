"""
Simple Graph Convolution (SGC) implementation.

This module provides an efficient implementation of the SGC model that
precomputes the graph propagation and uses a simple linear classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from typing import Optional, Callable, Union


def sgc_precompute(
    graph: dgl.DGLGraph,
    feats: torch.Tensor,
    K: int = 2,
    add_self_loop: bool = True
) -> torch.Tensor:
    """
    Precompute S^K X without requiring gradients.
    
    Args:
        graph: Input graph
        feats: Node feature matrix
        K: Number of propagation steps (equivalent to GCN depth)
        add_self_loop: Whether to add self-loops to the graph
        
    Returns:
        torch.Tensor: Propagated node features
    """
    if add_self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

    deg = graph.in_degrees().float().clamp(min=1)
    norm = deg.pow(-0.5).unsqueeze(1)        # D^{-1/2}

    h = feats
    for _ in range(K):
        h = h * norm                          # left norm
        graph.ndata['h'] = h
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h = graph.ndata.pop('h') * norm       # right norm
    
    return h  # detached tensor – ready for any classifier


class SGC(nn.Module):
    """
    Simple Graph Convolution (SGC) model that precomputes graph propagation
    and uses a linear classifier.
    
    Args:
        in_feats: Input feature dimension
        out_feats: Output feature dimension (number of classes)
        K: Number of propagation steps
        cached: Whether to cache the propagated features
        add_self_loop: Whether to add self-loops to the graph
        bias: Whether to use bias in the linear classifier
    """
    def __init__(
        self, 
        in_feats: int, 
        out_feats: int, 
        K: int = 2,
        cached: bool = True,
        add_self_loop: bool = True,
        bias: bool = True
    ):
        super(SGC, self).__init__()
        self.K = K
        self.cached = cached
        self.add_self_loop = add_self_loop
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.cached_h = None
        
    def reset_parameters(self):
        self.linear.reset_parameters()
        self.cached_h = None
        
    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SGC model.
        
        Args:
            g: Input graph
            features: Node feature matrix
            
        Returns:
            torch.Tensor: Node classification logits
        """
        if self.cached and self.cached_h is not None:
            h = self.cached_h
        else:
            h = sgc_precompute(g, features, self.K, self.add_self_loop)
            if self.cached:
                self.cached_h = h
        
        return self.linear(h)
    

    """
Selective Simple Graph Convolution (SelectiveSGC) implementation.

This module provides an implementation of a selective SGC model that can
operate on multiple graph structures and choose the best one for each node.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import List, Optional


class SelectiveSGC(nn.Module):
    """
    Selective Simple Graph Convolution (SelectiveSGC) model.
    
    This model applies SGC to multiple versions of the input graph,
    then selects the best version for each node based on a mask provided 
    in the graph's node features.
    
    Args:
        in_feats: Input feature dimension
        out_feats: Output feature dimension (number of classes)
        K: Number of propagation steps
        cached: Whether to cache the propagated features
        add_self_loop: Whether to add self-loops to the graph
        bias: Whether to use bias in the linear classifier
    """
    def __init__(
        self, 
        in_feats: int, 
        out_feats: int, 
        K: int = 2,
        cached: bool = True,
        add_self_loop: bool = True,
        bias: bool = True
    ):
        super(SelectiveSGC, self).__init__()
        self.K = K
        self.cached = cached
        self.add_self_loop = add_self_loop
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.cached_h_list = None
        self.out_feats = out_feats
        
    def reset_parameters(self):
        """Reset model parameters and clear cache."""
        self.linear.reset_parameters()
        self.cached_h_list = None
    
    def sgc_precompute(self, graph, feats):
        """Precompute S^K X for a single graph."""
        if self.add_self_loop:
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)

        deg = graph.in_degrees().float().clamp(min=1)
        norm = deg.pow(-0.5).unsqueeze(1)  # D^{-1/2}

        h = feats
        for _ in range(self.K):
            h = h * norm  # left norm
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = graph.ndata.pop('h') * norm  # right norm
        
        return h
        
    def forward(self, g_list: List[dgl.DGLGraph], features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SelectiveSGC model.
        
        This method applies SGC to each graph in the list, then selects
        the appropriate output for each node based on the 'mask' field 
        in the first graph.
        
        Args:
            g_list: List of input graphs
            features: Node feature matrix
            
        Returns:
            torch.Tensor: Node classification logits selected from the best graph for each node
        """
        device = features.device
        
        # Process each graph with precomputed features
        if self.cached and self.cached_h_list is not None:
            h_list = self.cached_h_list
        else:
            h_list = []
            for g in g_list:
                h = self.sgc_precompute(g, features)
                h_list.append(h)
            
            if self.cached:
                self.cached_h_list = h_list
        
        # Apply linear transform to each set of features
        logits_list = []
        for h in h_list:
            logits = self.linear(h)
            logits_list.append(logits)
        
        # Stack logits from all graphs
        h_out = torch.stack(logits_list)  # Shape: [num_graphs, num_nodes, out_feats]
        
        # Get the mask that determines which graph to use for each node
        mask = g_list[0].ndata['mask'].long()  # Shape: [num_nodes]
        
        # Generate node indices
        n = features.size(0)
        node_indices = torch.arange(n, device=device)
        
        # Select the appropriate features based on the mask
        h_selected = h_out[mask, node_indices, :]
        
        return h_selected