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
        graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'))
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