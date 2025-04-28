"""
Neural network models for graph learning and rewiring.

This module provides Graph Neural Network model implementations, including
Graph Convolutional Networks (GCN) and their variants.
"""

from .gcn import GCN, HPGraphConv
from .selective_models import SelectiveGCN
from .sgc import SGC, sgc_precompute

__all__ = ['GCN', 'HPGraphConv', 'SelectiveGCN', 'SGC', 'sgc_precompute']