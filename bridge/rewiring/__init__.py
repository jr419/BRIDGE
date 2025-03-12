"""
Graph rewiring utilities for optimizing graph neural networks.

This module provides functions for rewiring graph structures to improve
the performance of graph neural networks while preserving important information.
"""

from .operations import create_rewired_graph
from .pipeline import run_bridge_pipeline, run_bridge_experiment

__all__ = ['create_rewired_graph', 'run_bridge_pipeline', 'run_bridge_experiment']
