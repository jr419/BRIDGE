"""
Hyperparameter optimization for graph neural networks.

This module provides utilities for optimizing hyperparameters of graph neural networks
using Optuna, with a focus on graph rewiring approaches.
"""

from .optuna_objectives import objective_gcn, objective_rewiring, collect_float_metrics

__all__ = ['objective_gcn', 'objective_rewiring', 'collect_float_metrics']
