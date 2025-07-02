"""
Hyperparameter optimization for graph neural networks.

This module provides utilities for optimizing hyperparameters of graph neural networks
using Optuna, with a focus on graph rewiring approaches.
"""

from .optuna_objectives import (
    objective_mpnn, 
    objective_rewiring, 
    objective_iterative_rewiring,
    collect_float_metrics,
    train_and_evaluate_mpnn
)

__all__ = [
    'objective_mpnn', 
    'objective_rewiring',
    'objective_iterative_rewiring', 
    'collect_float_metrics',
    'train_and_evaluate_mpnn'
]
