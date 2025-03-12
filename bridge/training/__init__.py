"""
Training and evaluation utilities for graph neural networks.

This module provides functions for training and evaluating graph neural networks,
including metrics for evaluation and training loops.
"""

from .metrics import evaluate_metrics, get_metric_type
from .train import train, train_selective, train_one_epoch, validate

__all__ = [
    'evaluate_metrics', 
    'get_metric_type', 
    'train', 
    'train_selective', 
    'train_one_epoch', 
    'validate'
]
