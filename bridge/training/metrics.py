"""
Metrics for evaluating graph neural networks.

This module provides functions for computing various metrics for graph neural networks,
including accuracy, ROC AUC, and more.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from typing import Tuple, List, Dict, Union, Optional, Any


def evaluate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    metric_type: str = 'accuracy'
) -> float:
    """
    Compute classification metrics on the specified mask.
    
    Args:
        logits: Model output logits with shape (n_nodes, n_classes)
        labels: True labels with shape (n_nodes,)
        mask: Boolean mask indicating which nodes to evaluate
        metric_type: Type of metric to compute, either 'accuracy' or 'roc_auc'
        
    Returns:
        float: The computed metric value
    """
    logits = logits[mask]
    labels = labels[mask]
    
    if metric_type == 'accuracy':
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = mask.sum().item()
        return correct / total if total > 0 else 0.0
    
    elif metric_type == 'roc_auc':
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
 
        # For multi-class, use one-vs-rest ROC AUC
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1])
        return roc_auc_score(
            labels_one_hot.cpu().numpy(), 
            probs.detach().cpu().numpy(),
            multi_class='ovr' if logits.shape[1] > 2 else 'raise'
        )
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def get_metric_type(dataset_name: str) -> str:
    """
    Determine which metric to use based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        str: Either 'accuracy' or 'roc_auc'
    """
    accuracy_datasets = {'roman-empire', 'amazon-ratings'}
    roc_auc_datasets = {'minesweeper', 'tolokers', 'questions'}
    
    dataset_name = dataset_name.lower()
    if dataset_name in accuracy_datasets:
        return 'accuracy'
    elif dataset_name in roc_auc_datasets:
        return 'roc_auc'
    else:
        # Default to accuracy for other datasets
        return 'accuracy'
