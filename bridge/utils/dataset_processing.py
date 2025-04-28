"""
Utility functions for processing datasets.

This module provides functions for loading, preprocessing and datasets
"""
import os
import numpy as np
import pandas as pd
import torch
import dgl
from typing import List, Dict, Union, Optional, Any 


def add_train_val_test_splits(g: dgl.DGLGraph, split_ratio: float = 0.8, num_splits: int = 1) -> dgl.DGLGraph:
    """
    Add train, validation, and test masks to the graph.
    The graph is split into train, validation, and test sets based on the given ratio.
    Args:
        g (dgl.DGLGraph): The input graph.
        split_ratio (float): The ratio for splitting the graph into train, validation, and test sets.
    """
    n = g.num_nodes()

    if num_splits == 1:
        train_mask = torch.zeros(n, dtype=torch.bool, device=g.device)
        val_mask = torch.zeros(n, dtype=torch.bool, device=g.device)
        test_mask = torch.zeros(n, dtype=torch.bool, device=g.device)
        full_mask = np.random.choice(n, n, replace=False)
        train_indices = full_mask[:int(n * split_ratio)]
        val_indices = full_mask[int(n * split_ratio):int(n * (1/2+split_ratio/2))]
        test_indices = full_mask[int(n * (1/2+split_ratio/2)):]
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
    else:
        train_mask = torch.zeros((n, num_splits), dtype=torch.bool, device=g.device)
        val_mask = torch.zeros((n, num_splits), dtype=torch.bool, device=g.device)
        test_mask = torch.zeros((n, num_splits), dtype=torch.bool, device=g.device)
        for i in range(num_splits):
            full_mask = np.random.choice(n, n, replace=False)
            train_indices = full_mask[:int(n * split_ratio)]
            val_indices = full_mask[int(n * split_ratio):int(n * (1/2+split_ratio/2))]
            test_indices = full_mask[int(n * (1/2+split_ratio/2)):]
            train_mask[train_indices, i] = True
            val_mask[val_indices, i] = True
            test_mask[test_indices, i] = True
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
    return g