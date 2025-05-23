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


import urllib.request
import tempfile
import tarfile
from pathlib import Path

_DATA_URLS = {
    "squirrel_filtered":
        "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/squirrel_filtered_directed.npz",
    "chameleon_filtered":
        "https://github.com/yandex-research/heterophilous-graphs/raw/main/data/chameleon_filtered_directed.npz",
}

class _SingleGraphDataset(dgl.data.DGLDataset):
    """Wrap a single pre-built DGLGraph so that BRIDGE can keep using `dataset[0]`."""

    def __init__(self, name: str, graph: dgl.DGLGraph):
        self._graph = graph
        super().__init__(name=name)

    def process(self):               # nothing to process – we already have the graph
        pass

    def __getitem__(self, idx):      # BRIDGE always does `dataset[0]`
        if idx != 0:
            raise IndexError
        return self._graph

    def __len__(self):
        return 1

    @property
    def name(self):
        return self._name

def load_filtered_heterophily(name: str,
                              raw_dir: str = "~/.dgl/filtered_heterophily",
                              make_bidirected: bool = False) -> _SingleGraphDataset:
    """
    Download (if necessary) and return the leakage-free *squirrel* or *chameleon*
    graph in native DGL format.

    Returns
    -------
    dgl.data.DGLDataset
        Wrapper with a single DGLGraph inside, ready for BRIDGE.
    """
    name = name.lower()
    if name not in _DATA_URLS:
        raise ValueError(f"Unknown filtered dataset: {name}")

    root = Path(raw_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    local_file = root / f"{name}.npz"

    # download once
    if not local_file.exists():
        print(f"→ downloading {name}...")
        urllib.request.urlretrieve(_DATA_URLS[name], local_file)

    data = np.load(local_file, allow_pickle=True)
    edge_index = torch.tensor(data["edges"], dtype=torch.long).T
    feats       = torch.tensor(data["node_features"], dtype=torch.float32)
    labels      = torch.tensor(data["node_labels"], dtype=torch.long)
    

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=feats.shape[0])
    if make_bidirected:
        g = dgl.to_bidirected(g, copy_ndata=False)

    g.ndata["feat"]  = feats
    g.ndata["label"] = labels

    # masks (10 official splits) are already inside the .npz – keep BRIDGE happy
    for key in ("train_masks", "val_masks", "test_masks"):
        if key in data.files:                       # shape (10, N)
            print(f"→ adding {key} to graph")
            print(data[key].shape)
            g.ndata[key[:-1]] = torch.tensor(data[key]).T

    return _SingleGraphDataset(name=name, graph=g)



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


