---
layout: default
title: run_bridge_pipeline
parent: API Reference
---

# run_bridge_pipeline
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `run_bridge_pipeline` function implements the complete BRIDGE (Block Rewiring from Inference-Derived Graph Ensembles) pipeline. This pipeline optimizes graph neural networks through inference-derived graph rewiring.

## Function Signature

```python
def run_bridge_pipeline(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    h_feats_gcn: int = 64,
    n_layers_gcn: int = 2,
    dropout_p_gcn: float = 0.5,
    model_lr_gcn: float = 1e-3,
    wd_gcn: float = 0.0,
    h_feats_selective: int = 64,
    n_layers_selective: int = 2,
    dropout_p_selective: float = 0.5,
    model_lr_selective: float = 1e-3,
    wd_selective: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    temperature: float = 1.0,
    p_add: float = 0.1,
    p_remove: float = 0.1,
    d_out: float = 10,
    num_graphs: int = 1,
    device: Union[str, torch.device] = 'cpu',
    seed: int = 0,
    log_training: bool = False,
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
    test_mask: Optional[torch.Tensor] = None,
    dataset_name: str = 'unknown',
    do_hp: bool = False,
    do_self_loop: bool = False,
    do_residual_connections: bool = False
) -> Dict[str, Any]
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Input graph |
| `P_k` | np.ndarray | Permutation matrix for rewiring |
| `h_feats_gcn` | int | Hidden feature dimension for the base GCN |
| `n_layers_gcn` | int | Number of layers for the base GCN |
| `dropout_p_gcn` | float | Dropout probability for the base GCN |
| `model_lr_gcn` | float | Learning rate for the base GCN |
| `wd_gcn` | float | Weight decay for the base GCN |
| `h_feats_selective` | int | Hidden feature dimension for the selective GCN |
| `n_layers_selective` | int | Number of layers for the selective GCN |
| `dropout_p_selective` | float | Dropout probability for the selective GCN |
| `model_lr_selective` | float | Learning rate for the selective GCN |
| `wd_selective` | float | Weight decay for the selective GCN |
| `n_epochs` | int | Maximum number of training epochs |
| `early_stopping` | int | Number of epochs to look back for early stopping |
| `temperature` | float | Temperature for softmax when computing class probabilities |
| `p_add` | float | Probability of adding new edges during rewiring |
| `p_remove` | float | Probability of removing existing edges during rewiring |
| `d_out` | float | Desired output mean degree |
| `num_graphs` | int | Number of rewired graphs to generate |
| `device` | Union[str, torch.device] | Device to perform computations on |
| `seed` | int | Random seed for reproducibility |
| `log_training` | bool | Whether to print training progress |
| `train_mask` | Optional[torch.Tensor] | Boolean mask indicating training nodes |
| `val_mask` | Optional[torch.Tensor] | Boolean mask indicating validation nodes |
| `test_mask` | Optional[torch.Tensor] | Boolean mask indicating test nodes |
| `dataset_name` | str | Name of the dataset |
| `do_hp` | bool | Whether to use high-pass filters |
| `do_self_loop` | bool | Whether to add self-loops |
| `do_residual_connections` | bool | Whether to use residual connections |

## Returns

A dictionary containing the following keys:

| Key | Description |
|-----|-------------|
| `cold_start` | Results for the base GCN including train/val/test accuracy |
| `selective` | Results for the selective GCN including train/val/test accuracy |
| `original_stats` | Statistics for the original graph (nodes, edges, degree, homophily) |
| `rewired_stats` | Statistics for the rewired graph (nodes, edges, degree, homophily, edges added/removed) |

## Pipeline Steps

The `run_bridge_pipeline` function implements the following steps:

1. **Cold-Start GCN Training**: Trains a base GCN on the original graph
2. **Class Probability Prediction**: Uses the trained GCN to infer node classes
3. **Optimal Block Matrix Computation**: Computes an optimal block matrix for rewiring
4. **Graph Rewiring**: Rewires the graph based on the optimal block matrix
5. **Selective GCN Training**: Trains a selective GCN on both the original and rewired graphs

## Example Usage

```python
import dgl
import torch
import numpy as np
from bridge.rewiring import run_bridge_pipeline
from bridge.utils import generate_all_symmetric_permutation_matrices

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Generate permutation matrices
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]  # Choose the first permutation matrix

# Run the rewiring pipeline
results = run_bridge_pipeline(
    g=g,
    P_k=P_k,
    h_feats_gcn=64,
    n_layers_gcn=2,
    dropout_p_gcn=0.5,
    model_lr_gcn=1e-3,
    h_feats_selective=64,
    n_layers_selective=2,
    dropout_p_selective=0.5,
    model_lr_selective=1e-3,
    temperature=1.0,
    p_add=0.1,
    p_remove=0.1,
    d_out=10,
    num_graphs=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Print the results
print(f"Original Graph: Nodes={results['original_stats']['num_nodes']}, Edges={results['original_stats']['num_edges']}")
print(f"Rewired Graph: Nodes={results['rewired_stats']['num_nodes']}, Edges={results['rewired_stats']['num_edges']}")
print(f"Base GCN Test Accuracy: {results['cold_start']['test_acc']:.4f}")
print(f"Selective GCN Test Accuracy: {results['selective']['test_acc']:.4f}")
```

## Notes

- The `P_k` permutation matrix determines the optimal block structure for connecting different classes. Different permutation matrices can lead to different rewiring patterns.
- The `temperature` parameter controls the sharpness of the softmax function when converting logits to class probabilities. Lower values produce more confident (sharper) distributions.
- The `p_add` and `p_remove` parameters control how aggressive the rewiring process is. Higher values lead to more edge modifications.
- The `do_hp` parameter enables the use of high-pass filters, which can be beneficial for heterophilic graphs.
