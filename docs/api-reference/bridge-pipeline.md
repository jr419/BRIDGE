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

Rewiring now always uses hard predictions (argmax over logits) and fully resamples the adjacency from optimal edge probabilities. There is no temperature parameter and no p_add/p_remove. The pipeline trains a standard model only (no selective models).

## Function Signature

```python
def run_bridge_pipeline(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
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
| `h_feats_mpnn` | int | Hidden feature dimension for the base model |
| `n_layers_mpnn` | int | Number of layers for the base model |
| `dropout_p_mpnn` | float | Dropout probability for the base model |
| `model_lr_mpnn` | float | Learning rate for the base model |
| `wd_mpnn` | float | Weight decay for the base model |
| `n_epochs` | int | Maximum number of training epochs |
| `early_stopping` | int | Number of epochs to look back for early stopping |
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
| `cold_start` | Results for the base model including train/val/test accuracy |
| `rewired` | Results for the standard model trained on the rewired graph |
| `original_stats` | Statistics for the original graph (nodes, edges, degree, homophily) |
| `rewired_stats` | Statistics for the rewired graph (nodes, edges, degree, homophily, edges added/removed) |

## Pipeline Steps

The `run_bridge_pipeline` function implements the following steps:

1. Cold-Start model training on the original graph
2. Hard class prediction (argmax over logits, no temperature)
3. Optimal block matrix computation
4. Graph rewiring via full resampling from optimal probabilities
5. Train a standard model on the rewired graph

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
P_k = all_matrices[0]

# Run the rewiring pipeline
results = run_bridge_pipeline(
    g=g,
    P_k=P_k,
    h_feats_mpnn=64,
    n_layers_mpnn=2,
    dropout_p_mpnn=0.5,
    model_lr_mpnn=1e-3,
    d_out=10,
    num_graphs=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Original Graph: Nodes={results['original_stats']['num_nodes']}, Edges={results['original_stats']['num_edges']}")
print(f"Rewired Graph: Nodes={results['rewired_stats']['num_nodes']}, Edges={results['rewired_stats']['num_edges']}")
print(f"Base Test Accuracy: {results['cold_start']['test_acc']:.4f}")
print(f"Rewired Test Accuracy: {results['rewired']['test_acc']:.4f}")
```

## Notes

- `P_k` determines the optimal block structure across classes.
- Predictions are hard; there is no softmax temperature and no partial add/remove probabilities.
- The `do_hp` parameter enables high-pass filters, which can be beneficial for heterophilic graphs.
