---
layout: default
title: run_bridge_experiment
parent: API Reference
---

# run_bridge_experiment
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `run_bridge_experiment` function extends the BRIDGE pipeline to run multiple trials across different data splits. It returns aggregated statistics and confidence intervals. Rewiring uses hard predictions (argmax) and full resampling; there is no temperature and no p_add/p_remove. Only standard models are trained.

## Function Signature

```python
def run_bridge_experiment(
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
    num_splits: int = 10,
    log_training: bool = False,
    dataset_name: str = 'unknown',
    do_self_loop: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]
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
| `num_splits` | int | Number of times to repeat the experiment (or inferred from masks) |
| `log_training` | bool | Whether to print training progress |
| `dataset_name` | str | Name of the dataset |
| `do_self_loop` | bool | Whether to add self-loops |

## Returns

A tuple containing:

1. Dictionary of aggregated statistics with confidence intervals:
   - `test_acc_mean`, `test_acc_ci`
   - `val_acc_mean`, `val_acc_ci`
   - Graph change statistics (density, homophily, degree, edges added/removed)

2. List of individual trial results (each from `run_bridge_pipeline`).

## Usage Examples

```python
import dgl
import torch
import numpy as np
from bridge.rewiring import run_bridge_experiment
from bridge.utils import generate_all_symmetric_permutation_matrices

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Generate permutation matrices
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]  # Choose the first permutation matrix

# Run the experiment with multiple trials
stats, results_list = run_bridge_experiment(
    g=g,
    P_k=P_k,
    h_feats_mpnn=64,
    n_layers_mpnn=2,
    dropout_p_mpnn=0.5,
    model_lr_mpnn=1e-3,
    num_splits=5,  # Run 5 trials
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Print the aggregated results
print(f"Mean test accuracy: {stats['test_acc_mean']:.4f}")
print(f"95% CI: ({stats['test_acc_ci'][0]:.4f}, {stats['test_acc_ci'][1]:.4f})")
```

## Implementation Details

- Wraps `run_bridge_pipeline` across multiple splits or seeds.
- Aggregates metrics and computes confidence intervals.
- Uses hard predictions and full-resampling rewiring.

## Related Components

- [run_bridge_pipeline](api-reference/bridge-pipeline.html)
