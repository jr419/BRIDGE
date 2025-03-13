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

The `run_bridge_experiment` function extends the BRIDGE pipeline to run multiple trials across different data splits. This function is useful for obtaining statistically significant results and confidence intervals on model performance.

## Function Signature

```python
def run_bridge_experiment(
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
    num_repeats: int = 10,
    log_training: bool = False,
    dataset_name: str = 'unknown',
    do_hp: bool = False,
    do_self_loop: bool = False,
    do_residual_connections: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]
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
| `num_repeats` | int | Number of times to repeat the experiment |
| `log_training` | bool | Whether to print training progress |
| `dataset_name` | str | Name of the dataset |
| `do_hp` | bool | Whether to use high-pass filters |
| `do_self_loop` | bool | Whether to add self-loops |
| `do_residual_connections` | bool | Whether to use residual connections |

## Returns

A tuple containing:

1. **Dictionary of aggregated statistics** with means and confidence intervals:
   - `test_acc_mean`: Mean test accuracy
   - `test_acc_ci`: Confidence interval for test accuracy
   - `val_acc_mean`: Mean validation accuracy
   - `val_acc_ci`: Confidence interval for validation accuracy
   - Various statistics about the original and rewired graphs

2. **List of individual trial results**, where each element is the output from a single `run_bridge_pipeline` call

## Usage Examples

### Basic Usage

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
    h_feats_gcn=64,
    n_layers_gcn=2,
    dropout_p_gcn=0.5,
    model_lr_gcn=1e-3,
    num_repeats=5,  # Run 5 trials
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Print the aggregated results
print(f"Mean test accuracy: {stats['test_acc_mean']:.4f}")
print(f"95% CI: ({stats['test_acc_ci'][0]:.4f}, {stats['test_acc_ci'][1]:.4f})")
```

### Using Multiple Dataset Splits

When your dataset has multiple training/validation/test splits, `run_bridge_experiment` can use them automatically:

```python
# If g.ndata['train_mask'] is 2D with shape [num_nodes, num_splits]
# the function will use each split for a separate trial
stats, results_list = run_bridge_experiment(
    g=g,
    P_k=P_k,
    # other parameters...
)
```

### Analyzing Rewiring Statistics

```python
# Analyze how rewiring affects graph properties
print(f"Original density: {stats['original_stats']['density_mean']:.4f}")
print(f"Rewired density: {stats['rewired_stats']['density_mean']:.4f}")
print(f"Original homophily: {stats['original_stats']['homophily_mean']:.4f}")
print(f"Rewired homophily: {stats['rewired_stats']['homophily_mean']:.4f}")
print(f"Average edges added: {stats['edges_added_mean']:.1f}")
print(f"Average edges removed: {stats['edges_removed_mean']:.1f}")
```

## Implementation Details

The `run_bridge_experiment` function is a wrapper around `run_bridge_pipeline` that:

1. Runs the pipeline multiple times, either:
   - Using different random seeds for each trial
   - Using different dataset splits if available

2. Collects statistics across trials, computing:
   - Mean performance
   - Confidence intervals using bootstrap resampling
   - Aggregated statistics about graph changes

3. Handles multiple dataset splits automatically when available in the input graph

The function is particularly useful for:
- Obtaining reliable performance estimates with confidence intervals
- Controlling for randomness in model initialization and training
- Comparing different rewiring strategies across multiple trials

## Related Components

- [run_bridge_pipeline](api-reference/bridge-pipeline.html): The core pipeline function that this function wraps
