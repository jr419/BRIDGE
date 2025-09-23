---
layout: default
title: create_rewired_graph
parent: API Reference
---

# create_rewired_graph
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `create_rewired_graph` function creates a rewired version of a graph using hard class assignments (one-hot) and an optimal block matrix. Rewiring is performed by fully resampling the adjacency from the optimal edge probabilities. There are no p_add/p_remove parameters and no temperature scaling; predictions are treated as hard labels.

## Function Signature

```python
def create_rewired_graph(
    g: dgl.DGLGraph,
    B_opt_tensor: torch.Tensor,
    pred: torch.Tensor,
    Z_pred: torch.Tensor,
    sym_type: str = 'upper',
    device: Union[str, torch.device] = 'cpu'
) -> dgl.DGLGraph
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Original graph to rewire |
| `B_opt_tensor` | torch.Tensor | Optimal block matrix (k×k tensor, where k is the number of classes) |
| `pred` | torch.Tensor | Predicted class labels for each node (hard assignments) |
| `Z_pred` | torch.Tensor | One-hot class assignments for each node (shape: [n_nodes, k]) |
| `sym_type` | str | Type of symmetry to enforce: 'upper', 'lower', or 'asymetric' |
| `device` | Union[str, torch.device] | Device to perform computations on |

## Returns

| Return Type | Description |
|-------------|-------------|
| dgl.DGLGraph | The rewired graph |

## Detailed Description

The `create_rewired_graph` function implements a full-resampling rewiring strategy. The process consists of the following steps:

1. Compute optimal edge probabilities based on hard class assignments and the optimal block matrix:
   - `A_opt_p = (Z_pred @ B_opt_tensor @ Z_pred.T) / n_nodes`
   - Clamp to [0, 1] and set NaNs to 0 if any occur.
2. Fully resample the adjacency by sampling every potential edge independently:
   - `A ~ Bernoulli(A_opt_p)`
3. Ensure symmetry (if requested) by mirroring the chosen triangular part.
4. Build the rewired graph by removing all edges in a clone of `g` and adding edges where `A > 0`.

This approach replaces the older partial add/remove scheme and always uses hard predictions (no softmax temperature).

## Usage Examples

### Basic Usage

```python
import torch
import dgl
import numpy as np
from bridge.rewiring import create_rewired_graph
from bridge.utils import generate_all_symmetric_permutation_matrices

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Assume we have trained a model and obtained hard predictions
# pred: tensor of predicted class labels (argmax of logits)
pred = g.ndata['label']  # example placeholder; replace with your model's argmax predictions

# Build one-hot Z_pred
n_nodes = g.num_nodes()
k = int(g.ndata['label'].max().item()) + 1
Z_pred = torch.zeros(n_nodes, k)
Z_pred.scatter_(1, pred.view(-1, 1), 1.0)

# Generate a permutation matrix for the optimal block structure
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]  # Choose a permutation

# Compute the optimal block matrix
pi = Z_pred.numpy().sum(0) / n_nodes
Pi_inv = np.diag(1 / np.clip(pi, 1e-8, None))
d_out = 10  # Desired mean degree
B_opt = (d_out / k) * Pi_inv @ P_k @ Pi_inv
B_opt_tensor = torch.tensor(B_opt, dtype=torch.float32)

# Create the rewired graph
g_rewired = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Original edges: {g.num_edges()} | Rewired edges: {g_rewired.num_edges()}")
```

### Different Symmetry Types

```python
# Upper triangular symmetry (default)
g_rewired_upper = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    sym_type='upper'
)

# Lower triangular symmetry
g_rewired_lower = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    sym_type='lower'
)

# No symmetry enforcement (asymmetric)
g_rewired_asym = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    sym_type='asymetric'
)
```

## Implementation Details

- Optimal Edge Probability Calculation:
  ```python
  A_opt_p = (Z_pred @ B_opt_tensor @ Z_pred.T) / n_nodes
  A_opt_p = torch.clamp(A_opt_p, 0, 1)
  A_opt_p[torch.isnan(A_opt_p)] = 0
  ```
- Full Resampling:
  ```python
  A = torch.bernoulli(A_opt_p)
  ```
- Symmetry Enforcement:
  - `'upper'`: `A = torch.triu(A) + torch.triu(A, 1).T`
  - `'lower'`: `A = torch.tril(A) + torch.tril(A, -1).T`
  - `'asymetric'`: no mirroring
- Graph Construction:
  - Clone `g`, remove all edges, and add edges where `A > 0`.

Notes:
- Predictions are hard labels (argmax); there is no temperature scaling and no partial add/remove probabilities.
- Ensure shapes: `Z_pred` is `[n_nodes, k]` and `B_opt_tensor` is `[k, k]`.
