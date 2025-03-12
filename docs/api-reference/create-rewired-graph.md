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

The `create_rewired_graph` function is a low-level function that creates a rewired version of a graph using predicted class probabilities and an optimal block matrix. This function is a core component of the BRIDGE rewiring pipeline.

## Function Signature

```python
def create_rewired_graph(
    g: dgl.DGLGraph,
    B_opt_tensor: torch.Tensor,
    pred: torch.Tensor,
    Z_pred: torch.Tensor,
    p_remove: float,
    p_add: float,
    sym_type: str = 'upper',
    device: Union[str, torch.device] = 'cpu'
) -> dgl.DGLGraph
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Original graph to rewire |
| `B_opt_tensor` | torch.Tensor | Optimal block matrix (k×k tensor, where k is the number of classes) |
| `pred` | torch.Tensor | Predicted class labels for each node |
| `Z_pred` | torch.Tensor | Predicted class probabilities for each node (softmax outputs) |
| `p_remove` | float | Probability of removing existing edges |
| `p_add` | float | Probability of adding new edges |
| `sym_type` | str | Type of symmetry to enforce: 'upper', 'lower', or 'asymetric' |
| `device` | Union[str, torch.device] | Device to perform computations on |

## Returns

| Return Type | Description |
|-------------|-------------|
| dgl.DGLGraph | The rewired graph |

## Detailed Description

The `create_rewired_graph` function implements a likelihood-based rewiring strategy. The process consists of the following steps:

1. **Compute Optimal Edge Probabilities**: Calculate the probability of an edge existing between any two nodes based on their predicted class probabilities and the optimal block matrix.

2. **Determine Edge Modifications**:
   - For existing edges: Keep with probability `(1 - p_remove * (1 - A_opt_p))`
   - For non-existing edges: Add with probability `(p_add * A_opt_p)`

3. **Sample New Adjacency Matrix**: Sample a new adjacency matrix based on these probabilities.

4. **Ensure Symmetry** (if required): Enforce symmetry according to the specified `sym_type`.

5. **Build Rewired Graph**: Construct a new DGL graph based on the modified adjacency matrix.

The function uses the probabilistic approach to rewiring rather than a deterministic one, which helps avoid over-fitting to the predicted class structure and maintains some of the original graph's characteristics.

## Usage Examples

### Basic Usage

```python
import torch
import dgl
import numpy as np
from bridge.rewiring import create_rewired_graph
from bridge.utils import generate_all_symmetric_permutation_matrices, optimal_B

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Assume we have trained a GCN and obtained predictions
# pred: tensor of predicted class labels
# Z_pred: tensor of predicted class probabilities (softmax outputs)

# Generate a permutation matrix for the optimal block structure
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]  # Choose the first permutation matrix

# Compute class proportions
n_nodes = g.num_nodes()
pi = Z_pred.cpu().numpy().sum(0) / n_nodes
Pi_inv = np.diag(1/pi)

# Compute the optimal block matrix
d_out = 10  # Desired mean degree
B_opt = (d_out/k) * Pi_inv @ P_k @ Pi_inv
B_opt_tensor = torch.tensor(B_opt, dtype=torch.float32)

# Create the rewired graph
g_rewired = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    p_remove=0.1,
    p_add=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Check the number of edges before and after rewiring
print(f"Original graph: {g.num_edges()} edges")
print(f"Rewired graph: {g_rewired.num_edges()} edges")
```

### Different Symmetry Types

```python
# Create a rewired graph with upper triangular symmetry
g_rewired_upper = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    p_remove=0.1,
    p_add=0.1,
    sym_type='upper'  # Upper triangular symmetry
)

# Create a rewired graph with lower triangular symmetry
g_rewired_lower = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    p_remove=0.1,
    p_add=0.1,
    sym_type='lower'  # Lower triangular symmetry
)

# Create a rewired graph without enforcing symmetry
g_rewired_asym = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    p_remove=0.1,
    p_add=0.1,
    sym_type='asymetric'  # No symmetry enforcement
)
```

## Implementation Details

The function implements the following rewiring algorithm:

1. **Optimal Edge Probability Calculation**:
   ```python
   A_opt_p = (Z_pred @ B_opt_tensor @ Z_pred.T) / n_nodes
   ```
   
   This matrix gives the probability of an edge existing between each pair of nodes based on their predicted class memberships and the optimal block structure.

2. **Edge Modification Probabilities**:
   ```python
   A_p = A_old * (1 - p_remove * (1 - A_opt_p)) + (1 - A_old) * A_opt_p * p_add
   ```
   
   Where:
   - `A_old` is the original adjacency matrix
   - `p_remove` controls how likely existing edges are to be removed
   - `p_add` controls how likely new edges are to be added
   - `A_opt_p` biases additions and preservations toward the optimal structure

3. **Symmetry Enforcement**:
   
   Depending on the `sym_type` parameter:
   - `'upper'`: Ensure symmetry using the upper triangular part
   - `'lower'`: Ensure symmetry using the lower triangular part
   - `'asymetric'`: No symmetry enforcement

4. **Graph Construction**:
   
   The function reconstructs the graph using the new adjacency matrix while preserving all node features from the original graph.

The rewiring process balances several objectives:
- Moving toward the optimal class-based connectivity structure
- Preserving some aspects of the original graph
- Introducing stochasticity to avoid overfitting

## Related Components

- [run_bridge_pipeline]({% link api-reference/bridge-pipeline.md %}): Uses this function as part of the complete rewiring pipeline
- [local_homophily]({% link api-reference/local-homophily.md %}): Used to evaluate the quality of the rewired graph
