---
layout: default
title: Homophily Metrics
parent: API Reference
---

# Homophily Metrics

Documentation for computing local and global connectivity metrics, including
class-bottlenecking score, self bottlenecking score (self-connectivity), and
total bottlenecking score.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The homophily metrics module provides functions for computing various measures of homophily (similarity between connected nodes) in graph neural networks. These metrics are essential for understanding the information flow and potential bottlenecks in message passing architectures.

## Metrics Functions

### self_bottlenecking_score

```python
def self_bottlenecking_score(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> np.ndarray
```

Computes the self bottlenecking score (local self-connectivity) for each node. Self-connectivity measures how strongly a node reconnects to itself through its neighborhood, regardless of class labels.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | int | The order for the self bottlenecking score |
| `g` | dgl.DGLGraph | Input graph |
| `self_loops` | bool | Whether to include self-loops in the adjacency matrix |
| `fix_d` | bool | Whether to fix the degree distribution by normalizing |
| `sym` | bool | Whether to symmetrize the adjacency matrix |
| `device` | Union[str, torch.device] | Device to perform computations on |

#### Returns

| Return Type | Description |
|-------------|-------------|
| np.ndarray | An array containing the self bottlenecking scores (self-connectivity) for each node |

#### Mathematical Definition

For a node i, the local p-self-connectivity (self bottlenecking score) is defined as:

$$\omega^{(p)}_i = \sum_j (\hat{A}^p_{ij})^2$$

where $\hat{A}$ is the normalized adjacency matrix.

#### Example

```python
import torch
import dgl
from bridge.utils import self_bottlenecking_score

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Compute 2-hop self bottlenecking score (self-connectivity) for each node
self_connectivity_scores = self_bottlenecking_score(p=2, g=g)

# Print average self-connectivity
print(f"Average 2-hop self-connectivity: {self_connectivity_scores.mean():.4f}")
```

### total_bottlenecking_score

```python
def total_bottlenecking_score(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> np.ndarray
```

Computes the total bottlenecking score for each node in the graph. This measures how well a node connects within its p-hop neighborhood overall.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | int | The order of the local connectivity |
| `g` | dgl.DGLGraph | Input graph |
| `self_loops` | bool | Whether to include self-loops in the adjacency matrix |
| `fix_d` | bool | Whether to fix the degree distribution by normalizing |
| `sym` | bool | Whether to symmetrize the adjacency matrix |
| `device` | Union[str, torch.device] | Device to perform computations on |

#### Returns

| Return Type | Description |
|-------------|-------------|
| np.ndarray | An array containing the total bottlenecking scores for each node |

#### Mathematical Definition

For a node i, the total bottlenecking score at order p is defined as:

$$\tau^{(p)}_i = \left(\sum_j \hat{A}^p_{ij}\right)^2$$

where $\hat{A}$ is the normalized adjacency matrix.

#### Example

```python
import torch
import dgl
from bridge.utils import total_bottlenecking_score

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Compute 2-hop total bottlenecking score for each node
connectivity_scores = total_bottlenecking_score(p=2, g=g)

# Print average total bottlenecking score
print(f"Average 2-hop total bottlenecking score: {connectivity_scores.mean():.4f}")
```

### class_bottlenecking_score

```python
def class_bottlenecking_score(
    p: int, 
    g: dgl.DGLGraph, 
    y: Optional[torch.Tensor] = None,
    self_loops: bool = False,
    fix_d: bool = True, 
    sym: bool = False, 
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor
```

Computes the local class-bottlenecking score for each node in the graph. This measures how strongly a node connects to same-class nodes within p hops.

See full documentation: [class_bottlenecking_score]({% link api-reference/class-bottlenecking-score.md %})

## Utility Functions

### compute_label_matrix

```python
def compute_label_matrix(
    y: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]
```

Creates a one-hot label matrix from a label vector.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | torch.Tensor | Label tensor of shape (n_nodes,) |
| `device` | Optional[torch.device] | Device to place the output tensor |

#### Returns

| Return Type | Description |
|-------------|-------------|
| Tuple[torch.Tensor, torch.Tensor] | Tuple containing (one_hot_label_matrix, unique_class_labels) |

#### Example

```python
import torch
from bridge.utils import compute_label_matrix

# Create some example labels
labels = torch.tensor([0, 1, 2, 1, 0, 2])

# Convert to one-hot matrix
one_hot, classes = compute_label_matrix(labels)

print(f"One-hot label matrix shape: {one_hot.shape}")
print(f"Unique classes: {classes}")
```

## Matrix Operation Functions

### power_adj_times_matrix

```python
def power_adj_times_matrix(
    A: torch.Tensor, 
    M: torch.Tensor, 
    p: int
) -> torch.Tensor
```

Computes (A^p) M using repeated multiplication in sparse form.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | torch.Tensor | Sparse adjacency matrix |
| `M` | torch.Tensor | Dense matrix to multiply with |
| `p` | int | Power to raise the adjacency matrix to |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Result of (A^p) M |

### sparse_mm

```python
def sparse_mm(
    sparse_A: torch.Tensor, 
    dense_B: torch.Tensor
) -> torch.Tensor
```

Performs sparse-dense matrix multiplication.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sparse_A` | torch.Tensor | Sparse matrix of shape (m, n) |
| `dense_B` | torch.Tensor | Dense matrix of shape (n, k) |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Result of sparse_A @ dense_B of shape (m, k) |

## Relationship to Graph Neural Networks

These metrics are particularly useful for:

1. **Understanding GNN Performance**: Higher class-bottlenecking scores typically lead to better GNN performance for standard architectures.

2. **Identifying Bottlenecks**: Nodes with low class-bottlenecking scores may act as bottlenecks for information flow.

3. **Selecting Graph Rewiring Strategies**: The metrics can guide the selection of optimal permutation matrices for BRIDGE rewiring.



## Usage in BRIDGE Pipeline

In the BRIDGE rewiring pipeline, these metrics are used to:

1. Evaluate the quality of the original graph
2. Guide the rewiring process to increase higher-order class connectivity
3. Analyze the improvement in class-bottlenecking score after rewiring

Example:

```python
from bridge.utils import class_bottlenecking_score
from bridge.rewiring import run_bridge_pipeline

# Run the BRIDGE pipeline
results = run_bridge_pipeline(
    g=g,
    P_k=P_k,
    # other parameters...
)

# Compare class-bottlenecking score before and after rewiring
original_cb = results['original_stats']['mean_class_bottlenecking_score']
rewired_cb = results['rewired_stats']['mean_class_bottlenecking_score']

print(f"Original class-bottlenecking score: {original_cb:.4f}")
print(f"Rewired class-bottlenecking score: {rewired_cb:.4f}")
print(f"Improvement: {(rewired_cb - original_cb):.4f}")
```
