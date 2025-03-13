---
layout: default
title: homophily_metrics
parent: API Reference
---

# homophily_metrics
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The homophily metrics module provides functions for computing various measures of homophily (similarity between connected nodes) in graph neural networks. These metrics are essential for understanding the information flow and potential bottlenecks in message passing architectures.

## Metrics Functions

### local_autophily

```python
def local_autophily(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> np.ndarray
```

Computes the local autophily for each node in the graph. Autophily measures how similar a node is to itself through its neighborhood, regardless of class labels.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | int | The order of the local autophily |
| `g` | dgl.DGLGraph | Input graph |
| `self_loops` | bool | Whether to include self-loops in the adjacency matrix |
| `fix_d` | bool | Whether to fix the degree distribution by normalizing |
| `sym` | bool | Whether to symmetrize the adjacency matrix |
| `device` | Union[str, torch.device] | Device to perform computations on |

#### Returns

| Return Type | Description |
|-------------|-------------|
| np.ndarray | An array containing the local autophily scores for each node |

#### Mathematical Definition

For a node i, the local p-autophily is defined as:

$$\omega^{(p)}_i = \sum_j (\hat{A}^p_{ij})^2$$

where $\hat{A}$ is the normalized adjacency matrix.

#### Example

```python
import torch
import dgl
from bridge.utils import local_autophily

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Compute 2-hop local autophily for each node
autophily_scores = local_autophily(p=2, g=g)

# Print average autophily
print(f"Average 2-hop autophily: {autophily_scores.mean():.4f}")
```

### local_total_connectivity

```python
def local_total_connectivity(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> np.ndarray
```

Computes the local total connectivity for each node in the graph. Total connectivity measures how well connected a node is to its p-hop neighborhood.

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
| np.ndarray | An array containing the local total connectivity scores for each node |

#### Mathematical Definition

For a node i, the local p-total connectivity is defined as:

$$\tau^{(p)}_i = \left(\sum_j \hat{A}^p_{ij}\right)^2$$

where $\hat{A}$ is the normalized adjacency matrix.

#### Example

```python
import torch
import dgl
from bridge.utils import local_total_connectivity

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Compute 2-hop local total connectivity for each node
connectivity_scores = local_total_connectivity(p=2, g=g)

# Print average connectivity
print(f"Average 2-hop connectivity: {connectivity_scores.mean():.4f}")
```

### local_homophily

```python
def local_homophily(
    p: int, 
    g: dgl.DGLGraph, 
    y: Optional[torch.Tensor] = None,
    self_loops: bool = False,
    do_hp: bool = False,
    fix_d: bool = True, 
    sym: bool = False, 
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor
```

Computes the local p-homophily for each node in the graph. Local homophily measures how similar a node's features are to its p-hop neighbors with respect to class labels.

See full documentation: [local_homophily]({% link api-reference/local-homophily.md %})

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

The homophily metrics provided in this module are particularly useful for:

1. **Understanding GNN Performance**: Higher local homophily typically leads to better GNN performance for standard architectures.

2. **Identifying Bottlenecks**: Nodes with low local homophily may act as bottlenecks for information flow.

3. **Selecting Graph Rewiring Strategies**: The metrics can guide the selection of optimal permutation matrices for BRIDGE rewiring.

4. **Choosing Between High-Pass and Low-Pass Filters**: The relationship between homophily and heterophily can inform the choice of GNN architecture.

## Usage in BRIDGE Pipeline

In the BRIDGE rewiring pipeline, homophily metrics are used to:

1. Evaluate the quality of the original graph
2. Guide the rewiring process to increase higher-order homophily
3. Create node masks for the selective GCN based on local homophily
4. Analyze the improvement in homophily after rewiring

Example:

```python
from bridge.utils import local_homophily
from bridge.rewiring import run_bridge_pipeline

# Run the BRIDGE pipeline
results = run_bridge_pipeline(
    g=g,
    P_k=P_k,
    # other parameters...
)

# Compare homophily before and after rewiring
original_homophily = results['original_stats']['mean_local_homophily']
rewired_homophily = results['rewired_stats']['mean_local_homophily']

print(f"Original homophily: {original_homophily:.4f}")
print(f"Rewired homophily: {rewired_homophily:.4f}")
print(f"Improvement: {(rewired_homophily - original_homophily):.4f}")
```
