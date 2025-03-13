---
layout: default
title: graph_utils
parent: API Reference
---

# graph_utils
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `graph_utils` module provides utility functions for manipulating and analyzing graphs. This includes functions for setting random seeds, checking graph properties, building and manipulating adjacency matrices, and computing various graph metrics.

## Basic Utilities

### set_seed

```python
def set_seed(seed: int) -> None
```

Sets random seeds for reproducibility across all relevant libraries.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | int | Random seed value |

#### Example

```python
from bridge.utils import set_seed

# Set seed for reproducible experiments
set_seed(42)
```

### check_symmetry

```python
def check_symmetry(g: dgl.DGLGraph) -> bool
```

Checks if a DGL graph is symmetric (undirected).

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | DGL graph to check |

#### Returns

| Return Type | Description |
|-------------|-------------|
| bool | True if graph is symmetric, False otherwise |

#### Example

```python
import dgl
from bridge.utils import check_symmetry

# Create a directed graph
g = dgl.graph(([0, 1], [1, 2]))

# Check if it's symmetric
is_symmetric = check_symmetry(g)
print(f"Graph is symmetric: {is_symmetric}")

# Create a symmetric graph
g_sym = dgl.graph(([0, 1, 1, 0], [1, 0, 2, 2]))
print(f"Symmetric graph check: {check_symmetry(g_sym)}")
```

### make_symmetric

```python
def make_symmetric(
    g: dgl.DGLGraph, 
    sym_type: str = 'both'
) -> dgl.DGLGraph
```

Makes a DGL graph symmetric by adding reverse edges.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | DGL graph to symmetrize |
| `sym_type` | str | Symmetrization type: 'both', 'upper', or 'lower' |

#### Returns

| Return Type | Description |
|-------------|-------------|
| dgl.DGLGraph | New symmetric DGL graph |

#### Example

```python
import dgl
from bridge.utils import make_symmetric

# Create a directed graph
g = dgl.graph(([0, 1], [1, 2]))

# Make it symmetric by adding all reverse edges
g_sym_both = make_symmetric(g, sym_type='both')

# Make it symmetric using upper triangular part
g_sym_upper = make_symmetric(g, sym_type='upper')

# Make it symmetric using lower triangular part
g_sym_lower = make_symmetric(g, sym_type='lower')

print(f"Original edges: {g.num_edges()}")
print(f"Symmetric (both) edges: {g_sym_both.num_edges()}")
print(f"Symmetric (upper) edges: {g_sym_upper.num_edges()}")
print(f"Symmetric (lower) edges: {g_sym_lower.num_edges()}")
```

## Graph Metrics

### homophily

```python
def homophily(g: dgl.DGLGraph) -> float
```

Computes the homophily score of a graph, defined as the fraction of edges that connect nodes of the same class.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Input graph with node labels in g.ndata['label'] |

#### Returns

| Return Type | Description |
|-------------|-------------|
| float | Homophily score between 0 and 1 |

#### Example

```python
import dgl
import torch
from bridge.utils import homophily

# Create a graph with labels
g = dgl.graph(([0, 0, 1, 1, 2], [1, 2, 0, 2, 1]))
g.ndata['label'] = torch.tensor([0, 0, 1])  # Nodes 0 and 1 are class 0, node 2 is class 1

# Compute homophily
h = homophily(g)
print(f"Graph homophily: {h:.4f}")
```

## Adjacency Matrix Operations

### build_sparse_adj_matrix

```python
def build_sparse_adj_matrix(
    g: dgl.DGLGraph,
    self_loops: bool = False,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor
```

Builds a sparse adjacency matrix from a DGL graph.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Input DGL graph |
| `self_loops` | bool | Whether to include self-loops in the adjacency matrix |
| `sym` | bool | Whether to symmetrize the adjacency matrix |
| `device` | Union[str, torch.device] | Device to place the output tensor |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Sparse adjacency matrix of shape (n_nodes, n_nodes) |

#### Example

```python
import dgl
from bridge.utils import build_sparse_adj_matrix

# Create a graph
g = dgl.graph(([0, 1], [1, 2]))

# Build sparse adjacency matrix with self-loops
A_sparse = build_sparse_adj_matrix(g, self_loops=True)

# Convert to dense for printing
A_dense = A_sparse.to_dense()
print(f"Adjacency matrix with self-loops:\n{A_dense}")
```

### normalize_sparse_adj

```python
def normalize_sparse_adj(A: torch.Tensor) -> torch.Tensor
```

Normalizes a sparse adjacency matrix using D^{-1/2}AD^{-1/2} normalization.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | torch.Tensor | Input sparse adjacency matrix |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Normalized sparse adjacency matrix |

#### Example

```python
import torch
import dgl
from bridge.utils import build_sparse_adj_matrix, normalize_sparse_adj

# Create a graph
g = dgl.graph(([0, 1, 1], [1, 0, 2]))

# Build sparse adjacency matrix
A_sparse = build_sparse_adj_matrix(g)

# Normalize the adjacency matrix
A_norm = normalize_sparse_adj(A_sparse)

# Convert to dense for printing
print(f"Original adjacency:\n{A_sparse.to_dense()}")
print(f"Normalized adjacency:\n{A_norm.to_dense()}")
```

### sparse_add

```python
def sparse_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor
```

Adds two sparse COO tensors.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | torch.Tensor | First sparse tensor |
| `B` | torch.Tensor | Second sparse tensor |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Sum of the two sparse tensors |

### add_sparse

```python
def add_sparse(
    A: torch.Tensor,
    new_indices: torch.Tensor,
    new_values: torch.Tensor
) -> torch.Tensor
```

Adds new indices and values to a sparse COO tensor.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | torch.Tensor | Input sparse tensor |
| `new_indices` | torch.Tensor | New indices to add |
| `new_values` | torch.Tensor | New values to add |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Updated sparse tensor |

## Adjacency Power Functions

### get_A_hat_p

```python
def get_A_hat_p(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor
```

Computes the normalized adjacency matrix raised to the power p.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | int | The power to which the normalized adjacency matrix is raised |
| `g` | dgl.DGLGraph | Input graph |
| `self_loops` | bool | Whether to include self-loops in the adjacency matrix |
| `fix_d` | bool | Whether to fix the degree distribution by normalizing |
| `sym` | bool | Whether to symmetrize the adjacency matrix |
| `device` | Union[str, torch.device] | Device to perform computations on |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | The normalized adjacency matrix raised to the power p |

#### Example

```python
import dgl
from bridge.utils import get_A_hat_p

# Create a graph
g = dgl.graph(([0, 1, 1, 2], [1, 0, 2, 1]))

# Compute A^2
A_2 = get_A_hat_p(p=2, g=g)

# Compute A^3
A_3 = get_A_hat_p(p=3, g=g)

print(f"A^2:\n{A_2}")
print(f"A^3:\n{A_3}")
```

### get_A_p

```python
def get_A_p(
    p: int,
    g: dgl.DGLGraph,
    self_loops: bool = False,
    fix_d: bool = True,
    sym: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor
```

Computes the adjacency matrix raised to the power p (without normalization).

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | int | The power to which the adjacency matrix is raised |
| `g` | dgl.DGLGraph | Input graph |
| `self_loops` | bool | Whether to include self-loops in the adjacency matrix |
| `fix_d` | bool | Placeholder for compatibility; not used here |
| `sym` | bool | Whether to symmetrize the adjacency matrix |
| `device` | Union[str, torch.device] | Device to perform computations on |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | The adjacency matrix raised to the power p |

## Usage in BRIDGE Pipeline

The graph utilities are essential components of the BRIDGE rewiring pipeline:

1. `check_symmetry` determines whether the graph is directed or undirected
2. `build_sparse_adj_matrix` and `normalize_sparse_adj` prepare adjacency matrices for computation
3. `get_A_hat_p` is used for computing higher-order connectivity patterns
4. `homophily` measures the quality of the graph structure

Example integration in the pipeline:

```python
from bridge.utils import check_symmetry, homophily
from bridge.rewiring import create_rewired_graph

# Check if the input graph is symmetric
is_symmetric = check_symmetry(g)
sym_type = 'upper' if is_symmetric else 'asymetric'

# Create a rewired graph
g_rewired = create_rewired_graph(
    g=g,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    p_remove=0.1,
    p_add=0.1,
    sym_type=sym_type
)

# Compare homophily before and after rewiring
h_original = homophily(g)
h_rewired = homophily(g_rewired)

print(f"Original homophily: {h_original:.4f}")
print(f"Rewired homophily: {h_rewired:.4f}")
```

## Related Components

- [local_homophily]({% link api-reference/local-homophily.md %}): Function for computing node-level homophily
- [run_bridge_pipeline]({% link api-reference/bridge-pipeline.md %}): Uses these utilities in the rewiring process
- [create_rewired_graph]({% link api-reference/create-rewired-graph.md %}): Graph rewiring function that uses these utilities
