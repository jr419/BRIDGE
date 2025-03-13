---
layout: default
title: local_homophily
parent: API Reference
---

# local_homophily
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `local_homophily` function computes the local p-hop homophily for each node in a graph. Local homophily measures how similar a node's features are to its p-hop neighbors with respect to class labels. This metric is crucial for understanding the information flow in graph neural networks and identifying homophilic bottlenecks.

## Function Signature

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

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | int | The 'order' of homophily (number of hops) |
| `g` | dgl.DGLGraph | Input graph |
| `y` | Optional[torch.Tensor] | Node labels of shape (n_nodes,); if None will use g.ndata['label'] |
| `self_loops` | bool | Whether to include self-loops in adjacency |
| `do_hp` | bool | Whether to compute higher-order polynomial version (I - A) |
| `fix_d` | bool | If True, row-normalize adjacency (D^{-1}A) |
| `sym` | bool | Whether to symmetrize adjacency (A <- A + A^T) |
| `device` | Union[str, torch.device] | Device to perform computation on |

## Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Tensor of shape (n_nodes,) containing the local homophily scores for each node |

## Mathematical Definition

For a node i, the local p-homophily is defined as:

$$h^{(p)}_i = \sum_{c=1}^{C} \left( \sum_{j: y_j = c} \hat{A}^p_{ij} \right)^2$$

where:
- $\hat{A}$ is the normalized adjacency matrix
- $p$ is the number of hops
- $y_j$ is the class label of node j
- $C$ is the number of classes

This measure quantifies how much information from nodes of the same class can reach a target node through p-hop paths.

## Usage Examples

### Basic Usage

```python
import torch
import dgl
from bridge.utils import local_homophily

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Compute 2-hop local homophily for each node
homophily_scores = local_homophily(p=2, g=g)

# Print average homophily
print(f"Average 2-hop homophily: {homophily_scores.mean().item():.4f}")

# Find nodes with lowest homophily (bottlenecks)
low_homophily_nodes = torch.argsort(homophily_scores)[:10]
print(f"Nodes with lowest homophily: {low_homophily_nodes}")
```

### Using High-Pass Filter

```python
# Compute high-pass filter version (I - A) homophily
hp_homophily_scores = local_homophily(p=2, g=g, do_hp=True)

# Compare with standard homophily
standard_homophily = local_homophily(p=2, g=g, do_hp=False)

# Print the difference
diff = hp_homophily_scores - standard_homophily
print(f"Mean difference (high-pass - standard): {diff.mean().item():.4f}")
```

### Using Custom Labels

```python
import torch

# Create custom/predicted labels
n_nodes = g.num_nodes()
custom_labels = torch.randint(0, 3, (n_nodes,))  # 3 classes

# Compute homophily with respect to these labels
custom_homophily = local_homophily(p=2, g=g, y=custom_labels)

# Compare with homophily based on true labels
true_homophily = local_homophily(p=2, g=g)  # Uses g.ndata['label']

print(f"Custom labels homophily: {custom_homophily.mean().item():.4f}")
print(f"True labels homophily: {true_homophily.mean().item():.4f}")
```

### Comparison Across Multiple Hop Distances

```python
# Compute homophily at different hop distances
homophily_1hop = local_homophily(p=1, g=g)
homophily_2hop = local_homophily(p=2, g=g)
homophily_3hop = local_homophily(p=3, g=g)

# Print averages
print(f"1-hop homophily: {homophily_1hop.mean().item():.4f}")
print(f"2-hop homophily: {homophily_2hop.mean().item():.4f}")
print(f"3-hop homophily: {homophily_3hop.mean().item():.4f}")
```

## Implementation Details

The `local_homophily` function implements the following algorithm:

1. **Build Sparse Adjacency Matrix**:
   - Converts the DGL graph to a sparse adjacency matrix
   - Optionally adds self-loops or symmetrizes the matrix
   - Normalizes the adjacency matrix using D^{-1/2}AD^{-1/2}

2. **Apply High-Pass Filter** (if `do_hp=True`):
   - Transforms the adjacency matrix to I - A
   - This emphasizes differences between nodes rather than similarities

3. **Compute One-Hot Label Matrix**:
   - Creates a matrix M where M[i,c] = 1 if node i has class c, otherwise 0

4. **Compute (A^p)M**:
   - Raises the adjacency matrix to power p
   - Multiplies by the label matrix
   - This gives the influence from each class on each node through p-hop paths

5. **Compute Homophily Scores**:
   - For each node i, computes the sum of squares of influences from each class
   - This measures how much influence comes from nodes of the same class

The implementation is optimized for sparse graphs and can handle large networks efficiently.

