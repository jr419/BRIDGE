---
layout: default
title: class_bottlenecking_score
parent: API Reference
---

# class_bottlenecking_score
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `class_bottlenecking_score` function computes the local class-bottlenecking score for each node in a graph over p hops. This score measures how strongly a node connects to same-class nodes within p hops. It is crucial for understanding the information flow in graph neural networks and identifying class-bottlenecks.

## Function Signature

```python
def class_bottlenecking_score(
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
| torch.Tensor | Tensor of shape (n_nodes,) containing the class-bottlenecking scores for each node |

## Mathematical Definition

For a node i, the local class-bottlenecking score (order p) is defined as:

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
from bridge.utils import class_bottlenecking_score

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Compute 2-hop class-bottlenecking score for each node
cb_scores = class_bottlenecking_score(p=2, g=g)

# Print average class-bottlenecking score
print(f"Average 2-hop class-bottlenecking score: {cb_scores.mean().item():.4f}")

# Find nodes with lowest scores (bottlenecks)
low_cb_nodes = torch.argsort(cb_scores)[:10]
print(f"Nodes with lowest class-bottlenecking score: {low_cb_nodes}")
```

### Using High-Pass Filter

```python
# Compute high-pass filter version (I - A) of the score
hp_scores = class_bottlenecking_score(p=2, g=g, do_hp=True)

# Compare with the standard version
standard_scores = class_bottlenecking_score(p=2, g=g, do_hp=False)

# Print the difference
diff = hp_scores - standard_scores
print(f"Mean difference (high-pass - standard): {diff.mean().item():.4f}")
```

### Using Custom Labels

```python
import torch

# Create custom/predicted labels
n_nodes = g.num_nodes()
custom_labels = torch.randint(0, 3, (n_nodes,))  # 3 classes

# Compute scores with respect to these labels
custom_scores = class_bottlenecking_score(p=2, g=g, y=custom_labels)

# Compare with scores based on true labels
true_scores = class_bottlenecking_score(p=2, g=g)  # Uses g.ndata['label']

print(f"Custom labels class-bottlenecking: {custom_scores.mean().item():.4f}")
print(f"True labels class-bottlenecking: {true_scores.mean().item():.4f}")
```

### Comparison Across Multiple Hop Distances

```python
# Compute scores at different hop distances
cb_1hop = class_bottlenecking_score(p=1, g=g)
cb_2hop = class_bottlenecking_score(p=2, g=g)
cb_3hop = class_bottlenecking_score(p=3, g=g)

# Print averages
print(f"1-hop class-bottlenecking: {cb_1hop.mean().item():.4f}")
print(f"2-hop class-bottlenecking: {cb_2hop.mean().item():.4f}")
print(f"3-hop class-bottlenecking: {cb_3hop.mean().item():.4f}")
```

## Implementation Details

The `class_bottlenecking_score` function implements the following algorithm:

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

5. **Compute Class-Bottlenecking Scores**:
   - For each node i, computes the sum of squares of influences from each class
   - This measures how much influence comes from nodes of the same class

The implementation is optimized for sparse graphs and can handle large networks efficiently.
