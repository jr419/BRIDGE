---
layout: default
title: matrix_utils
parent: API Reference
---

# matrix_utils
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `matrix_utils` module provides functions for working with matrices in the context of graph analysis, particularly for stochastic block models and permutation matrices. This module contains utilities for computing optimal block matrices, generating permutation matrices, and statistical analysis.

## Functions

### compute_confidence_interval

```python
def compute_confidence_interval(
    data: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float, float]
```

Computes the mean and confidence interval for a list of values.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | List[float] | List of values to compute confidence interval for |
| `confidence` | float | Confidence level (default: 0.95 for 95% CI) |

#### Returns

| Return Type | Description |
|-------------|-------------|
| Tuple[float, float, float] | Tuple containing (mean, lower_bound, upper_bound) |

#### Example

```python
from bridge.utils import compute_confidence_interval

# Compute 95% confidence interval for a list of accuracy values
accuracies = [0.85, 0.87, 0.84, 0.86, 0.83]
mean, lower, upper = compute_confidence_interval(accuracies)

print(f"Mean accuracy: {mean:.4f}")
print(f"95% CI: ({lower:.4f}, {upper:.4f})")
```

### infer_B

```python
def infer_B(
    g: torch.Tensor, 
    Z: torch.Tensor, 
    sym: bool = True
) -> torch.Tensor
```

Infers the Stochastic Block Model (SBM) block matrix parameters from a graph.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Input graph |
| `Z` | torch.Tensor | One-hot encoding of the block assignment vector |
| `sym` | bool | Whether to enforce symmetry in the block matrix |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Inferred block matrix B |

#### Example

```python
import torch
import dgl
from bridge.utils import infer_B

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Create one-hot encoded class labels
n = g.num_nodes()
labels = g.ndata['label']
num_classes = len(torch.unique(labels))
Z = torch.zeros(n, num_classes)
Z[torch.arange(n), labels] = 1

# Infer the block matrix
B = infer_B(g, Z)
print(f"Inferred block matrix shape: {B.shape}")
```

### optimal_B

```python
def optimal_B(
    g: torch.Tensor,
    y_label: torch.Tensor,
    y_adj: torch.Tensor,
    P_k: np.ndarray,
    lam: float = 0.5,
    k: Optional[int] = None,
    d_out: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]
```

Computes the optimal block matrix for a given graph and permutation matrix.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Input graph |
| `y_label` | torch.Tensor | Node labels |
| `y_adj` | torch.Tensor | Adjacency matrix |
| `P_k` | np.ndarray | Symmetric permutation matrix to use |
| `lam` | float | Regularization parameter |
| `k` | Optional[int] | Number of unique labels (if None, inferred from y_label) |
| `d_out` | Optional[float] | Desired output mean degree (if None, inferred from the graph) |

#### Returns

| Return Type | Description |
|-------------|-------------|
| Tuple[np.ndarray, np.ndarray] | Tuple containing (optimal_block_matrix, original_block_matrix) |

#### Example

```python
import torch
import numpy as np
from bridge.utils import optimal_B, generate_all_symmetric_permutation_matrices

# Generate permutation matrices
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]  # Choose the first permutation matrix

# Compute optimal block matrix
B_opt, B_original = optimal_B(
    g=g,
    y_label=g.ndata['label'],
    y_adj=g.ndata['label'],  # Using true labels for adjacency
    P_k=P_k,
    d_out=10  # Desired mean degree
)

print(f"Original block matrix:\n{B_original}")
print(f"Optimal block matrix:\n{B_opt}")
```

### generate_all_symmetric_permutation_matrices

```python
def generate_all_symmetric_permutation_matrices(k: int) -> List[np.ndarray]
```

Generates all possible k×k symmetric permutation matrices.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | int | Size of the matrices |

#### Returns

| Return Type | Description |
|-------------|-------------|
| List[np.ndarray] | List of all symmetric permutation matrices of size k×k |

#### Example

```python
from bridge.utils import generate_all_symmetric_permutation_matrices

# Generate all symmetric permutation matrices for k=3
matrices = generate_all_symmetric_permutation_matrices(3)

print(f"Number of symmetric permutation matrices for k=3: {len(matrices)}")
for i, P in enumerate(matrices):
    print(f"Matrix {i+1}:\n{P}")
```

### closest_symmetric_permutation_matrix

```python
def closest_symmetric_permutation_matrix(B: np.ndarray) -> np.ndarray
```

Finds the closest symmetric permutation matrix to a given square matrix.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `B` | np.ndarray | Square matrix to approximate |

#### Returns

| Return Type | Description |
|-------------|-------------|
| np.ndarray | The closest symmetric permutation matrix to B |

#### Example

```python
import numpy as np
from bridge.utils import closest_symmetric_permutation_matrix

# Create a non-symmetric matrix
B = np.array([
    [0.8, 0.1, 0.2],
    [0.3, 0.7, 0.0],
    [0.1, 0.3, 0.9]
])

# Find the closest symmetric permutation matrix
P = closest_symmetric_permutation_matrix(B)
print(f"Input matrix:\n{B}")
print(f"Closest symmetric permutation matrix:\n{P}")
```

## Usage in BRIDGE Pipeline

The `matrix_utils` functions are essential components of the BRIDGE rewiring pipeline:

1. `generate_all_symmetric_permutation_matrices` creates potential block structures
2. `optimal_B` computes the optimal block matrix for rewiring
3. `compute_confidence_interval` analyzes results across multiple trials

Example integration in the pipeline:

```python
from bridge.utils import generate_all_symmetric_permutation_matrices, optimal_B
from bridge.rewiring import run_bridge_pipeline

# Generate all permutation matrices
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)

# Test different permutation matrices
results = []
for i, P_k in enumerate(all_matrices):
    result = run_bridge_pipeline(
        g=g,
        P_k=P_k,
        # other parameters...
    )
    results.append(result['selective']['test_acc'])
    print(f"Matrix {i+1}: Test accuracy = {result['selective']['test_acc']:.4f}")

# Compute confidence interval
mean, lower, upper = compute_confidence_interval(results)
print(f"Mean accuracy: {mean:.4f}")
print(f"95% CI: ({lower:.4f}, {upper:.4f})")
```
