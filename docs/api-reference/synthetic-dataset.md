---
layout: default
title: SyntheticGraphDataset
parent: API Reference
---

# SyntheticGraphDataset
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `SyntheticGraphDataset` class is a DGL Dataset implementation that generates synthetic graph datasets with controllable properties such as homophily and community structure. This is particularly useful for benchmarking and analyzing graph neural networks under various controlled conditions.

## Class Definition

```python
class SyntheticGraphDataset(DGLDataset):
    def __init__(
        self,
        n: int = 100,
        k: int = 3,
        h: float = 0.8,
        d_mean: float = 3,
        sigma_intra_scalar: float = 0.1,
        sigma_inter_scalar: float = -0.05,
        tau_scalar: float = 1,
        eta_scalar: float = 1,
        in_feats: int = 5,
        d_in: Optional[int] = None,
        alpha: Optional[float] = None,
        sym: bool = True,
        mu: Optional[np.ndarray] = None
    ):
        # Implementation details...
```

## Parameters

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | int | Number of nodes |
| `k` | int | Number of communities (classes/blocks) |
| `h` | float | Homophily ratio (higher values favor intra-community edges) |
| `d_mean` | float | Mean degree scaling factor |
| `sigma_intra_scalar` | float | Scalar for intra-class covariance |
| `sigma_inter_scalar` | float | Scalar for inter-class covariance |
| `tau_scalar` | float | Scalar for the global covariance (shared across nodes) |
| `eta_scalar` | float | Scalar for node-wise noise covariance |
| `in_feats` | int | Dimensionality of node features |
| `d_in` | Optional[int] | Input dimension for feature generation (defaults to k if not provided) |
| `alpha` | Optional[float] | Dirichlet concentration parameter for block proportions |
| `sym` | bool | If True, creates an undirected (symmetric) graph |
| `mu` | Optional[np.ndarray] | Class-specific mean matrix (if None, defaults are used) |

## Methods

### process

```python
def process(self) -> None
```

Generates the synthetic SBM graph and its features. This method builds a block connectivity matrix, generates community assignments, creates the graph adjacency, constructs a DGL graph, and assigns node features.

### _generate_features

```python
def _generate_features(self, num_mu_samples: int = 1) -> None
```

Generates node features using a block-based covariance model, where features are a sum of class-specific mean vectors, global random variation, and node-specific noise.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_mu_samples` | int | Number of independent realizations to generate |

### static generate_features

```python
@staticmethod
def generate_features(
    num_nodes: int,
    num_features: int,
    labels: np.ndarray,
    inter_class_cov: np.ndarray,
    intra_class_cov: np.ndarray,
    global_cov: np.ndarray,
    noise_cov: np.ndarray,
    mu_repeats: int = 1
) -> np.ndarray
```

Static method that generates node features according to a block-based model. For each node, a class-specific mean vector is drawn from a multivariate normal with covariance having intra- and inter-class blocks. Then a global variation and node-specific noise are added.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_nodes` | int | Number of nodes in the graph |
| `num_features` | int | Dimensionality of node features |
| `labels` | np.ndarray | Node labels/community assignments |
| `inter_class_cov` | np.ndarray | Covariance matrix for inter-class relationships |
| `intra_class_cov` | np.ndarray | Covariance matrix for intra-class relationships |
| `global_cov` | np.ndarray | Covariance matrix for global variations |
| `noise_cov` | np.ndarray | Covariance matrix for node-specific noise |
| `mu_repeats` | int | Number of independent realizations to generate |

#### Returns

| Return Type | Description |
|-------------|-------------|
| np.ndarray | Generated features of shape (num_nodes, num_features, mu_repeats) |

### __getitem__

```python
def __getitem__(self, idx: int) -> dgl.DGLGraph
```

Gets the graph at the specified index. Since the dataset contains only one graph, only index 0 is valid.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `idx` | int | Index of the graph to retrieve |

#### Returns

| Return Type | Description |
|-------------|-------------|
| dgl.DGLGraph | The graph at the specified index |

#### Raises

| Exception | Description |
|-----------|-------------|
| IndexError | If the index is out of bounds (only index 0 is valid) |

### __len__

```python
def __len__(self) -> int
```

Gets the number of graphs in the dataset (always 1).

#### Returns

| Return Type | Description |
|-------------|-------------|
| int | The number of graphs in the dataset (always 1) |

## Usage Examples

### Basic Usage

```python
from bridge.datasets import SyntheticGraphDataset

# Create a synthetic dataset with default parameters
dataset = SyntheticGraphDataset()

# Get the generated graph
g = dataset[0]

print(f"Number of nodes: {g.num_nodes()}")
print(f"Number of edges: {g.num_edges()}")
print(f"Feature dimensions: {g.ndata['feat'].shape}")
print(f"Number of classes: {len(torch.unique(g.ndata['label']))}")
```

### Custom Homophily and Size

```python
# Create a heterophilic graph with 1000 nodes and 5 classes
hetero_dataset = SyntheticGraphDataset(
    n=1000,          # 1000 nodes
    k=5,             # 5 classes
    h=0.2,           # Low homophily (heterophilic)
    d_mean=15,       # Higher mean degree
    in_feats=10      # 10-dimensional features
)

g_hetero = hetero_dataset[0]
print(f"Heterophilic graph created with {g_hetero.num_nodes()} nodes and {g_hetero.num_edges()} edges")
```

### Custom Feature Generation

```python
import numpy as np
from bridge.datasets import SyntheticGraphDataset

# Create a dataset with custom feature generation parameters
custom_dataset = SyntheticGraphDataset(
    n=500,
    k=4,
    h=0.6,
    sigma_intra_scalar=0.2,     # Stronger intra-class correlation
    sigma_inter_scalar=-0.1,    # Stronger inter-class distinction
    tau_scalar=0.5,             # Reduced global variation
    eta_scalar=0.8,             # Reduced node-specific noise
    in_feats=8
)

g_custom = custom_dataset[0]

# Analyze the feature distributions by class
labels = g_custom.ndata['label']
features = g_custom.ndata['feat']

# Calculate mean feature vector for each class
for class_id in range(4):
    class_mask = (labels == class_id)
    class_features = features[class_mask]
    class_mean = torch.mean(class_features, dim=0)
    print(f"Class {class_id} mean feature vector: {class_mean}")
```

### Using with GNN Models

```python
import torch
import dgl
from bridge.models import GCN
from bridge.datasets import SyntheticGraphDataset
from bridge.training import train

# Create synthetic dataset with controlled homophily
dataset = SyntheticGraphDataset(
    n=800,
    k=3,
    h=0.7,
    in_feats=5
)
g = dataset[0]

# Create a GCN model
in_feats = g.ndata['feat'].shape[1]
out_feats = len(torch.unique(g.ndata['label']))
model = GCN(
    in_feats=in_feats,
    h_feats=64,
    out_feats=out_feats,
    n_layers=2,
    dropout_p=0.5
)

# Train the model
train_acc, val_acc, test_acc, trained_model = train(
    g=g,
    model=model,
    train_mask=g.ndata['train_mask'],
    val_mask=g.ndata['val_mask'],
    test_mask=g.ndata['test_mask'],
    n_epochs=200,
    early_stopping=30
)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
```

## Implementation Details

The `SyntheticGraphDataset` class generates synthetic graphs using a Stochastic Block Model (SBM), which is a random graph model that incorporates community structure. The implementation involves:

1. **Block Matrix Construction**:
   - Builds a block connectivity matrix B where intra-community connections (on the diagonal) have probability proportional to homophily h
   - Inter-community connections (off-diagonal) have probability proportional to (1-h)/(k-1)

2. **Node Assignment**:
   - Assigns each node to a community/class based on specified or uniform proportions
   - This assignment determines the node labels

3. **Edge Generation**:
   - Creates edges according to the probabilities in the block matrix
   - The probability of an edge between nodes i and j depends on their community assignments

4. **Feature Generation**:
   - Features are generated using a sophisticated covariance structure
   - Each node's features are a sum of:
     - Class-specific mean vector (different for each class)
     - Global random shift (shared across all nodes)
     - Node-specific noise

5. **Train/Val/Test Split**:
   - Creates a random split of nodes for training, validation, and testing
   - By default, uses 80% for training, 10% for validation, and 10% for testing

The dataset is particularly useful for studying the relationship between graph structure (especially homophily) and GNN performance, as it allows precise control over these properties.

## Related Components

- [run_sensitivity_experiment]({% link api-reference/sensitivity-analysis.md %}): Uses synthetic datasets for controlled experiments
- [GCN]({% link api-reference/gcn.md %}): Graph Neural Network model that can be trained on synthetic datasets
- [run_bridge_pipeline]({% link api-reference/bridge-pipeline.md %}): Pipeline that can be applied to synthetic datasets
- [generate_features]({% link api-reference/feature-generation.md %}): Feature generation function used in sensitivity analysis
