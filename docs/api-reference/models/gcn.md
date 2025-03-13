---
layout: default
title: GCN
parent: API Reference
---

# GCN (Graph Convolutional Network)
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `GCN` class implements a Graph Convolutional Network as described in [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907). This implementation supports variable depth and optional residual connections.

## Class Definition

```python
class GCN(nn.Module):
    def __init__(
        self, 
        in_feats: int, 
        h_feats: int, 
        out_feats: int, 
        n_layers: int, 
        dropout_p: float, 
        activation: Callable = F.relu, 
        bias: bool = True, 
        residual_connection: bool = False,
        do_hp: bool = False
    ):
        # Implementation details...
        
    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        # Implementation details...
```

## Parameters

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_feats` | int | Input feature dimension |
| `h_feats` | int | Hidden feature dimension |
| `out_feats` | int | Output feature dimension |
| `n_layers` | int | Number of GCN layers |
| `dropout_p` | float | Dropout probability |
| `activation` | Callable | Activation function to use (default: F.relu) |
| `bias` | bool | Whether to use bias in GraphConv layers |
| `residual_connection` | bool | Whether to use residual connections |
| `do_hp` | bool | Whether to use HPGraphConv instead of GraphConv |

### Forward Method Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Input graph |
| `features` | torch.Tensor | Node feature matrix with shape (num_nodes, in_feats) |

## Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Node embeddings with shape (num_nodes, out_feats) |

## Usage Examples

### Basic Usage

```python
import torch
import dgl
from bridge.models import GCN

# Create a simple graph
g = dgl.graph(([0, 1], [1, 2]))
g.ndata['feat'] = torch.randn(3, 5)  # 3 nodes, 5 features each

# Create a GCN model
model = GCN(
    in_feats=5,     # Input feature size
    h_feats=16,     # Hidden layer size
    out_feats=2,    # Output size (e.g., number of classes)
    n_layers=2,     # Number of GCN layers
    dropout_p=0.5   # Dropout rate
)

# Forward pass
output = model(g, g.ndata['feat'])
print(output.shape)  # Should be (3, 2)
```

### With High-Pass Filter

```python
# Create a GCN model with high-pass filters
high_pass_model = GCN(
    in_feats=5,
    h_feats=16,
    out_feats=2,
    n_layers=2,
    dropout_p=0.5,
    do_hp=True  # Use high-pass filters
)

# Forward pass
output = high_pass_model(g, g.ndata['feat'])
```

### With Residual Connections

```python
# Create a GCN model with residual connections
residual_model = GCN(
    in_feats=5,
    h_feats=16,
    out_feats=2,
    n_layers=3,
    dropout_p=0.5,
    residual_connection=True  # Use residual connections
)

# Forward pass
output = residual_model(g, g.ndata['feat'])
```

## Implementation Details

The GCN class consists of a stack of graph convolution layers. If `do_hp` is False, the model uses DGL's standard `GraphConv` layers. If `do_hp` is True, the model uses the `HPGraphConv` layers, which implement a high-pass filter.

The forward pass processes the input features through the layers, applying activation functions and dropout after each layer except the final one.

For each layer in the stack:
1. Apply graph convolution to the input features
2. If not the output layer, apply activation function
3. If not the output layer, apply dropout

Note that if `do_hp` is True, the architecture changes to use high-pass filters, which emphasize the difference between a node's features and its neighbors' features (computed as I - GCN).

## Related Components

- [HPGraphConv](api-reference/hpgraphconv.html): High-Pass Graph Convolution layer used when `do_hp=True`
- [SelectiveGCN](api-reference/selectivegcn.html): Extension of GCN that can operate on multiple graph versions
