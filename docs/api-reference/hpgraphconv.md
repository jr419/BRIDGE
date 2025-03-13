---
layout: default
title: HPGraphConv
parent: API Reference
---

# HPGraphConv (High-Pass Graph Convolution)
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `HPGraphConv` class implements a High-Pass Graph Convolution layer. This layer implements a high-pass filter for graph convolution, represented as I - GCN, which emphasizes the difference between a node's features and its neighbors' features.

## Class Definition

```python
class HPGraphConv(nn.Module):
    def __init__(
        self, 
        in_feats: int, 
        out_feats: int, 
        activation: Optional[Callable] = None, 
        bias: bool = True,
        weight: bool = True,
        allow_zero_in_degree: bool = True
    ):
        # Implementation details...
        
    def forward(self, g: dgl.DGLGraph, features: torch.Tensor, do_hp: bool = True) -> torch.Tensor:
        # Implementation details...
```

## Parameters

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_feats` | int | Input feature dimension |
| `out_feats` | int | Output feature dimension |
| `activation` | Optional[Callable] | Activation function to use (default: None) |
| `bias` | bool | Whether to use bias |
| `weight` | bool | Whether to apply a linear transformation |
| `allow_zero_in_degree` | bool | Whether to allow nodes with zero in-degree |

### Forward Method Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g` | dgl.DGLGraph | Input graph |
| `features` | torch.Tensor | Node feature matrix with shape (num_nodes, in_feats) |
| `do_hp` | bool | Whether to compute High-Pass (I - GCN) or just GCN |

## Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Transformed node features with shape (num_nodes, out_feats) |

## Usage Examples

### Basic Usage

```python
import torch
import dgl
from bridge.models import HPGraphConv

# Create a simple graph
g = dgl.graph(([0, 1], [1, 2]))
g.ndata['feat'] = torch.randn(3, 5)  # 3 nodes, 5 features each

# Create a high-pass graph convolution layer
hp_conv = HPGraphConv(
    in_feats=5,     # Input feature size
    out_feats=10,   # Output feature size
    bias=True       # Use bias
)

# Forward pass with high-pass filtering
output_hp = hp_conv(g, g.ndata['feat'], do_hp=True)
print(output_hp.shape)  # Should be (3, 10)

# Forward pass without high-pass filtering (standard GCN)
output_std = hp_conv(g, g.ndata['feat'], do_hp=False)
print(output_std.shape)  # Should be (3, 10)
```

### With Activation Function

```python
import torch.nn.functional as F
from bridge.models import HPGraphConv

# Create a high-pass layer with ReLU activation
hp_conv_relu = HPGraphConv(
    in_feats=5,
    out_feats=10,
    activation=F.relu
)

# Forward pass
output = hp_conv_relu(g, g.ndata['feat'])
```

### In a GNN Architecture

```python
class HighPassGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.conv1 = HPGraphConv(in_feats, h_feats, activation=F.relu)
        self.conv2 = HPGraphConv(h_feats, out_feats)
        
    def forward(self, g, features):
        h = self.conv1(g, features)
        h = self.conv2(g, h)
        return h

# Create the model
model = HighPassGNN(in_feats=5, h_feats=16, out_feats=2)

# Forward pass
output = model(g, g.ndata['feat'])
```

## Implementation Details

The `HPGraphConv` layer is built on top of DGL's `GraphConv` layer, but it modifies the standard graph convolution operation to implement a high-pass filter.

When `do_hp` is True (the default), the layer computes:
```
H = features - GraphConv(g, features)
```

This effectively highlights the differences between a node's features and the aggregated features from its neighbors, acting as a high-pass filter.

When `do_hp` is False, the layer behaves like a standard graph convolution:
```
H = GraphConv(g, features)
```

After the graph convolution operation, a linear transformation is applied to the results if `weight` is True, followed by an activation function if provided.

## Related Components

- [GCN](api-reference/gcn.html): Graph Convolutional Network that can use HPGraphConv layers
- [SelectiveGCN](api-reference/selectivegcn.html): Extension of GCN that can operate on multiple graph versions
