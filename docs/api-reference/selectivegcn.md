---
layout: default
title: SelectiveGCN
parent: API Reference
---

# SelectiveGCN
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `SelectiveGCN` class extends the standard Graph Convolutional Network (GCN) to operate on multiple graph versions. It applies the same GCN architecture to different versions of the input graph, then selects the best output for each node based on local homophily.

## Class Definition

```python
class SelectiveGCN(nn.Module):
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
        
    def forward(self, g_list: List[dgl.DGLGraph], features: torch.Tensor) -> torch.Tensor:
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
| `g_list` | List[dgl.DGLGraph] | List of input graphs |
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
from bridge.models import SelectiveGCN

# Create two versions of a graph
g1 = dgl.graph(([0, 1], [1, 2]))
g2 = dgl.graph(([0, 1, 0], [1, 2, 2]))  # A modified version with an extra edge

# Add node features
g1.ndata['feat'] = torch.randn(3, 5)  # 3 nodes, 5 features each
g2.ndata['feat'] = g1.ndata['feat'].clone()  # Same features

# Add a mask to indicate which graph to use for each node (this is usually computed during rewiring)
# In this example, node 0 uses graph 0, node 1 uses graph 1, and node 2 uses graph 0
g1.ndata['mask'] = torch.tensor([0, 1, 0])
g2.ndata['mask'] = g1.ndata['mask'].clone()

# Create a SelectiveGCN model
model = SelectiveGCN(
    in_feats=5,     # Input feature size
    h_feats=16,     # Hidden layer size
    out_feats=2,    # Output size (e.g., number of classes)
    n_layers=2,     # Number of GCN layers
    dropout_p=0.5   # Dropout rate
)

# Forward pass
output = model([g1, g2], g1.ndata['feat'])
print(output.shape)  # Should be (3, 2)
```

### With High-Pass Filter

```python
# Create a SelectiveGCN model with high-pass filters
high_pass_model = SelectiveGCN(
    in_feats=5,
    h_feats=16,
    out_feats=2,
    n_layers=2,
    dropout_p=0.5,
    do_hp=True  # Use high-pass filters
)

# Forward pass
output = high_pass_model([g1, g2], g1.ndata['feat'])
```

### With the BRIDGE Pipeline

```python
from bridge.rewiring import run_bridge_pipeline
from bridge.utils import generate_all_symmetric_permutation_matrices

# Generate permutation matrices
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]

# Run the BRIDGE pipeline
results = run_bridge_pipeline(
    g=g,
    P_k=P_k,
    h_feats_gcn=64,
    n_layers_gcn=2,
    h_feats_selective=64,
    n_layers_selective=2,
    dropout_p_selective=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# The results include the trained SelectiveGCN model:
selective_model = results['selective']['model']
```

## Implementation Details

The `SelectiveGCN` class extends the standard GCN architecture to operate on multiple graph versions. It has the following key components:

1. **Multiple Graph Processing**: The forward method takes a list of graphs `g_list` and applies the same GCN layers to each graph.

2. **Node-wise Selection**: Each node in the graphs has a `mask` attribute that indicates which graph's output should be used for that node. The mask values are indices into the list of graphs.

3. **Architecture**: Similar to the standard GCN, the SelectiveGCN consists of multiple graph convolution layers (either standard or high-pass), with activation functions and dropout applied between layers.

The selective approach allows the model to leverage different graph structures for different nodes, which can be particularly useful when dealing with graphs that have both homophilic and heterophilic regions.

## Related Components

- [GCN](api-reference/gcn.md): Standard Graph Convolutional Network
- [HPGraphConv](api-reference/hpgraphconv.md): High-Pass Graph Convolution layer used when `do_hp=True`
- [run_bridge_pipeline](api-reference/bridge-pipeline.md): Pipeline that uses SelectiveGCN for improved node classification
