---
layout: default
title: train_selective
parent: API Reference
---

# train_selective
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `train_selective` function provides a specialized training pipeline for Selective Graph Neural Networks (SGNNs), which can operate on multiple graph versions. This function extends the standard training process to handle a list of graphs instead of a single graph.

## Function Signature

```python
def train_selective(
    g_list: List[dgl.DGLGraph],
    model_selective: nn.Module,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    model_lr: float = 1e-3,
    optimizer_weight_decay: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    log_training: bool = False,
    metric_type: str = 'accuracy'
) -> Tuple[float, float, float, nn.Module]
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `g_list` | List[dgl.DGLGraph] | List of input graphs with features and masks |
| `model_selective` | nn.Module | Selective neural network model to train |
| `train_mask` | torch.Tensor | Boolean mask indicating training nodes |
| `val_mask` | torch.Tensor | Boolean mask indicating validation nodes |
| `test_mask` | torch.Tensor | Boolean mask indicating test nodes |
| `model_lr` | float | Learning rate for the optimizer |
| `optimizer_weight_decay` | float | Weight decay for the optimizer |
| `n_epochs` | int | Maximum number of training epochs |
| `early_stopping` | int | Number of epochs to look back for early stopping |
| `log_training` | bool | Whether to print training progress |
| `metric_type` | str | Type of metric to compute, either 'accuracy' or 'roc_auc' |

## Returns

A tuple containing:

| Return Value | Type | Description |
|--------------|------|-------------|
| final_train_acc | float | Final training accuracy or ROC AUC |
| final_val_acc | float | Final validation accuracy or ROC AUC |
| final_test_acc | float | Final test accuracy or ROC AUC |
| model_selective | nn.Module | Trained selective model with the best validation performance |

## Early Stopping Strategy

Similar to the `train` function, `train_selective` implements an early stopping strategy based on validation loss. Training is halted when:
```
current_val_loss > mean(previous_early_stopping_val_losses)
```

This approach is robust to fluctuations and helps prevent stopping too early or too late, especially important when training on multiple graph variants.

## Graph Selection Mechanism

The `train_selective` function is designed to work with the `SelectiveGCN` model, which can choose between different graph versions for each node. Each graph in `g_list` should have a `mask` attribute in its node data that indicates which graph should be used for that node. The primary mechanism works as follows:

1. The original graph (typically `g_list[0]`) and rewired graph(s) (e.g., `g_list[1]`) are processed in parallel
2. For each node, the model selects the representation from the graph that provides the best local homophily
3. This adaptive selection allows the model to leverage the best graph structure for each node

## Usage Examples

### Basic Usage with Original and Rewired Graphs

```python
import torch
import dgl
from bridge.models import SelectiveGCN
from bridge.training import train_selective
from bridge.rewiring import create_rewired_graph

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g_orig = dataset[0]

# Create a rewired version of the graph
# (assumes you have already computed B_opt_tensor, pred, Z_pred)
g_rewired = create_rewired_graph(
    g=g_orig,
    B_opt_tensor=B_opt_tensor,
    pred=pred,
    Z_pred=Z_pred,
    p_remove=0.1,
    p_add=0.1
)

# Compute local homophily for each node in both graphs
# (simplified example - you would typically use local_homophily function)
for i, g_i in enumerate([g_orig, g_rewired]):
    # Calculate local homophily for each node
    # ...
    
# Determine which graph has better homophily for each node
node_mask = torch.argmax(torch.stack(lh_list), dim=0)

# Assign masks to graphs
g_orig.ndata['mask'] = node_mask
g_rewired.ndata['mask'] = node_mask

# Create a SelectiveGCN model
in_feats = g_orig.ndata['feat'].shape[1]
out_feats = len(torch.unique(g_orig.ndata['label']))
model = SelectiveGCN(
    in_feats=in_feats,
    h_feats=64,
    out_feats=out_feats,
    n_layers=2,
    dropout_p=0.5
)

# Train the selective model on both graphs
train_acc, val_acc, test_acc, trained_model = train_selective(
    g_list=[g_orig, g_rewired],
    model_selective=model,
    train_mask=g_orig.ndata['train_mask'],
    val_mask=g_orig.ndata['val_mask'],
    test_mask=g_orig.ndata['test_mask'],
    model_lr=1e-3,
    optimizer_weight_decay=5e-4,
    n_epochs=200,
    early_stopping=30,
    log_training=True
)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
```

### Using with Multiple Rewired Graphs

```python
# Create multiple rewired versions with different parameters
g_rewired_1 = create_rewired_graph(g_orig, B_opt_tensor, pred, Z_pred, 0.1, 0.1)
g_rewired_2 = create_rewired_graph(g_orig, B_opt_tensor, pred, Z_pred, 0.2, 0.2)
g_rewired_3 = create_rewired_graph(g_orig, B_opt_tensor, pred, Z_pred, 0.3, 0.3)

# Determine which graph has best homophily for each node
# ...

# Add mask to all graphs
for g_i in [g_orig, g_rewired_1, g_rewired_2, g_rewired_3]:
    g_i.ndata['mask'] = node_mask

# Train with multiple graph versions
train_acc, val_acc, test_acc, trained_model = train_selective(
    g_list=[g_orig, g_rewired_1, g_rewired_2, g_rewired_3],
    model_selective=model,
    train_mask=g_orig.n