---
layout: default
title: train
parent: API Reference
---

# train
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `train` function provides a comprehensive training pipeline for Graph Neural Networks (GNNs) with early stopping. This function handles the entire training process, including optimization, evaluation, and model checkpoint saving.

## Function Signature

```python
def train(
    g: dgl.DGLGraph,
    model: nn.Module,
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
| `g` | dgl.DGLGraph | Input graph with node features in `g.ndata['feat']` and labels in `g.ndata['label']` |
| `model` | nn.Module | Neural network model to train |
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
| final_train_metric | float | Final training metric (accuracy or ROC AUC) |
| final_val_metric | float | Final validation metric (accuracy or ROC AUC) |
| final_test_metric | float | Final test metric (accuracy or ROC AUC) |
| model | nn.Module | Trained model with the best validation performance |

## Early Stopping Strategy

The `train` function implements a sophisticated early stopping strategy that goes beyond simple validation metric tracking. Instead of stopping when the validation metric doesn't improve for a certain number of epochs, it compares the current validation loss with the average of the previous `early_stopping` validation losses.

Specifically, training is halted when:
```
current_val_loss > mean(previous_early_stopping_val_losses)
```

This approach is more robust to fluctuations and helps prevent stopping too early or too late.

## Usage Examples

### Basic Usage

```python
import torch
import dgl
from bridge.models import GCN
from bridge.training import train

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
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

### Using ROC AUC Metric

```python
# Train with ROC AUC as the evaluation metric
train_auc, val_auc, test_auc, trained_model = train(
    g=g,
    model=model,
    train_mask=g.ndata['train_mask'],
    val_mask=g.ndata['val_mask'],
    test_mask=g.ndata['test_mask'],
    model_lr=1e-3,
    optimizer_weight_decay=5e-4,
    n_epochs=200,
    early_stopping=30,
    metric_type='roc_auc'  # Use ROC AUC instead of accuracy
)

print(f"Train ROC AUC: {train_auc:.4f}")
print(f"Validation ROC AUC: {val_auc:.4f}")
print(f"Test ROC AUC: {test_auc:.4f}")
```

### Custom Training Loop

```python
# Disable early stopping by setting it to 0
train_acc, val_acc, test_acc, trained_model = train(
    g=g,
    model=model,
    train_mask=g.ndata['train_mask'],
    val_mask=g.ndata['val_mask'],
    test_mask=g.ndata['test_mask'],
    model_lr=1e-3,
    optimizer_weight_decay=5e-4,
    n_epochs=100,  # Fixed number of epochs
    early_stopping=0,  # Disable early stopping
    log_training=True
)
```

## Implementation Details

The `train` function implements the following training pipeline:

1. **Initialization**:
   - Sets up an Adam optimizer with the specified learning rate and weight decay
   - Initializes a cross-entropy loss function
   - Creates tracking variables for the best model and metrics

2. **Training Loop**:
   - For each epoch:
     - Trains the model for one epoch (calling `train_one_epoch`)
     - Evaluates on the validation set
     - Tracks the validation loss and metric for early stopping
     - Updates the best model if the validation metric improves
     - Checks the early stopping criterion

3. **Early Stopping Check**:
   - After accumulating at least (early_stopping+1) epochs:
   - Calculates the mean of the previous `early_stopping` validation losses
   - Compares the current validation loss to this mean
   - Stops training if the current loss exceeds the mean

4. **Final Evaluation**:
   - Loads the best model state
   - Evaluates on the train, validation, and test sets

The function utilizes a unique early stopping approach that is more robust to temporary fluctuations in validation loss, making it particularly suitable for graph learning where such fluctuations are common.
