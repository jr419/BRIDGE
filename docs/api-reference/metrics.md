---
layout: default
title: evaluate_metrics
parent: API Reference
---

# evaluate_metrics
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `evaluate_metrics` function computes classification metrics for graph neural networks. It supports both accuracy and ROC AUC evaluation metrics, making it suitable for balanced and imbalanced classification tasks.

## Function Signature

```python
def evaluate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    metric_type: str = 'accuracy'
) -> float
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | torch.Tensor | Model output logits with shape (n_nodes, n_classes) |
| `labels` | torch.Tensor | True labels with shape (n_nodes,) |
| `mask` | torch.Tensor | Boolean mask indicating which nodes to evaluate |
| `metric_type` | str | Type of metric to compute, either 'accuracy' or 'roc_auc' |

## Returns

| Return Type | Description |
|-------------|-------------|
| float | The computed metric value (accuracy or ROC AUC) |

## Metric Types

### Accuracy

When `metric_type` is set to `'accuracy'`, the function computes the standard classification accuracy:

```
accuracy = number of correctly classified nodes / total number of evaluated nodes
```

### ROC AUC

When `metric_type` is set to `'roc_auc'`, the function computes the Receiver Operating Characteristic Area Under Curve (ROC AUC):

- For binary classification, it computes the standard ROC AUC
- For multi-class classification, it uses the one-vs-rest approach to compute ROC AUC

## Usage Examples

### Accuracy Metric

```python
import torch
import dgl
from bridge.training import evaluate_metrics
from bridge.models import GCN

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Create and initialize a model
in_feats = g.ndata['feat'].shape[1]
out_feats = len(torch.unique(g.ndata['label']))
model = GCN(in_feats, 64, out_feats, 2, 0.5)

# Perform forward pass to get logits
model.eval()
with torch.no_grad():
    logits = model(g, g.ndata['feat'])

# Evaluate accuracy on test set
test_mask = g.ndata['test_mask']
labels = g.ndata['label']
accuracy = evaluate_metrics(logits, labels, test_mask, metric_type='accuracy')
print(f"Test accuracy: {accuracy:.4f}")
```

### ROC AUC Metric

```python
# Evaluate ROC AUC on test set (useful for imbalanced classification)
roc_auc = evaluate_metrics(logits, labels, test_mask, metric_type='roc_auc')
print(f"Test ROC AUC: {roc_auc:.4f}")

# For multi-class problems, it uses one-vs-rest approach
```

### Custom Evaluation Loop

```python
# Custom evaluation function for different data splits
def evaluate_model(model, graph, splits=['train', 'val', 'test']):
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata['feat'])
        labels = graph.ndata['label']
        results = {}
        
        for split in splits:
            mask = graph.ndata[f'{split}_mask']
            accuracy = evaluate_metrics(logits, labels, mask, 'accuracy')
            roc_auc = evaluate_metrics(logits, labels, mask, 'roc_auc')
            results[split] = {'accuracy': accuracy, 'roc_auc': roc_auc}
            
    return results
```

## Implementation Details

The implementation depends on the `metric_type` parameter:

1. **Accuracy Metric** (`metric_type='accuracy'`):
   - Computes the predicted class by taking the argmax of the logits along dimension 1
   - Counts how many predictions match the true labels
   - Divides by the total number of masked nodes

2. **ROC AUC Metric** (`metric_type='roc_auc'`):
   - Converts logits to probabilities using softmax
   - Converts labels to one-hot encoding
   - Uses scikit-learn's `roc_auc_score` function 
   - For multi-class, applies the one-vs-rest approach

The function is designed to handle both binary and multi-class classification problems, making it versatile for different graph learning tasks.

## Helper Function: get_metric_type

Along with `evaluate_metrics`, the module provides a helper function `get_metric_type` that determines the appropriate metric type based on the dataset name:

```python
def get_metric_type(dataset_name: str) -> str:
    """
    Determine which metric to use based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        str: Either 'accuracy' or 'roc_auc'
    """
    accuracy_datasets = {'roman-empire', 'amazon-ratings'}
    roc_auc_datasets = {'minesweeper', 'tolokers', 'questions'}
    
    dataset_name = dataset_name.lower()
    if dataset_name in accuracy_datasets:
        return 'accuracy'
    elif dataset_name in roc_auc_datasets:
        return 'roc_auc'
    else:
        # Default to accuracy for other datasets
        return 'accuracy'
```

This function makes it easy to automatically choose the appropriate metric for known datasets.

## Related Components

- [train]({% link api-reference/train.md %}): Complete training function that uses evaluate_metrics
- [train_one_epoch]({% link api-reference/train-one-epoch.md %}): Training function that uses evaluate_metrics
- [train_selective]({% link api-reference/train-selective.md %}): Training function for SelectiveGCN models that uses evaluate_metrics
