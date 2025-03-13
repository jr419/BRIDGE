---
layout: default
title: Sensitivity Analysis
parent: API Reference
---

# Sensitivity Analysis
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The sensitivity analysis module provides functions for computing the sensitivity of graph neural networks to various types of input perturbations, including signal, noise, and global variations. These sensitivity measures are key to understanding the Signal-to-Noise Ratio of GNNs.

## Sensitivity Functions

### estimate_sensitivity_analytic

```python
def estimate_sensitivity_analytic(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    labels: torch.Tensor, 
    sensitivity_type: Literal["signal", "noise", "global"]
) -> torch.Tensor
```

Estimates the sensitivity for a *Linear* GCN analytically (no autograd). This function computes the sensitivity matrix for a linear GCN model analytically, based on the model weights and graph structure, without requiring autograd.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | A linear GCN model with a weight attribute |
| `graph` | dgl.DGLGraph | The input graph |
| `labels` | torch.Tensor | Node labels (used for signal sensitivity) |
| `sensitivity_type` | Literal["signal", "noise", "global"] | Type of sensitivity to compute |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | A sensitivity tensor of shape [num_nodes, num_classes, in_feats, in_feats] |

#### Sensitivity Types

- **"signal"**: Sensitivity to coherent class-specific changes in the input features
- **"noise"**: Sensitivity to unstructured, IID noise
- **"global"**: Sensitivity to global shifts in the input features

#### Example

```python
import torch
import dgl
from bridge.sensitivity import estimate_sensitivity_analytic
from bridge.models import LinearGCN

# Create a linear GCN model
model = LinearGCN(in_feats=5, hidden_feats=16, out_feats=3)

# Load a graph
g = dgl.data.CoraGraphDataset()[0]
labels = g.ndata['label']

# Compute signal sensitivity
signal_sensitivity = estimate_sensitivity_analytic(
    model=model,
    graph=g,
    labels=labels,
    sensitivity_type="signal"
)

print(f"Signal sensitivity shape: {signal_sensitivity.shape}")
print(f"Average signal sensitivity: {signal_sensitivity.mean().item():.4f}")
```

### compute_jacobian

```python
def compute_jacobian(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    x: torch.Tensor, 
    device: str = "cuda"
) -> torch.Tensor
```

Computes the Jacobian matrix of the model with respect to the input. The Jacobian J has the form:
J[i, j, k, l] = d (model(graph, x)[i, j]) / d x[k, l]

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | The neural network model |
| `graph` | dgl.DGLGraph | The input graph |
| `x` | torch.Tensor | Input features of shape (N, in_feats) |
| `device` | str | Device to compute on ("cuda" or "cpu") |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | A tensor of shape (N, out_feats, N, in_feats) containing the Jacobian |

#### Example

```python
import torch
import dgl
from bridge.sensitivity import compute_jacobian
from bridge.models import TwoLayerGCN

# Create a non-linear GCN model
model = TwoLayerGCN(in_feats=5, hidden_feats=16, out_feats=3)

# Load a graph
g = dgl.data.CoraGraphDataset()[0]

# Create input features (zeros for simplicity)
x = torch.zeros(g.num_nodes(), 5, device="cuda" if torch.cuda.is_available() else "cpu")

# Compute Jacobian
jacobian = compute_jacobian(
    model=model,
    graph=g,
    x=x,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Jacobian shape: {jacobian.shape}")
```

### estimate_sensitivity_autograd

```python
def estimate_sensitivity_autograd(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    in_feats: int,
    labels: torch.Tensor, 
    sensitivity_type: Literal["signal", "noise", "global"], 
    device: str = "cuda"
) -> torch.Tensor
```

Estimates sensitivity using autograd-computed Jacobian. This function computes the sensitivity matrix for any differentiable model using PyTorch's automatic differentiation. It handles different sensitivity types based on the paper's definitions.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | The neural network model |
| `graph` | dgl.DGLGraph | The input graph |
| `in_feats` | int | Number of input features |
| `labels` | torch.Tensor | Node labels (used for signal sensitivity) |
| `sensitivity_type` | Literal["signal", "noise", "global"] | Type of sensitivity to compute |
| `device` | str | Device to compute on ("cuda" or "cpu") |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | A sensitivity tensor of shape [num_nodes, num_classes, in_feats, in_feats] |

#### Example

```python
import torch
import dgl
from bridge.sensitivity import estimate_sensitivity_autograd
from bridge.models import TwoLayerGCN

# Create a non-linear GCN model
model = TwoLayerGCN(in_feats=5, hidden_feats=16, out_feats=3)

# Load a graph
g = dgl.data.CoraGraphDataset()[0]
labels = g.ndata['label']

# Compute signal sensitivity using autograd
signal_sensitivity = estimate_sensitivity_autograd(
    model=model,
    graph=g,
    in_feats=5,
    labels=labels,
    sensitivity_type="signal",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Average signal sensitivity: {signal_sensitivity.mean().item():.4f}")
```

## Experiment Functions

### run_sensitivity_experiment

```python
def run_sensitivity_experiment(
    model: nn.Module,
    graph: dgl.DGLGraph,
    feature_generator: Callable,
    in_feats: int,
    num_acc_repeats: int = 100,
    num_monte_carlo_samples: int = 100,
    num_epochs: int = 200,
    lr: float = 0.01, 
    weight_decay: float = 1e-3,
    sigma_intra: Optional[torch.Tensor] = None,
    sigma_inter: Optional[torch.Tensor] = None,
    tau: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    device: str = "cuda",
    do_mean: bool = True
) -> Dict[str, Any]
```

Runs a comprehensive sensitivity analysis experiment. This function trains a model multiple times with different feature realizations, estimates SNR using Monte Carlo and theorem-based approaches, and computes accuracy and other metrics.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | The neural network model to evaluate |
| `graph` | dgl.DGLGraph | The input graph |
| `feature_generator` | Callable | Function to generate features |
| `in_feats` | int | Number of input features |
| `num_acc_repeats` | int | Number of training repetitions for accuracy estimation |
| `num_monte_carlo_samples` | int | Number of samples for Monte Carlo SNR estimation |
| `num_epochs` | int | Number of training epochs |
| `lr` | float | Learning rate |
| `weight_decay` | float | Weight decay for regularization |
| `sigma_intra` | Optional[torch.Tensor] | Intra-class covariance matrix |
| `sigma_inter` | Optional[torch.Tensor] | Inter-class covariance matrix |
| `tau` | Optional[torch.Tensor] | Global shift covariance matrix |
| `eta` | Optional[torch.Tensor] | Noise covariance matrix |
| `device` | str | Device to compute on |
| `do_mean` | bool | Whether to return node-averaged metrics (True) or node-level metrics (False) |

#### Returns

| Return Type | Description |
|-------------|-------------|
| Dict[str, Any] | Dictionary with experiment results |

#### Output Dictionary Keys

- `estimated_snr_mc`: Monte Carlo SNR estimate
- `estimated_snr_theorem`: Theorem-based SNR estimate
- `mean_test_acc`: Mean test accuracy
- `mean_test_loss`: Mean test loss
- `homophily`: Graph homophily

#### Example

```python
from bridge.sensitivity import run_sensitivity_experiment, create_feature_generator
from bridge.models import LinearGCN

# Create a model
model = LinearGCN(in_feats=5, hidden_feats=16, out_feats=3)

# Create a feature generator
feature_generator = create_feature_generator(
    sigma_intra=0.1 * torch.eye(5),
    sigma_inter=-0.05 * torch.eye(5),
    tau=torch.eye(5),
    eta=torch.eye(5)
)

# Run a sensitivity experiment
results = run_sensitivity_experiment(
    model=model,
    graph=g,
    feature_generator=feature_generator,
    in_feats=5,
    num_acc_repeats=10,
    num_monte_carlo_samples=50,
    sigma_intra=0.1 * torch.eye(5),
    sigma_inter=-0.05 * torch.eye(5),
    tau=torch.eye(5),
    eta=torch.eye(5),
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Monte Carlo SNR: {results['estimated_snr_mc'].item():.4f}")
print(f"Theorem-based SNR: {results['estimated_snr_theorem'].item():.4f}")
print(f"Test Accuracy: {results['mean_test_acc']:.4f}")
```

### run_multi_graph_experiment

```python
def run_multi_graph_experiment(
    graph_generator: Callable,
    model_constructor: Callable,
    feature_generator: Callable,
    in_feats: int,
    num_nodes: int,
    num_classes: int,
    homophily_values: List[float],
    mean_degree: int = 10,
    num_samples: int = 5,
    **experiment_kwargs
) -> Dict[str, List[Tuple[float, float]]]
```

Runs sensitivity analysis on multiple graphs with varying homophily. This function generates multiple graphs with different homophily values, runs sensitivity experiments on each, and collects the results.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph_generator` | Callable | Function to generate a graph given parameters |
| `model_constructor` | Callable | Function to construct a model given in_feats |
| `feature_generator` | Callable | Function to generate features |
| `in_feats` | int | Number of input features |
| `num_nodes` | int | Number of nodes in generated graphs |
| `num_classes` | int | Number of classes in generated graphs |
| `homophily_values` | List[float] | List of homophily values to test |
| `mean_degree` | int | Mean degree for generated graphs |
| `num_samples` | int | Number of graph samples per homophily value |
| `experiment_kwargs` | Any | Additional arguments for run_sensitivity_experiment |

#### Returns

| Return Type | Description |
|-------------|-------------|
| Dict[str, List[Tuple[float, float]]] | Dictionary with lists of (mean, std) tuples for each metric |

#### Output Dictionary Keys

- `estimated_snr_mc_list`: Monte Carlo SNR estimates
- `estimated_snr_theorem_val_list`: Theorem-based SNR estimates
- `acc_list`: Test accuracy values
- `loss_list`: Test loss values
- `homophily_list`: Actual homophily values

#### Example

```python
from bridge.sensitivity import run_multi_graph_experiment, create_feature_generator
from bridge.datasets import SyntheticGraphDataset
from bridge.models import LinearGCN

# Define a graph generator function
def graph_generator(num_nodes, num_classes, homophily, mean_degree):
    dataset = SyntheticGraphDataset(
        n=num_nodes,
        k=num_classes,
        h=homophily,
        d_mean=mean_degree,
        in_feats=5
    )
    return dataset[0]

# Define a model constructor function
def model_constructor(in_feats, num_classes):
    return LinearGCN(in_feats=in_feats, hidden_feats=16, out_feats=num_classes)

# Create a feature generator
feature_generator = create_feature_generator(
    sigma_intra=0.1 * torch.eye(5),
    sigma_inter=-0.05 * torch.eye(5),
    tau=torch.eye(5),
    eta=torch.eye(5)
)

# Run experiments across different homophily values
homophily_values = [0.1, 0.3, 0.5, 0.7, 0.9]
results = run_multi_graph_experiment(
    graph_generator=graph_generator,
    model_constructor=model_constructor,
    feature_generator=feature_generator,
    in_feats=5,
    num_nodes=500,
    num_classes=3,
    homophily_values=homophily_values,
    mean_degree=10,
    num_samples=3,
    num_acc_repeats=5,
    num_monte_carlo_samples=20
)

# Print results for each homophily value
for i, h in enumerate(homophily_values):
    snr_mc_mean, snr_mc_std = results['estimated_snr_mc_list'][i]
    acc_mean, acc_std = results['acc_list'][i]
    print(f"Homophily {h:.1f}: SNR = {snr_mc_mean:.4f}±{snr_mc_std:.4f}, Acc = {acc_mean:.4f}±{acc_std:.4f}")
```

## Helper Functions

### node_level_evaluate

```python
def node_level_evaluate(
    model: nn.Module,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor
```

Evaluates a GNN model on the given graph and returns node-level accuracy.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | The neural network model to evaluate |
| `graph` | dgl.DGLGraph | The input graph |
| `features` | torch.Tensor | Node features |
| `labels` | torch.Tensor | Node labels |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Binary tensor indicating correct (1) or incorrect (0) prediction for each node |

### get_sample_statistics

```python
def get_sample_statistics(
    values: List[float], 
    remove_nan: bool = True
) -> Tuple[float, float]
```

Computes mean and standard deviation from a list of values.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | List[float] | List of values |
| `remove_nan` | bool | Whether to remove NaN values before computing statistics |

#### Returns

| Return Type | Description |
|-------------|-------------|
| Tuple[float, float] | Tuple of (mean, std) |

## Mathematical Background

### Signal Sensitivity

For a node i, output feature p, and input features q and r, the signal sensitivity is defined as:

$$S^{(\ell)}_{i,p,q,r} := \sum_{j,k \in V} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{jq}} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{kr}} \delta_{y_j,y_k}$$

This measures how the model output changes in response to coordinated changes in input features from nodes of the same class.

### Noise Sensitivity

Similarly, the noise sensitivity is defined as:

$$N^{(\ell)}_{i,p,q,r} := \sum_{j \in V} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{jq}} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{jr}}$$

This measures how the model output changes in response to unstructured noise in input features.

### Global Sensitivity

The global sensitivity is defined as:

$$T^{(\ell)}_{i,p,q,r} := \sum_{j,k \in V} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{jq}} \frac{\partial H^{(\ell)}_{ip}}{\partial X_{kr}}$$

This measures how the model output changes in response to global shifts in input features.

## Relationship to Graph Neural Networks

Sensitivity analysis provides insights into:

1. **Information Flow**: How effectively information propagates through the graph structure
2. **Bottleneck Identification**: Nodes or regions with low signal sensitivity
3. **Architecture Selection**: Whether to use standard or high-pass filters based on graph properties
4. **Optimal Graph Structure**: What structures maximize signal-to-noise ratio

