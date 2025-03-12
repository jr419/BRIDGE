# BRIDGE Sensitivity Analysis Package

This package provides tools for analyzing the signal-to-noise ratio (SNR) and sensitivity of graph neural networks. It offers methods to understand when and how graph structure affects model performance, based on the paper:

> "The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing"

## Overview

The BRIDGE sensitivity analysis framework helps you:

1. **Understand when MPNNs help**: Determine conditions where graph structure provides an advantage over non-relational models
2. **Identify bottlenecks**: Locate problematic structural patterns that limit model performance
3. **Quantify sensitivity**: Measure how different model architectures respond to various input perturbations
4. **Optimize graph structure**: Guide graph rewiring for improved performance

## Key Components

The package includes:

- **Models**: Linear and nonlinear GNN implementations
- **SNR estimation**: Methods to estimate signal-to-noise ratio through Monte Carlo simulation and theoretical approaches
- **Sensitivity analysis**: Tools to analyze model sensitivity to different input perturbations
- **Feature generation**: Utilities for generating synthetic node features with controlled covariance structure
- **Visualization**: Functions for visualizing sensitivity analysis results
- **Experiment utilities**: Helpers for running comprehensive experiments

## Theoretical Framework

Our analysis is based on a novel signal-to-noise ratio framework that relates model performance to three key sensitivity measures:

1. **Signal sensitivity**: Measures the model's response to coherent class-specific changes in input features
2. **Noise sensitivity**: Measures the model's response to unstructured, IID noise
3. **Global sensitivity**: Measures the model's response to global shifts in input features

For a linear GCN, these sensitivities are directly related to the higher-order homophily of the graph, which quantifies the multi-hop connectivity between same-class nodes.

## Usage

### Basic Example

```python
import torch
import dgl
from bridge.sensitivity import (
    LinearGCN, 
    create_feature_generator,
    run_sensitivity_experiment
)

# Create graph (e.g., from a standard dataset)
graph = dgl.data.CoraGraphDataset()[0]
labels = graph.ndata['label']

# Define feature covariance matrices
in_feats = 5
scale = 1e-4
sigma_intra = torch.eye(in_feats) * 0.1 * scale
sigma_inter = -torch.eye(in_feats) * 0.05 * scale
tau = torch.eye(in_feats) * scale
eta = torch.eye(in_feats) * scale

# Create feature generator
feature_generator = create_feature_generator(sigma_intra, sigma_inter, tau, eta)

# Create model
model = LinearGCN(in_feats, 32, len(torch.unique(labels)))

# Run experiment
results = run_sensitivity_experiment(
    model=model,
    graph=graph,
    feature_generator=feature_generator,
    in_feats=in_feats,
    num_acc_repeats=100,
    num_monte_carlo_samples=100
)

print(f"Monte Carlo SNR: {results['estimated_snr_mc'].item():.4f}")
print(f"Test Accuracy: {results['mean_test_acc']:.4f}")
```

### Analyzing Multiple Graphs

```python
from bridge.sensitivity import run_multi_graph_experiment, plot_snr_vs_homophily
import numpy as np

# Run analysis over multiple homophily values
homophily_values = np.linspace(0.1, 0.9, 9)

multi_results = run_multi_graph_experiment(
    graph_generator=your_graph_generator_function,
    model_constructor=lambda in_feats, num_classes: LinearGCN(in_feats, 32, num_classes),
    feature_generator=feature_generator,
    in_feats=in_feats,
    num_nodes=1000,
    num_classes=7,
    homophily_values=homophily_values,
    mean_degree=10,
    num_samples=5
)

# Visualize results
plot_snr_vs_homophily(
    homophily_values=multi_results["homophily_list"],
    snr_mc_means=multi_results["estimated_snr_mc_list"],
    snr_mc_stds=multi_results["estimated_snr_mc_stds"],
    accuracy_means=multi_results["acc_list"],
    accuracy_stds=multi_results["acc_stds"],
    title="SNR and Test Accuracy vs Edge Homophily",
    save_path="snr_vs_homophily.png"
)
```

## Further Reading

For a detailed explanation of the theoretical framework and experimental results, please refer to the paper:

> "The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing"

## Citation

If you use this package in your research, please cite:

```
@article{rubin2025limits,
  title={The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing},
  author={Rubin, Jonathan and Loomba, Sahil and Jones, Nick S.},
  journal={arXiv preprint},
  year={2025}
}
```
