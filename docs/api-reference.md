---
layout: default
title: API Reference
nav_order: 7
has_children: true
permalink: /api-reference
---

# API Reference
{: .no_toc }

The BRIDGE library consists of several core modules and subpackages that provide functionality for graph rewiring, sensitivity analysis, and experiment utilities.

## Main Components

- **models**: GCN and SelectiveGCN implementations 
- **rewiring**: Graph rewiring operations and pipelines
- **training**: Training loops for graph neural networks
- **utils**: Utility functions for graphs and matrices
- **datasets**: Synthetic graph dataset generation
- **optimization**: Hyperparameter optimization for GNNs
- **sensitivity**: Sensitivity and SNR analysis tools

## Core Modules

### bridge.models

The `models` package provides implementations of Graph Convolutional Networks (GCNs) and their variants.

- [GCN](api-reference/models/gcn.html): Standard Graph Convolutional Network implementation
- [HPGraphConv](api-reference/models/hpgraphconv.html): High-Pass Graph Convolution layer
- [SelectiveGCN](api-reference/models/selectivegcn.html): GCN that can selectively operate on different graph structures

### bridge.rewiring

The `rewiring` package contains functions for rewiring graph structures to optimize MPNN performance.

- [run_bridge_pipeline](api-reference/rewiring/bridge-pipeline.html): Main pipeline for running the BRIDGE rewiring process
- [run_bridge_experiment](api-reference/rewiring/bridge-experiment.html): Function for running multiple rewiring trials across data splits
- [create_rewired_graph](api-reference/rewiring/create-rewired-graph.html): Low-level function to create a rewired version of a graph

### bridge.training

The `training` package provides functions for training and evaluating graph neural networks.

- [train](api-reference/training/train.html): Train a GNN with early stopping
- [train_selective](api-reference/training/train-selective.html): Train a selective GNN on multiple graph versions

### bridge.utils

The `utils` package includes utility functions for working with graphs and matrices.

- [local_homophily](api-reference/utils/local-homophily.html): Compute local homophily for each node
- [matrix_utils](api-reference/utils/matrix-utils.html): Utilities for working with matrices
- [graph_utils](api-reference/utils/graph-utils.html): Utilities for working with graphs

### bridge.datasets

The `datasets` package provides functionality for creating and working with graph datasets.

- [SyntheticGraphDataset](api-reference/datasets/synthetic-dataset.html): Generate synthetic graphs with controllable properties

### bridge.optimization

The `optimization` package provides objective functions for hyperparameter optimization with Optuna.

- [objective_gcn](api-reference/optimization/objective-gcn.html): Objective function for optimizing base GCN hyperparameters
- [objective_rewiring](api-reference/optimization/objective-rewiring.html): Objective function for optimizing rewiring parameters

### bridge.sensitivity

The `sensitivity` package offers tools for sensitivity and SNR analysis of graph neural networks.

- [SNR Estimation](api-reference/sensitivity/snr.html): Estimate the SNR of GNN outputs
- [Sensitivity Analysis](api-reference/sensitivity/sensitivity-analysis.html): Compute sensitivity measures for GNNs
- [Feature Generation](api-reference/sensitivity/feature-generation.html): Generate features with controlled covariance structure
