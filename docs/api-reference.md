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

- [GCN](bridgeapi-reference/gcn.md): Standard Graph Convolutional Network implementation
- [HPGraphConv](bridgeapi-reference/hpgraphconv.md): High-Pass Graph Convolution layer
- [SelectiveGCN](bridgeapi-reference/selectivegcn.md): GCN that can selectively operate on different graph structures

### bridge.rewiring

The `rewiring` package contains functions for rewiring graph structures to optimize MPNN performance.

- [run_bridge_pipeline](bridgeapi-reference/bridge-pipeline.md): Main pipeline for running the BRIDGE rewiring process
- [run_bridge_experiment](bridgeapi-reference/bridge-experiment.md): Function for running multiple rewiring trials across data splits
- [create_rewired_graph](bridgeapi-reference/create-rewired-graph.md): Low-level function to create a rewired version of a graph

### bridge.training

The `training` package provides functions for training and evaluating graph neural networks.

- [train](bridgeapi-reference/train.md): Train a GNN with early stopping
- [train_selective](bridgeapi-reference/train-selective.md): Train a selective GNN on multiple graph versions

### bridge.utils

The `utils` package includes utility functions for working with graphs and matrices.

- [local_homophily](bridgeapi-reference/local-homophily.md): Compute local homophily for each node
- [matrix_utils](bridgeapi-reference/matrix-utils.md): Utilities for working with matrices
- [graph_utils](bridgeapi-reference/graph-utils.md): Utilities for working with graphs

### bridge.datasets

The `datasets` package provides functionality for creating and working with graph datasets.

- [SyntheticGraphDataset](bridgeapi-reference/synthetic-dataset.md): Generate synthetic graphs with controllable properties

### bridge.optimization

The `optimization` package provides objective functions for hyperparameter optimization with Optuna.

- [objective_gcn](bridgeapi-reference/objective-gcn.md): Objective function for optimizing base GCN hyperparameters
- [objective_rewiring](api-reference/objective-rewiring.md): Objective function for optimizing rewiring parameters

### bridge.sensitivity

The `sensitivity` package offers tools for sensitivity and SNR analysis of graph neural networks.

- [SNR Estimation](api-reference/snr.md): Estimate the SNR of GNN outputs
- [Sensitivity Analysis](api-reference/sensitivity-analysis.md): Compute sensitivity measures for GNNs
- [Feature Generation](api-reference/feature-generation.md): Generate features with controlled covariance structure
