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

- [GCN]({% link api-reference/gcn.md %}): Standard Graph Convolutional Network implementation
- [HPGraphConv]({% link api-reference/hpgraphconv.md %}): High-Pass Graph Convolution layer
- [SelectiveGCN]({% link api-reference/selectivegcn.md %}): GCN that can selectively operate on different graph structures

### bridge.rewiring

The `rewiring` package contains functions for rewiring graph structures to optimize MPNN performance.

- [run_bridge_pipeline]({% link api-reference/bridge-pipeline.md %}): Main pipeline for running the BRIDGE rewiring process
- [run_bridge_experiment]({% link api-reference/bridge-experiment.md %}): Function for running multiple rewiring trials across data splits
- [create_rewired_graph]({% link api-reference/create-rewired-graph.md %}): Low-level function to create a rewired version of a graph

### bridge.training

The `training` package provides functions for training and evaluating graph neural networks.

- [train]({% link api-reference/train.md %}): Train a GNN with early stopping
- [train_selective]({% link api-reference/train-selective.md %}): Train a selective GNN on multiple graph versions

