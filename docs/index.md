---
layout: default
title: Home
nav_order: 1
permalink: /
---

# BRIDGE: Block Rewiring from Inference-Derived Graph Ensembles
{: .fs-9 }

Graph rewiring utilities and sensitivity analysis tools for modern graph neural networks.
{: .fs-6 .fw-300 }

[Get Started](getting-started.md){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/jr419/BRIDGE){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Overview

BRIDGE (Block Rewiring from Inference-Derived Graph Ensembles) is a technique for optimizing graph neural networks through graph rewiring. The repository implements the methods and experiments described in:

> **The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing**  
> *Jonathan Rubin, Sahil Loomba, Nick S. Jones*

## Repository Structure

This repository contains two main packages:

1. **BRIDGE Rewiring Package** - The core implementation of the BRIDGE technique for graph rewiring to optimize the performance of graph neural networks.

2. **Sensitivity Analysis Package** - Tools for analyzing the signal-to-noise ratio and sensitivity of graph neural networks, which were used to derive the theoretical results in the paper.

## Key Concepts from the Paper

- **Signal-to-Noise Ratio (SNR) Framework**: A novel approach to quantify MPNN performance through signal, noise, and global sensitivity metrics
- **Higher-Order Homophily**: Measures of multi-hop connectivity between same-class nodes that bound MPNN sensitivity
- **Homophilic Bottlenecks**: Network structures that restrict information flow between nodes of the same class
- **Optimal Graph Structures**: Characterization of graph structures that maximize performance for given class assignments
- **Graph Rewiring**: Techniques to modify graph topology to increase higher-order homophily

## Features

- **Graph Rewiring**
  - SBM-based graph rewiring to optimize network structure
  - Iterative rewiring with SGC-based predictions
  - Support for both homophilic and heterophilic settings
  - Selective GNN models that choose the best graph structure for each node

- **GNN Models**
  - Graph Convolutional Networks (GCN) with various configurations
  - High/Low-Pass graph convolution filter models
  - Selective GNN models that can choose the best graph structure for each node

- **Sensitivity Analysis**
  - Signal, noise, and global sensitivity estimation
  - SNR calculation via Monte Carlo or theorem-based formulas
  - Node-level analysis of homophilic bottlenecks

- **Optimization & Experiments**
  - Hyperparameter optimization with Optuna
  - Support for standard graph datasets and synthetic graph generation
  - Comprehensive evaluation metrics and visualization tools

## Citation

If you use this library in your research, please cite:

```bibtex
@article{rubin2025limits,
  author = {Jonathan Rubin, Sahil Loomba, Nick S. Jones},
  title = {The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing},
  year = {2025},
  journal = {}, 
  url = {https://github.com/jr419/BRIDGE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/jr419/BRIDGE/blob/main/LICENSE) file for details.
