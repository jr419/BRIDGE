BRIDGE: Block Rewiring from Inference-Derived Graph Ensembles
=============================================================

A novel graph rewiring technique that leverages Stochastic Block Models (SBMs) to create optimized graph structures for improved node classification.

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

The BRIDGE (Block Rewiring from Inference-Derived Graph Ensembles) library implements the methods and experiments described in:

    **The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing**
    *Jonathan Rubin, Sahil Loomba, Nick S. Jones*

Repository Structure
-------------------

This repository contains two main packages:

1. **BRIDGE Rewiring Package** - The core implementation of the BRIDGE technique for graph rewiring to optimize the performance of graph neural networks.

2. **Sensitivity Analysis Package** - Tools for analyzing the signal-to-noise ratio and sensitivity of graph neural networks, which were used to derive the theoretical results in the paper.

Key Concepts from the Paper
--------------------------

- **Signal-to-Noise Ratio (SNR) Framework**: A novel approach to quantify MPNN performance through signal, noise, and global sensitivity metrics
- **Higher-Order Homophily**: Measures of multi-hop connectivity between same-class nodes that bound MPNN sensitivity
- **Homophilic Bottlenecks**: Network structures that restrict information flow between nodes of the same class
- **Optimal Graph Structures**: Characterization of graph structures that maximize performance for given class assignments
- **Graph Rewiring**: Techniques to modify graph topology to increase higher-order homophily

Features
--------

**Graph Rewiring**
   
- SBM-based graph rewiring to optimize network structure
- Support for both homophilic and heterophilic settings
- Selective GNN models that choose the best graph structure for each node

**GNN Models**
   
- Graph Convolutional Networks (GCN) with various configurations
- High/Low-Pass graph convolution filter models
- Selective GNN models that can choose the best graph structure for each node

**Sensitivity Analysis**
   
- Signal, noise, and global sensitivity estimation
- SNR calculation using both Monte Carlo and analytical methods
- Node-level analysis of homophilic bottlenecks

**Optimization & Experiments**
   
- Hyperparameter optimization with Optuna
- Support for standard graph datasets and synthetic graph generation
- Comprehensive evaluation metrics and visualization tools

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   getting-started
   theory
   rewiring/index
   sensitivity/index
   
.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api-reference/models
   api-reference/rewiring
   api-reference/training
   api-reference/utils
   api-reference/datasets
   api-reference/optimization
   api-reference/sensitivity
   
.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   cli-reference
   license

Citation
--------

If you use this library in your research, please cite:

.. code-block:: text

   @article{rubin2025limits,
     author = {Jonathan Rubin, Sahil Loomba, Nick S. Jones},
     title = {The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing},
     year = {2025},
     journal = {}, 
     url = {https://github.com/jr419/BRIDGE}
   }

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
