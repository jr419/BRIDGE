# BRIDGE: Block Rewiring from Inference-Derived Graph Ensembles

BRIDGE (Block Rewiring from Inference-Derived Graph Ensembles) is a graph rewiring framework that leverages Stochastic Block Models (SBMs) to create optimised graph structures for improved node classification. The repository implements the methods and experiments described in:

> **The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing**  
> *Jonathan Rubin, Sahil Loomba, Nick S. Jones*

**[📚 Documentation](https://jr419.github.io/BRIDGE/)**

## Repository Structure

This repository contains two main packages:

1. **BRIDGE Rewiring Package** - The core implementation of the BRIDGE technique for graph rewiring to optimize the performance of graph neural networks.

2. **Sensitivity Analysis Package** - Tools for analyzing the signal-to-noise ratio and sensitivity of graph neural networks, which were used to derive the theoretical results in the paper.

## Key Concepts from the Paper

- **Signal-to-Noise Ratio (SNR) Framework**: A novel approach to quantify MPNN performance through signal, noise, and global sensitivity metrics
- **Higher-Order Homophily**: Measures of multi-hop connectivity between same-class nodes that bound MPNN sensitivity
- **Class-Bottlenecks**: Network structures that restrict information flow between nodes of the same class
- **Optimal Graph Structures**: Characterization of graph structures that maximize performance for given class assignments
- **Graph Rewiring**: Techniques to modify graph topology to increase higher-order homophily

## Features

- **Graph Rewiring**
  - SBM-based graph rewiring to optimize network structure
  - Iterative rewiring for improved topology
  - Support for both homophilic and heterophilic settings
  - Standard training on original and rewired graphs

- **GNN Models**
  - Graph Convolutional Networks (GCN) with various configurations

- **Sensitivity Analysis**
  - Signal, noise, and global sensitivity estimation
  - SNR calculation using Monte Carlo or theorem-based formulas
- Node-level analysis of class-bottlenecks

- **Optimization & Experiments**
  - Hyperparameter optimization with Optuna
  - Support for standard graph datasets and synthetic graph generation
  - Comprehensive evaluation metrics and visualization tools

## Installation

```bash
git clone https://github.com/jr419/BRIDGE.git
cd bridge
pip install -e .
```

## Quick Start

```python
import dgl
import torch
from bridge.models import GCN
from bridge.rewiring import run_bridge_pipeline
from bridge.utils import generate_all_symmetric_permutation_matrices

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Generate permutation matrices
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]  # Choose the first permutation matrix

# Run the rewiring pipeline
results = run_bridge_pipeline(
    g=g,
    P_k=P_k,
    h_feats_mpnn=64,
    n_layers_mpnn=2,
    dropout_p_mpnn=0.5,
    model_lr_mpnn=1e-3,
    num_graphs=1,
    n_rewire=5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Print results
print(f"Base GCN accuracy: {results['cold_start']['test_acc']:.4f}")
print(f"Rewired GCN accuracy: {results['rewired']['test_acc']:.4f}")
```

## Sensitivity Analysis Examples

To analyze the Signal-to-Noise Ratio (SNR) and sensitivity of a graph neural network:

```python
import torch
import dgl
from bridge.sensitivity import (
    estimate_snr_theorem,
    estimate_sensitivity_autograd,
    run_sensitivity_experiment,
    plot_snr_vs_homophily
)

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Configure sensitivity analysis
feature_params = {
    'intra_class_cov': 0.1,
    'inter_class_cov': -0.05,
    'global_cov': 1.0,
    'noise_cov': 1.0,
    'feature_dim': 5
}

# Run experiment across multiple graphs with varying homophily
results = run_sensitivity_experiment(
    g,
    homophily_values=[0.1, 0.3, 0.5, 0.7, 0.9],
    feature_params=feature_params
)

# Visualize results
plot_snr_vs_homophily(results)
```

## Running Experiments

You can run the full optimization pipeline using the command-line interface:

```bash
python -m bridge.main --dataset_type standard --standard_datasets cora --num_trials 100 --experiment_name cora_experiment
```

For synthetic datasets:

```bash
python -m bridge.main --dataset_type synthetic --syn_homophily 0.3 --syn_nodes 3000 --syn_classes 4 --experiment_name synthetic_experiment
```

To use the iterative rewiring pipeline add `--use_iterative_rewiring`:

```bash
python -m bridge.main --dataset_type standard --standard_datasets cora \
    --use_iterative_rewiring --num_trials 50 --experiment_name cora_iterative
```

## Configuration

BRIDGE provides multiple configuration methods to customize your experiments. Get a full list of available options:

```bash
bridge --help
```

The repository includes example configuration files in YAML or JSON format:

```bash
bridge --config config_examples/real_datasets_test.yaml
```

## Documentation

### Main Components

- **Models**: Standard GCN implementation
- **Rewiring**: Functions for rewiring graph structures based on stochastic block models
- **Training**: Training loops and evaluation metrics for GNNs
- **Optimization**: Hyperparameter optimization with Optuna
- **Utils**: Utility functions for working with graphs and matrices
- **Datasets**: Synthetic graph dataset generator with controllable properties
- **Sensitivity**: Tools for sensitivity and SNR analysis of GNNs

### Key Parameters

- `do_self_loop`: Add self-loops to graph nodes
- `do_residual_connections`: Use residual connections in GCN layers
- `p_add`: Probability of adding new edges during rewiring
- `p_remove`: Probability of removing existing edges during rewiring
- `temperature`: Temperature for softmax when computing class probabilities

## Results and Outputs

All experiment results are saved in the `results/` directory with the following structure:

```
results/experiment_name_timestamp/
├── config.json                    # Saved configuration for reproducibility
├── gcn_study.db                   # Optuna study database
├── summary.csv                    # Summary of results across datasets
├── all_results.json               # Detailed results for all datasets
├── dataset1_results.json          # Results for dataset1
└── dataset2_results.json          # Results for dataset2
```

## Citation

If you use this library in your research, please cite:

```
@article{rubin2025limits,
  author = {Jonathan Rubin, Sahil Loomba, Nick S. Jones},
  title = {The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing},
  year = {2025},
  journal = {}, 
  url = {https://github.com/jr419/BRIDGE}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
