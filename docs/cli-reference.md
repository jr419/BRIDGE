---
layout: default
title: CLI Reference
nav_order: 6
---

# Command-Line Interface Reference
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

BRIDGE provides a command-line interface (CLI) for running experiments. The main entry point is the `bridge.main` module, which can be run as follows:

```bash
python -m bridge.main [options]
```

The CLI supports two primary experiment types:
1. **Rewiring**: Graph rewiring optimization experiments
2. **Sensitivity**: Sensitivity and SNR analysis experiments

## Basic Usage

### Rewiring Experiments

```bash
python -m bridge.main --experiment_type rewiring --dataset_type standard --standard_datasets cora citeseer --num_trials 100 --experiment_name my_experiment
```

### Sensitivity Analysis

```bash
python -m bridge.main --experiment_type sensitivity --config config_examples/snr_analysis.yaml
```

## Configuration Options

BRIDGE CLI offers many configuration options. You can view all available options by running:

```bash
python -m bridge.main --help
```

### Using Configuration Files

Instead of specifying all options on the command line, you can use a configuration file:

```bash
python -m bridge.main --config config_examples/real_datasets_test.yaml
```

Configuration files can be in YAML or JSON format. Command-line arguments take precedence over options in the configuration file.

## Experiment Type

```bash
--experiment_type {rewiring,sensitivity}
```

Specifies the type of experiment to run:
- `rewiring`: Run graph rewiring optimization
- `sensitivity`: Run sensitivity and SNR analysis

## General Settings

```bash
--seed INTEGER                  Random seed for reproducibility
--device TEXT                   Device to use (cuda or cpu)
--num_trials INTEGER            Number of optimization trials
--experiment_name TEXT          Name of the experiment
```

## Model Settings

```bash
--do_hp                         Use higher-order polynomial filters
--do_self_loop                  Add self-loops to graphs
--do_residual                   Use residual connections in GCN
--early_stopping INTEGER        Early stopping patience
```

## Dataset Settings

```bash
--dataset_type {standard,synthetic}  Type of dataset to use
--standard_datasets TEXT...     List of standard datasets to use
```

Supported standard datasets include:
- `cora`
- `citeseer`
- `pubmed`
- `actor`
- `chameleon`
- `squirrel`
- `wisconsin`
- `cornell`
- `texas`
- `minesweeper`
- `tolokers`

## Synthetic Dataset Parameters

```bash
--syn_nodes INTEGER             Number of nodes for synthetic dataset
--syn_classes INTEGER           Number of classes for synthetic dataset
--syn_homophily FLOAT           Homophily for synthetic dataset
--syn_degree FLOAT              Mean degree for synthetic dataset
--syn_features INTEGER          Number of features for synthetic dataset
```

## Optimization Parameters

### GCN Hyperparameters

```bash
--gcn_h_feats INTEGER...        Hidden feature dimensions to try for GCN
--gcn_n_layers INTEGER...       Number of layers to try for GCN
--gcn_dropout_range FLOAT FLOAT Dropout range for GCN [min, max]
```

### Rewiring Hyperparameters

```bash
--temperature_range FLOAT FLOAT Temperature range for softmax [min, max]
--p_add_range FLOAT FLOAT       Probability range for adding edges [min, max]
--p_remove_range FLOAT FLOAT    Probability range for removing edges [min, max]
```

### Selective GCN Hyperparameters

```bash
--h_feats_selective_options INTEGER...  Hidden feature dimensions to try for selective GCN
--n_layers_selective_options INTEGER... Number of layers to try for selective GCN
--dropout_selective_range FLOAT FLOAT   Dropout range for selective GCN [min, max]
--lr_selective_range FLOAT FLOAT        Learning rate range for selective GCN [min, max]
--wd_selective_range FLOAT FLOAT        Weight decay range for selective GCN [min, max]
```

## Symmetry Checking

```bash
--check_symmetry                Check and enforce graph symmetry
```

## Configuration File Examples

### Rewiring Experiment Configuration

Example YAML configuration for a rewiring experiment (`real_datasets_test.yaml`):

```yaml
# BRIDGE configuration for real datasets test experiment

# General settings
seed: 42
device: cuda
num_trials: 3
experiment_name: real_datasets_test

# Model settings
do_hp: false
do_self_loop: true
do_residual: false
early_stopping: 50

# Dataset settings
dataset_type: standard
standard_datasets:
  - cora
  - citeseer
  - actor
  - chameleon
  - squirrel
  - wisconsin
  - cornell
  - texas

# Optimization parameters
gcn_h_feats: [16, 32, 64, 128]
gcn_n_layers: [1, 2, 3]
gcn_dropout_range: [0.0, 0.7]
temperature_range: [1.0e-5, 2.0]
p_add_range: [0.0, 1.0]
p_remove_range: [0.0, 1.0]
h_feats_selective_options: [16, 32, 64, 128]
n_layers_selective_options: [1, 2, 3]
dropout_selective_range: [0.0, 0.7]
lr_selective_range: [1.0e-4, 1.0e-1]
wd_selective_range: [1.0e-6, 1.0e-3]

# Symmetry checking
check_symmetry: false
```

### Sensitivity Analysis Configuration

Example YAML configuration for a sensitivity analysis experiment (`snr_analysis.yaml`):

```yaml
# BRIDGE sensitivity analysis configuration for homophily sweep experiment

# Experiment metadata
experiment_name: homophily_sweep_experiment
experiment_type: sensitivity

# Graph parameters
num_nodes: 500
num_classes: 2
mean_degree: 20
homophily_min: 0.1
homophily_max: 0.9
homophily_steps: 30

# Feature parameters
feature_dim: 5
cov_scale: 1.0e-4
intra_class_cov: 0.1
inter_class_cov: -0.05
global_cov: 1.0
noise_cov: 1.0

# Model parameters
model_type: linear_gcn  # Options: linear_gcn, two_layer_gcn
hidden_dim: 16
learning_rate: 0.01
weight_decay: 1.0e-3
num_epochs: 1000

# Experiment settings
num_samples: 100              # Number of graph samples per homophily value
num_acc_repeats: 1            # Number of training repetitions for accuracy estimation
num_monte_carlo_samples: 200  # Number of samples for Monte Carlo SNR estimation
```

## Output Structure

The results of experiments are saved in a structured directory:

```
results/[experiment_type]/[experiment_name]_[timestamp]/
├── config.json                    # Saved configuration for reproducibility
├── gcn_study.db                   # Optuna study database
├── summary.csv                    # Summary of results across datasets
├── all_results.json               # Detailed results for all datasets
├── dataset1_results.json          # Results for dataset1
└── dataset2_results.json          # Results for dataset2
```

For sensitivity experiments, the output includes additional visualization files and analysis results.
