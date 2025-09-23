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

BRIDGE provides a command-line interface (CLI) for running experiments. The main entry point is the `bridge.main` module:

```bash
python -m bridge.main [options]
```

Rewiring uses hard predictions (no temperature) and full resampling (no p_add/p_remove).

The CLI supports two primary experiment types:
1. Rewiring: Graph rewiring optimization experiments
2. Sensitivity: Sensitivity and SNR analysis experiments

## Basic Usage

### Rewiring Experiments

```bash
python -m bridge.main --experiment_type rewiring --dataset_type standard --standard_datasets cora citeseer --num_trials 100 --experiment_name my_experiment
```

To use the iterative algorithm:

```bash
python -m bridge.main --experiment_type rewiring --standard_datasets cora \
    --use_iterative_rewiring --num_trials 50 --experiment_name my_iterative_run
```

### Sensitivity Analysis

```bash
python -m bridge.main --experiment_type sensitivity --config config_examples/snr_analysis.yaml
```

## Configuration Options

Run this to see all available options:

```bash
python -m bridge.main --help
```

### Using Configuration Files

```bash
python -m bridge.main --config config_examples/real_datasets_test.yaml
```

Command-line arguments take precedence over options in the configuration file.

## Experiment Type

```bash
--experiment_type {rewiring,sensitivity}
```

## General Settings

```bash
--seed INTEGER                  Random seed for reproducibility
--device TEXT                   Device to use (cuda or cpu)
--num_trials INTEGER            Number of optimization trials
--num_splits INTEGER            Number of splits for CI calculation
--experiment_name TEXT          Name of the experiment
```

## Model Settings

```bash
--do_self_loop                  Add self-loops to graphs
--do_residual                   Use residual connections in the base model
--early_stopping INTEGER        Early stopping patience
```

## Dataset Settings

```bash
--dataset_type {standard,synthetic}  Type of dataset to use
--standard_datasets TEXT...          List of standard datasets to use
```

Supported standard datasets include: cora, citeseer, pubmed, actor, chameleon, squirrel, wisconsin, cornell, texas, minesweeper, tolokers.

## Synthetic Dataset Parameters

```bash
--syn_nodes INTEGER             Number of nodes for synthetic dataset
--syn_classes INTEGER           Number of classes for synthetic dataset
--syn_homophily FLOAT          Homophily for synthetic dataset
--syn_degree FLOAT             Mean degree for synthetic dataset
--syn_features INTEGER         Number of features for synthetic dataset
```

## Optimization Parameters

### Base Model Hyperparameters

```bash
--model_type TEXT               Model type (GCN)
--mpnn_h_feats INTEGER...       Hidden feature dimensions to try for base model
--mpnn_n_layers INTEGER...      Number of layers to try for base model
--mpnn_dropout_range FLOAT FLOAT Dropout range for base model [min, max]
--lr_mpnn_range FLOAT FLOAT     Learning rate range for base model [min, max]
--wd_mpnn_range FLOAT FLOAT     Weight decay range for base model [min, max]
```

### Iterative Rewiring

```bash
--use_iterative_rewiring        Use the iterative rewiring pipeline
--n_rewire_iterations_range INTEGER INTEGER   Range of rewiring iterations [min, max]
--rewiring_method {bridge,sdrf,digl} Choose rewiring method
--sdrf_tau_range FLOAT FLOAT    SDRF tau range
--sdrf_iterations_range INTEGER INTEGER   SDRF iteration range
--sdrf_c_plus_range FLOAT FLOAT SDRF c_plus range
--digl_diffusion_type {ppr,heat}  DIGL diffusion type
--digl_alpha_range FLOAT FLOAT  DIGL alpha range
--digl_t_range FLOAT FLOAT      DIGL t range
--digl_epsilon_range FLOAT FLOAT DIGL epsilon range
--digl_k_options INTEGER...     DIGL k options
```

## Symmetry Checking

```bash
--check_symmetry                Check and enforce graph symmetry
```

## Output Structure

Results are saved under `results/[experiment_type]/[experiment_name]_[timestamp]/` with configuration, per-dataset JSONs, and a summary CSV.
