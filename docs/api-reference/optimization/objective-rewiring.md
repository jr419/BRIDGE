---
layout: default
title: objective_rewiring
parent: API Reference
---

# objective_rewiring
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `objective_rewiring` function is the optimization objective for Optuna to tune the hyperparameters of the BRIDGE rewiring process and the SelectiveGCN model. It runs the rewiring pipeline with various parameter combinations and returns the negative validation accuracy, which Optuna tries to minimize.

## Function Signature

```python
def objective_rewiring(
    trial: optuna.Trial,
    g: dgl.DGLGraph,
    best_gcn_params: Dict[str, Any],
    all_matrices: List[np.ndarray],
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    early_stopping: int = 50,
    do_hp: bool = False,
    do_self_loop: bool = False,
    do_residual_connections: bool = False,
    dataset_name: str = 'unknown',
    temperature_range: List[float] = None,
    p_add_range: List[float] = None,
    p_remove_range: List[float] = None,
    h_feats_selective_options: List[int] = None,
    n_layers_selective_options: List[int] = None,
    dropout_selective_range: List[float] = None,
    lr_selective_range: List[float] = None,
    wd_selective_range: List[float] = None
) -> float
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `trial` | optuna.Trial | The Optuna trial object for suggesting hyperparameters |
| `g` | dgl.DGLGraph | Input graph |
| `best_gcn_params` | Dict[str, Any] | Best hyperparameters for the base GCN |
| `all_matrices` | List[np.ndarray] | List of permutation matrices to consider |
| `device` | Union[str, torch.device] | Device to perform computations on |
| `n_epochs` | int | Maximum number of training epochs |
| `early_stopping` | int | Number of epochs to look back for early stopping |
| `do_hp` | bool | Whether to use high-pass filters |
| `do_self_loop` | bool | Whether to add self-loops |
| `do_residual_connections` | bool | Whether to use residual connections |
| `dataset_name` | str | Name of the dataset |
| `temperature_range` | List[float] | Range for temperature values [min, max] (default: [1e-5, 2.0]) |
| `p_add_range` | List[float] | Range for edge addition probability [min, max] (default: [0.0, 1.0]) |
| `p_remove_range` | List[float] | Range for edge removal probability [min, max] (default: [0.0, 1.0]) |
| `h_feats_selective_options` | List[int] | Hidden feature dimensions to try for selective GCN |
| `n_layers_selective_options` | List[int] | Number of layers to try for selective GCN |
| `dropout_selective_range` | List[float] | Dropout range for selective GCN [min, max] |
| `lr_selective_range` | List[float] | Learning rate range for selective GCN [min, max] |
| `wd_selective_range` | List[float] | Weight decay range for selective GCN [min, max] |

## Returns

| Return Type | Description |
|-------------|-------------|
| float | Negative validation accuracy (to be minimized) |

## Hyperparameter Search Space

The function samples hyperparameters from the following search spaces:

### Permutation Matrix Selection

For most datasets, the permutation matrix is sampled from the provided list:

```python
matrix_idx = trial.suggest_int('matrix_idx', 0, (len(all_matrices)-1))
```

For specific datasets like Cora, Citeseer, and PubMed, a fixed matrix (index 0) is used:

```python
fixed_matrix_datasets = ["cora", "citeseer", "pubmed"]
if dataset_name.lower() in fixed_matrix_datasets:
    matrix_idx = 0
```

### Rewiring Parameters

```python
p_add = trial.suggest_float('p_add', p_add_range[0], p_add_range[1])
p_remove = trial.suggest_float('p_remove', p_remove_range[0], p_remove_range[1])
temperature = trial.suggest_float('temperature', temperature_range[0], temperature_range[1])
d_out = trial.suggest_float('d_out', 10, np.sqrt(g.number_of_nodes()))
```

### Selective GCN Parameters

```python
h_feats_sel = trial.suggest_categorical('h_feats_selective', h_feats_selective_options)
n_layers_sel = trial.suggest_categorical('n_layers_selective', n_layers_selective_options)
dropout_p_sel = trial.suggest_float('dropout_p_selective', dropout_selective_range[0], dropout_selective_range[1])
model_lr_sel = trial.suggest_float('model_lr_selective', lr_selective_range[0], lr_selective_range[1], log=True)
wd_sel = trial.suggest_float('weight_decay_selective', wd_selective_range[0], wd_selective_range[1], log=True)
```

## Feature Importances

The function records various metrics as trial attributes, which can be used to analyze feature importances:

```python
# Store the permutation matrix used
trial.set_user_attr('P_k', P_k.tolist())

# Store standard metrics
for key, value in stats_dict.items():
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            trial.set_user_attr(f"{key}_{subkey}", subvalue)
    else:
        trial.set_user_attr(key, value)

# Collect and store all float metrics from the results
all_metrics = collect_float_metrics(results_list)
for metric_name, metric_stats in all_metrics.items():
    trial.set_user_attr(f"{metric_name}_mean", metric_stats['mean'])
    trial.set_user_attr(f"{metric_name}_ci", metric_stats['ci'])

# Store graph statistics from first run for reference
original_stats = results_list[0]['original_stats']
rewired_stats = results_list[0]['rewired_stats']
trial.set_user_attr('original_stats', original_stats)
trial.set_user_attr('rewired_stats', rewired_stats)
```

## Usage Examples

### Basic Usage

```python
import optuna
import torch
import dgl
import numpy as np
from bridge.optimization import objective_rewiring
from bridge.utils import generate_all_symmetric_permutation_matrices

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Generate permutation matrices
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)

# Best GCN parameters from previous optimization
best_gcn_params = {
    'h_feats': 64,
    'n_layers': 2,
    'dropout_p': 0.5,
    'model_lr': 1e-3,
    'weight_decay': 5e-4
}

# Define the search ranges
temperature_range = [1e-5, 2.0]
p_add_range = [0.0, 1.0]
p_remove_range = [0.0, 1.0]
h_feats_selective_options = [16, 32, 64, 128]
n_layers_selective_options = [1, 2, 3]
dropout_selective_range = [0.0, 0.7]
lr_selective_range = [1e-4, 1e-1]
wd_selective_range = [1e-6, 1e-3]

# Create and run Optuna study
study = optuna.create_study(direction='minimize')

# Define objective function
def objective(trial):
    return objective_rewiring(
        trial=trial,
        g=g,
        best_gcn_params=best_gcn_params,
        all_matrices=all_matrices,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_epochs=200,
        early_stopping=30,
        do_hp=False,
        do_self_loop=True,
        do_residual_connections=False,
        dataset_name='cora',
        temperature_range=temperature_range,
        p_add_range=p_add_range,
        p_remove_range=p_remove_range,
        h_feats_selective_options=h_feats_selective_options,
        n_layers_selective_options=n_layers_selective_options,
        dropout_selective_range=dropout_selective_range,
        lr_selective_range=lr_selective_range,
        wd_selective_range=wd_selective_range
    )

# Run optimization
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best parameters:", study.best_params)
print("Best validation accuracy:", -study.best_value)
```

### Using with Command-Line Interface

```python
from bridge.main import run_rewiring_experiment

# Parse command-line arguments
args = parse_args()

# Load datasets and prepare experiment
# ...

# Run GCN optimization
gcn_study = optuna.create_study(
    study_name=gcn_study_name,
    storage=f"sqlite:///{results_dir}/gcn_study.db",
    direction='minimize',
    load_if_exists=True
)
gcn_study.optimize(gcn_objective, n_trials=args.num_trials)

# Get best GCN parameters
best_gcn_params = gcn_study.best_params

# Run rewiring optimization
rewiring_study = optuna.create_study(
    study_name=rewiring_study_name,
    storage=f"sqlite:///{results_dir}/gcn_study.db",
    direction='minimize',
    load_if_exists=True
)

# Define objective function for rewiring optimization
def rewiring_objective(trial):
    return objective_rewiring(
        trial, 
        g, 
        best_gcn_params, 
        all_matrices,
        device=device,
        n_epochs=1000,
        early_stopping=args.early_stopping,
        do_hp=do_hp,
        do_self_loop=args.do_self_loop,
        do_residual_connections=args.do_residual,
        dataset_name=dataset_name,
        temperature_range=args.temperature_range,
        p_add_range=args.p_add_range,
        p_remove_range=args.p_remove_range,
        h_feats_selective_options=args.h_feats_selective_options,
        n_layers_selective_options=args.n_layers_selective_options,
        dropout_selective_range=args.dropout_selective_range,
        lr_selective_range=args.lr_selective_range,
        wd_selective_range=args.wd_selective_range
    )

# Run optimization
rewiring_study.optimize(rewiring_objective, n_trials=args.num_trials)
```

## Implementation Details

The `objective_rewiring` function performs the following steps:

1. **Hyperparameter Sampling**:
   - Samples a permutation matrix from the provided list
   - Samples rewiring parameters (p_add, p_remove, temperature, d_out)
   - Samples SelectiveGCN parameters (h_feats, n_layers, dropout, learning rate, weight decay)

2. **BRIDGE Experiment**:
   - Runs the BRIDGE rewiring experiment using `run_bridge_experiment`
   - Uses the best GCN parameters for the cold-start GCN
   - Performs multiple trials and computes confidence intervals

3. **Metric Recording**:
   - Stores all the metrics and statistics as trial attributes
   - Includes graph statistics, performance metrics, and confidence intervals

4. **Optimization Target**:
   - Returns negative validation accuracy for minimization
   - Optuna will try to find the parameters that maximize validation accuracy

The function is designed to be used with Optuna's optimization framework, which supports various sampling methods (grid, random, TPE) and pruning strategies.