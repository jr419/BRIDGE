---
layout: default
title: objective_gcn
parent: API Reference
---

# objective_gcn
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `objective_gcn` function is an objective function for Optuna to optimize base GCN hyperparameters. It trains and evaluates a standard GCN with various hyperparameter combinations and returns the negative validation accuracy, which Optuna tries to minimize.

## Function Signature

```python
def objective_gcn(
    trial: optuna.Trial,
    g: dgl.DGLGraph,
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    early_stopping: int = 50,
    do_hp: bool = False,
    do_residual_connections: bool = False,
    dataset_name: str = 'unknown',
    h_feats_options: List[int] = None,
    n_layers_options: List[int] = None,
    dropout_range: List[float] = None,
    lr_range: List[float] = None,
    weight_decay_range: List[float] = None
) -> float
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `trial` | optuna.Trial | The Optuna trial object for suggesting hyperparameters |
| `g` | dgl.DGLGraph | Input graph |
| `device` | Union[str, torch.device] | Device to perform computations on |
| `n_epochs` | int | Maximum number of training epochs |
| `early_stopping` | int | Number of epochs to look back for early stopping |
| `do_hp` | bool | Whether to use high-pass filters |
| `do_residual_connections` | bool | Whether to use residual connections |
| `dataset_name` | str | Name of the dataset |
| `h_feats_options` | List[int] | List of hidden feature dimensions to try (default: [16, 32, 64, 128]) |
| `n_layers_options` | List[int] | List of layer counts to try (default: [1, 2, 3]) |
| `dropout_range` | List[float] | Range for dropout values [min, max] (default: [0.0, 0.7]) |
| `lr_range` | List[float] | Range for learning rate values [min, max] (default: [1e-4, 1e-1]) |
| `weight_decay_range` | List[float] | Range for weight decay values [min, max] (default: [1e-6, 1e-3]) |

## Returns

| Return Type | Description |
|-------------|-------------|
| float | Negative validation accuracy (to be minimized) |

## Hyperparameter Search Space

The function samples hyperparameters from the following search spaces:

### Model Architecture Parameters

```python
# Sample hyperparameters for GCN
params = {
    'h_feats': trial.suggest_categorical('h_feats', h_feats_options),
    'n_layers': trial.suggest_categorical('n_layers', n_layers_options),
    'dropout_p': trial.suggest_float('dropout_p', dropout_range[0], dropout_range[1]),
    'model_lr': trial.suggest_float('model_lr', lr_range[0], lr_range[1], log=True),
    'weight_decay': trial.suggest_float('weight_decay', weight_decay_range[0], weight_decay_range[1], log=True)
}
```

### Default Parameter Values

If not explicitly provided, the function uses these default values:

```python
h_feats_options = h_feats_options or [16, 32, 64, 128]
n_layers_options = n_layers_options or [1, 2, 3]
dropout_range = dropout_range or [0.0, 0.7]
lr_range = lr_range or [1e-4, 1e-1]
weight_decay_range = weight_decay_range or [1e-6, 1e-3]
```

## Trial Attributes

The function records various metrics as trial attributes, which can be used for analysis:

```python
# Store metrics
trial.set_user_attr('train_acc', train_acc)
trial.set_user_attr('val_acc', val_acc)
trial.set_user_attr('test_acc', test_acc)
trial.set_user_attr('train_acc_ci', train_acc_ci)
trial.set_user_attr('val_acc_ci', val_acc_ci)
trial.set_user_attr('test_acc_ci', test_acc_ci)
```

## Usage Examples

### Basic Usage

```python
import optuna
import torch
import dgl
from bridge.optimization import objective_gcn

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Create and run Optuna study
study = optuna.create_study(direction='minimize')

# Define objective function
def objective(trial):
    return objective_gcn(
        trial=trial,
        g=g,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_epochs=200,
        early_stopping=30,
        do_hp=False,
        do_residual_connections=False,
        dataset_name='cora'
    )

# Run optimization
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best parameters:", study.best_params)
print("Best validation accuracy:", -study.best_value)
print("Best test accuracy:", study.best_trial.user_attrs['test_acc'])
```

### Custom Hyperparameter Ranges

```python
import optuna
from bridge.optimization import objective_gcn

# Define custom hyperparameter ranges
h_feats_options = [32, 64, 128, 256]
n_layers_options = [2, 3, 4]
dropout_range = [0.3, 0.8]
lr_range = [1e-3, 1e-2]
weight_decay_range = [1e-5, 1e-4]

# Create and run Optuna study
study = optuna.create_study(direction='minimize')

# Define objective function with custom ranges
def objective(trial):
    return objective_gcn(
        trial=trial,
        g=g,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_epochs=200,
        early_stopping=30,
        do_hp=True,  # Use high-pass filters
        do_residual_connections=True,  # Use residual connections
        dataset_name='citeseer',
        h_feats_options=h_feats_options,
        n_layers_options=n_layers_options,
        dropout_range=dropout_range,
        lr_range=lr_range,
        weight_decay_range=weight_decay_range
    )

# Run optimization
study.optimize(objective, n_trials=100)
```

### Using with Command-Line Interface

```python
from bridge.main import run_rewiring_experiment

# Parse command-line arguments
args = parse_args()

# Load datasets and prepare experiment
# ...

# Create and setup the objective function for GCN optimization
def gcn_objective(trial):
    return objective_gcn(
        trial, 
        g, 
        device=device,
        n_epochs=1000,
        early_stopping=args.early_stopping,
        do_hp=do_hp,
        do_residual_connections=args.do_residual,
        dataset_name=dataset_name,
        h_feats_options=args.gcn_h_feats,
        n_layers_options=args.gcn_n_layers,
        dropout_range=args.gcn_dropout_range
    )

# Create and run study for GCN optimization
gcn_study = optuna.create_study(
    study_name=gcn_study_name,
    storage=f"sqlite:///{results_dir}/gcn_study.db",
    direction='minimize',
    load_if_exists=True
)

gcn_study.optimize(gcn_objective, n_trials=args.num_trials)
```

## Implementation Details

The `objective_gcn` function performs the following steps:

1. **Hyperparameter Sampling**:
   - Samples GCN architecture parameters (hidden features, number of layers, dropout)
   - Samples optimization parameters (learning rate, weight decay)

2. **GCN Training and Evaluation**:
   - Calls `train_and_evaluate_gcn` with the sampled parameters
   - Performs multiple trials to compute means and confidence intervals

3. **Metric Recording**:
   - Stores all the performance metrics and confidence intervals as trial attributes
   - Includes train, validation, and test accuracy

4. **Optimization Target**:
   - Returns negative validation accuracy for minimization
   - Optuna will try to find the parameters that maximize validation accuracy

### train_and_evaluate_gcn Function

The `objective_gcn` function relies on `train_and_evaluate_gcn`, which:

```python
def train_and_evaluate_gcn(
    g: dgl.DGLGraph,
    h_feats: int,
    n_layers: int,
    dropout_p: float,
    model_lr: float,
    weight_decay: float,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    device: Union[str, torch.device] = 'cpu',
    num_repeats: int = 10,
    log_training: bool = False,
    do_hp: bool = False,
    do_residual_connections: bool = False,
    dataset_name: str = 'unknown'
) -> Tuple[float, float, float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]
```

This helper function:

1. Trains the GCN multiple times with different random seeds or data splits
2. Computes mean accuracies and confidence intervals
3. Returns comprehensive statistics about the model's performance
