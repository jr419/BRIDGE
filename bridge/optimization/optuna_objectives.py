"""
Objective functions for hyperparameter optimization with Optuna.

This module provides objective functions for optimizing Graph Neural Network hyperparameters
and graph rewiring parameters using Optuna.
"""

import torch
import dgl
import numpy as np
import optuna
from typing import Tuple, List, Dict, Union, Optional, Any, Callable
from tqdm import trange
from scipy import stats

from ..models import GCN
from ..training import train, get_metric_type
from ..utils import set_seed, compute_confidence_interval
from ..rewiring import run_bridge_experiment, run_iterative_bridge_experiment


def trial_suggest_or_fixed(
    trial,
    hyperparameter_range,
    name,
    param_type="float",  # "int", "float", or "categorical"
    log_scale=False
):
    """
    Function to sample a hyperparameter value or use a fixed value.
    Fixed values are stored as trial user attributes rather than parameters.
    
    Args:
        trial: Optuna trial object
        hyperparameter_range: Range for the hyperparameter [min, max] or list of options
        name: Name of the hyperparameter
        param_type: Type of parameter ("int", "float", or "categorical")
        log_scale: Whether to sample on log scale (for float parameters)
    
    Returns:
        The sampled or fixed parameter value
    """
    # Fixed value case: single item list or range with equal min/max
    if isinstance(hyperparameter_range, list):
        if len(hyperparameter_range) == 1:
            # Set fixed value as a user attribute, not a parameter
            value = hyperparameter_range[0]
            trial.set_user_attr(name, value)
            return value
        elif len(hyperparameter_range) == 2 and hyperparameter_range[0] == hyperparameter_range[1]:
            # Set fixed value as a user attribute, not a parameter
            value = hyperparameter_range[0]
            trial.set_user_attr(name, value)
            return value
        # Range case: min and max are different
        elif len(hyperparameter_range) == 2 and param_type != "categorical":
            if param_type == "int":
                return trial.suggest_int(name, hyperparameter_range[0], hyperparameter_range[1])
            elif param_type == "float":
                return trial.suggest_float(name, hyperparameter_range[0], hyperparameter_range[1], log=log_scale)
            else:
                raise ValueError(f"Invalid param_type '{param_type}' for range parameter")
        # Categorical options case: either explicitly categorical or list with more than 2 items
        else:
            return trial.suggest_categorical(name, hyperparameter_range)
    # Handle non-list inputs (like single values)
    else:
        # Set fixed value as a user attribute, not a parameter
        value = hyperparameter_range
        trial.set_user_attr(name, value)
        return value



def train_and_evaluate_mpnn(
    g: dgl.DGLGraph,
    h_feats: int,
    n_layers: int,
    dropout_p: float,
    model_lr: float,
    weight_decay: float,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    device: Union[str, torch.device] = 'cpu',
    num_splits: int = 100,
    log_training: bool = False,
    model_type: str = 'GCN',
    do_self_loop: bool = False,
    dataset_name: str = 'unknown'
) -> Tuple[float, float, float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Train and evaluate a standard GCN model multiple times and return mean accuracies with confidence intervals.
    
    Args:
        g: Input graph
        h_feats: Hidden feature dimension
        n_layers: Number of GCN layers
        dropout_p: Dropout probability
        model_lr: Learning rate
        weight_decay: Weight decay for regularization
        n_epochs: Maximum number of training epochs
        early_stopping: Number of epochs to look back for early stopping
        device: Device to perform computations on
        num_splits: Number of times to repeat the experiment
        log_training: Whether to print training progress
        model_type: Type of model to use (default: 'GCN')
        do_self_loop: Whether to add self-loops
        dataset_name: Name of the dataset
        
    Returns:
        Tuple containing:
        - Mean train accuracy
        - Mean validation accuracy
        - Mean test accuracy
        - Confidence intervals for train accuracy: (lower_bound, upper_bound)
        - Confidence intervals for validation accuracy: (lower_bound, upper_bound)
        - Confidence intervals for test accuracy: (lower_bound, upper_bound)
    """
    train_accs = []
    val_accs = []
    test_accs = []
    
    # Check if for multiple splits
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    # If masks are 2D, there are multiple splits
    has_multiple_splits = len(train_mask.shape) > 1
    num_splits = train_mask.shape[1] if has_multiple_splits else num_splits

    if do_self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        
    for split_idx in trange(num_splits):
        set_seed(split_idx)
        
        # Get masks for this split/repeat
        if has_multiple_splits:
            current_train_mask = train_mask[:, split_idx]
            current_val_mask = val_mask[:, split_idx]
            current_test_mask = test_mask[:, split_idx]
        else:
            current_train_mask = train_mask
            current_val_mask = val_mask
            current_test_mask = test_mask

        # Initialize model
        in_feats = g.ndata['feat'].shape[1]
        out_feats = int(g.ndata['label'].max().item()) + 1
        # Standard model only (GCN)
        model = GCN(
            in_feats,
            h_feats,
            out_feats,
            n_layers,
            dropout_p
        )
        model = model.to(device)
        

        # Train model
        train_acc, val_acc, test_acc, _ = train(
            g,
            model,
            current_train_mask,
            current_val_mask,
            current_test_mask,
            model_lr=model_lr,
            optimizer_weight_decay=weight_decay,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            log_training=log_training,
            metric_type=get_metric_type(dataset_name)
        )
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    # Calculate means
    mean_train = np.mean(train_accs)
    mean_val = np.mean(val_accs)
    mean_test = np.mean(test_accs)

    # Calculate 95% confidence intervals
    # def get_confidence_interval(data: List[float]) -> Tuple[float, float]:
    #     confidence = 0.95
    #     data = np.array(data)
    #     n = len(data)
    #     se = stats.sem(data)
    #     ci = stats.t.interval(confidence, n-1, loc=np.mean(data), scale=se)
    #     return ci
    
    #mean_, lower_bound, upper_bound
    _, *train_ci = compute_confidence_interval(train_accs, confidence=0.95)
    _, *val_ci = compute_confidence_interval(val_accs, confidence=0.95)
    _, *test_ci = compute_confidence_interval(test_accs, confidence=0.95)

    return (
        mean_train,
        mean_val,
        mean_test,
        train_ci,
        val_ci,
        test_ci
    )


def objective_mpnn(
    trial: optuna.Trial,
    g: dgl.DGLGraph,
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    num_splits: int = 100,
    early_stopping: int = 50,
    model_type: str = 'GCN',
    dataset_name: str = 'unknown',
    h_feats_options: List[int] = None,
    n_layers_options: List[int] = None,
    dropout_range: List[float] = None,
    lr_range: List[float] = None,
    weight_decay_range: List[float] = None
) -> float:
    """
    Objective function for optimizing base GCN hyperparameters with Optuna.
    
    Args:
        trial: Optuna trial object
        g: Input graph
        device: Device to perform computations on
        n_epochs: Maximum number of training epochs
        num_splits: Number of splits/repetitions for statistical significance
        early_stopping: Number of epochs to look back for early stopping
        model_type: Type of model to use (default: 'mpnn')
        dataset_name: Name of the dataset
        h_feats_options: List of hidden feature dimensions to try (default: [16, 32, 64, 128])
        n_layers_options: List of layer counts to try (default: [1, 2, 3])
        dropout_range: Range for dropout values [min, max] (default: [0.0, 0.7])
        
    Returns:
        float: Negative validation accuracy (to be minimized)
    """
    # Use provided options or fall back to defaults
    h_feats_options = h_feats_options or [16, 32, 64, 128]
    n_layers_options = n_layers_options or [1, 2, 3]
    dropout_range = dropout_range or [0.0, 0.7]
    lr_range = lr_range or [1e-4, 1e-1]
    weight_decay_range = weight_decay_range or [1e-6, 1e-3]
    
    # Sample hyperparameters for GCN
    # trial,
    #     n_rewire_iterations_range,
    #     'n_rewire_iterations',
    #     param_type="int"
    # params = {
    #     'h_feats': trial.suggest_categorical('h_feats', h_feats_options),
    #     'n_layers': trial_suggest_or_fixed(trial, n_layers_options, 'n_layers', param_type='categorical'),
    #     'dropout_p': trial.suggest_float('dropout_p', dropout_range[0], dropout_range[1]),
    #     'model_lr': trial.suggest_float('model_lr', lr_range[0], lr_range[1], log=True),
    #     'weight_decay': trial.suggest_float('weight_decay', weight_decay_range[0], weight_decay_range[1], log=True)
    # }
    
    # inside your objective function…

    params = {
        'h_feats': trial_suggest_or_fixed(
            trial,
            h_feats_options,              # e.g. [16, 32, 64]
            'h_feats',
            param_type='categorical'
        ),
        'n_layers': trial_suggest_or_fixed(
            trial,
            n_layers_options,             # e.g. [1,2,3] or [2] if fixed
            'n_layers',
            param_type='categorical'
        ),
        'dropout_p': trial_suggest_or_fixed(
            trial,
            dropout_range,                # e.g. [0.0, 0.5] or [0.2] if fixed
            'dropout_p',
            param_type='float'
        ),
        'model_lr': trial_suggest_or_fixed(
            trial,
            lr_range,                     # e.g. [1e-5, 1e-2] or [1e-3]
            'model_lr',
            param_type='float',
            log_scale=True
        ),
        'weight_decay': trial_suggest_or_fixed(
            trial,
            weight_decay_range,           # e.g. [1e-6, 1e-3] or [0.0]
            'weight_decay',
            param_type='float',
            log_scale=True
        ),
    }
    
    # Train and evaluate
    train_acc, val_acc, test_acc, train_acc_ci, val_acc_ci, test_acc_ci = train_and_evaluate_mpnn(
        g=g,
        device=device,
        num_splits=num_splits,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        model_type=model_type,
        dataset_name=dataset_name,
        **params
    )
    
    # Store metrics
    trial.set_user_attr('train_acc', train_acc)
    trial.set_user_attr('val_acc', val_acc)
    trial.set_user_attr('test_acc', test_acc)
    trial.set_user_attr('train_acc_ci', train_acc_ci)
    trial.set_user_attr('val_acc_ci', val_acc_ci)
    trial.set_user_attr('test_acc_ci', test_acc_ci)
    
    return -val_acc  # Minimize negative validation accuracy


def collect_float_metrics(results_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Union[float, Tuple[float, float]]]]:
    """
    Collect all float metrics from the results list and compute their means and confidence intervals.
    
    Args:
        results_list: List of result dictionaries from experiments
        
    Returns:
        Dict[str, Dict[str, Union[float, Tuple[float, float]]]]: 
            Dictionary mapping metric names to their statistics (mean and confidence interval)
    """
    # Initialize dictionaries to store metrics
    float_metrics = {}
    
    def process_dict(d, prefix=''):
        for key, value in d.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                process_dict(value, full_key)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if full_key not in float_metrics:
                    float_metrics[full_key] = []
                float_metrics[full_key].append(float(value))
    
    # Process each result in the list
    for result in results_list:
        process_dict(result)
    
    # Compute statistics for each metric
    stats = {}
    for key, values in float_metrics.items():
        if len(values) > 0:  # Ensure there are values
            mean, lower, upper = compute_confidence_interval(values)
            stats[key] = {
                'mean': mean,
                'ci': (lower, upper)
            }
    
    return stats


def objective_rewiring(
    trial: optuna.Trial,
    g: dgl.DGLGraph,
    best_mpnn_params: Dict[str, Any],
    all_matrices: List[np.ndarray],
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    num_splits: int = 100,
    early_stopping: int = 50,
    model_type: str = 'GCN',
    do_self_loop: bool = False,
    dataset_name: str = 'unknown'
) -> float:
    """
    Objective function for optimizing BRIDGE rewiring with a standard model only.
    Rewiring uses full resampling with hard predictions (no temperature, no p_add/p_remove).
    """
    # Select permutation matrix
    fixed_matrix_datasets = []
    if dataset_name.lower() in fixed_matrix_datasets:
        matrix_idx = 0
    else:
        matrix_idx = trial.suggest_int('matrix_idx', 0, (len(all_matrices)-1)) 
    d_out = trial.suggest_float('d_out', 10, np.sqrt(g.number_of_nodes()))

    P_k = all_matrices[matrix_idx]

    # Use best base model parameters
    params_mpnn = {
        'h_feats_mpnn': best_mpnn_params.get('h_feats', 64),
        'n_layers_mpnn': best_mpnn_params.get('n_layers', 1),
        'dropout_p_mpnn': best_mpnn_params.get('dropout_p', 0.5),
        'model_lr_mpnn': best_mpnn_params.get('model_lr', 1e-3),
        'wd_mpnn': best_mpnn_params.get('weight_decay', 1e-5),
    }

    # Run rewiring experiment (aggregated over splits)
    stats_dict, results_list = run_bridge_experiment(
        g,
        P_k=P_k,
        d_out=d_out,
        num_graphs=1,
        device=device,
        num_splits=num_splits,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=False,
        dataset_name=dataset_name,
        do_self_loop=do_self_loop,
        **params_mpnn,
    )

    # Store the permutation matrix used
    trial.set_user_attr('P_k', P_k.tolist())

    # Store metrics
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

    return -stats_dict['val_acc_mean']


def objective_iterative_rewiring(
    trial: optuna.Trial,
    g: dgl.DGLGraph,
    best_mpnn_params: Dict[str, Any],
    all_matrices: List[np.ndarray],
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    num_splits: int = 100,
    early_stopping: int = 50,
    model_type: str = 'GCN',
    do_self_loop: bool = False,
    dataset_name: str = 'unknown',
    n_rewire_iterations_range: List[int] = None,
    
    rewiring_method: str = "bridge",
    sdrf_tau_range: list = [0.01, 300],
    sdrf_n_iterations_range: list = [1, 300],
    sdrf_c_plus_range: list = [0, 50],
    digl_diffusion_type='ppr',
    digl_alpha_range=[0.05, 0.25],
    digl_k_options=[32, 64, 128],
    digl_t_range=[1.0, 10.0],
    digl_epsilon_range=[0.001, 0.1],
    simulated_acc: Optional[float] = None
) -> float:
    """
    Objective for optimizing iterative rewiring with a standard model only.
    Rewiring uses full resampling with hard predictions.
    """
    # Defaults
    n_rewire_iterations_range = n_rewire_iterations_range or [1, 20]

    # Hyperparameters
    n_rewire = trial_suggest_or_fixed(trial, n_rewire_iterations_range, 'n_rewire_iterations', param_type="int")

    fixed_matrix_datasets = []
    if dataset_name.lower() in fixed_matrix_datasets:
        matrix_idx = trial_suggest_or_fixed(trial, [0, 0], 'matrix_idx', param_type="int")
    else:
        matrix_idx = trial_suggest_or_fixed(trial, [0, (len(all_matrices)-1)], 'matrix_idx', param_type="int")

    d_out = trial_suggest_or_fixed(
        trial,
        [(10 if model_type != 'GAT' else 10), (np.sqrt(g.number_of_nodes()) if model_type != 'GAT' else 10)],
        'd_out',
        param_type="float"
    )

    P_k = all_matrices[matrix_idx]

    # Use best base model params
    params_mpnn = {
        'h_feats_mpnn': best_mpnn_params.get('h_feats', 64),
        'n_layers_mpnn': best_mpnn_params.get('n_layers', 1),
        'dropout_p_mpnn': best_mpnn_params.get('dropout_p', 0.5),
        'model_lr_mpnn': best_mpnn_params.get('model_lr', 1e-3),
        'wd_mpnn': best_mpnn_params.get('weight_decay', 1e-5),
    }

    

    tau = trial_suggest_or_fixed(trial, sdrf_tau_range, 'sdrf_tau', param_type="float")
    sdrf_iterations = trial_suggest_or_fixed(trial, sdrf_n_iterations_range, 'sdrf_iterations', param_type="int")
    c_plus = trial_suggest_or_fixed(trial, sdrf_c_plus_range, 'sdrf_c_plus', param_type="float")

    digl_alpha = trial_suggest_or_fixed(trial, digl_alpha_range, 'digl_alpha', param_type="float")
    digl_k = trial_suggest_or_fixed(trial, digl_k_options, 'digl_k', param_type="categorical")
    digl_t = trial_suggest_or_fixed(trial, digl_t_range, 'digl_t', param_type="float")
    digl_epsilon = trial_suggest_or_fixed(trial, digl_epsilon_range, 'digl_epsilon', param_type="float")

    # Run iterative rewiring experiment
    stats_dict, results_list = run_iterative_bridge_experiment(
        g,
        P_k=P_k,
        model_type=model_type,
        d_out=d_out,
        num_graphs=1,
        device=device,
        num_repeats=num_splits,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=False,
        dataset_name=dataset_name,
        do_self_loop=do_self_loop,
        n_rewire=n_rewire,
        rewiring_method=rewiring_method,
        tau=tau,
        sdrf_iterations=sdrf_iterations,
        c_plus=c_plus,
        digl_diffusion_type=digl_diffusion_type,
        digl_alpha=digl_alpha,
        digl_k=digl_k,
        digl_t=digl_t,
        digl_epsilon=digl_epsilon,
        simulated_acc=simulated_acc,
        **params_mpnn,
    )

    # Store the permutation matrix used
    trial.set_user_attr('P_k', P_k.tolist())

    # Store metrics
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

    # Store iterative rewiring specific information
    trial.set_user_attr('n_rewire_iterations', n_rewire)

    trial.set_user_attr('sdrf_tau', tau)
    trial.set_user_attr('sdrf_iterations', sdrf_iterations)
    trial.set_user_attr('sdrf_c_plus', c_plus)

    trial.set_user_attr('digl_diffusion_type', digl_diffusion_type)
    trial.set_user_attr('digl_alpha', digl_alpha)
    trial.set_user_attr('digl_k', digl_k)
    trial.set_user_attr('digl_t', digl_t)
    trial.set_user_attr('digl_epsilon', digl_epsilon)

    return -stats_dict['val_acc_mean']  # Minimize negative validation accuracy
