"""
Objective functions for hyperparameter optimization with Optuna.

This module provides objective functions for optimizing Graph Neural Network hyperparameters
and graph rewiring parameters using Optuna.
"""

import torch
import dgl
import numpy as np
import optuna
from typing import Tuple, List, Dict, Union, Optional, Any
from tqdm import trange
from scipy import stats

from ..models import GCN
from ..training import train, get_metric_type
from ..utils import set_seed, compute_confidence_interval
from ..rewiring import run_bridge_experiment, run_iterative_bridge_experiment


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
    num_splits: int = 100,
    log_training: bool = False,
    do_hp: bool = False,
    do_residual_connections: bool = False,
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
        do_hp: Whether to use higher-order polynomial filters
        do_residual_connections: Whether to use residual connections
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
        model = GCN(
            in_feats, h_feats, out_feats, n_layers, dropout_p, 
            residual_connection=do_residual_connections,
            do_hp=do_hp
        ).to(device)

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
    def get_confidence_interval(data: List[float]) -> Tuple[float, float]:
        confidence = 0.95
        data = np.array(data)
        n = len(data)
        se = stats.sem(data)
        ci = stats.t.interval(confidence, n-1, loc=np.mean(data), scale=se)
        return ci

    train_ci = get_confidence_interval(train_accs)
    val_ci = get_confidence_interval(val_accs)
    test_ci = get_confidence_interval(test_accs)

    return (
        mean_train,
        mean_val,
        mean_test,
        train_ci,
        val_ci,
        test_ci
    )


def objective_gcn(
    trial: optuna.Trial,
    g: dgl.DGLGraph,
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    num_splits: int = 100,
    early_stopping: int = 50,
    do_hp: bool = False,
    do_residual_connections: bool = False,
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
        early_stopping: Number of epochs to look back for early stopping
        do_hp: Whether to use higher-order polynomial filters
        do_residual_connections: Whether to use residual connections
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
    params = {
        'h_feats': trial.suggest_categorical('h_feats', h_feats_options),
        'n_layers': trial.suggest_categorical('n_layers', n_layers_options),
        'dropout_p': trial.suggest_float('dropout_p', dropout_range[0], dropout_range[1]),
        'model_lr': trial.suggest_float('model_lr', lr_range[0], lr_range[1], log=True),
        'weight_decay': trial.suggest_float('weight_decay', weight_decay_range[0], weight_decay_range[1], log=True)
    }
    
    # Train and evaluate
    train_acc, val_acc, test_acc, train_acc_ci, val_acc_ci, test_acc_ci = train_and_evaluate_gcn(
        g=g,
        device=device,
        num_splits=num_splits,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        do_hp=do_hp,
        do_residual_connections=do_residual_connections,
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
    best_gcn_params: Dict[str, Any],
    all_matrices: List[np.ndarray],
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    num_splits: int = 100,
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

) -> float:
    """
    Objective function for optimizing rewiring and selective GCN parameters with Optuna.
    
    Uses the best base GCN parameters for the cold-start GCN and optimizes the
    remaining parameters for graph rewiring and the selective GCN.
    
    Args:
        trial: Optuna trial object
        g: Input graph
        best_gcn_params: Best hyperparameters for the base GCN
        all_matrices: List of permutation matrices to consider
        device: Device to perform computations on
        n_epochs: Maximum number of training epochs
        early_stopping: Number of epochs to look back for early stopping
        do_hp: Whether to use higher-order polynomial filters
        do_self_loop: Whether to add self-loops
        do_residual_connections: Whether to use residual connections
        dataset_name: Name of the dataset
        temperature_range: Range for temperature values [min, max] (default: [1e-5, 2.0])
        p_add_range: Range for edge addition probability [min, max] (default: [0.0, 1.0])
        p_remove_range: Range for edge removal probability [min, max] (default: [0.0, 1.0])
        
    Returns:
        float: Negative validation accuracy (to be minimized)
    """
    # Use provided ranges or fall back to defaults
    temperature_range = temperature_range or [1e-5, 2.0]
    p_add_range = p_add_range or [0.0, 1.0]
    p_remove_range = p_remove_range or [0.0, 1.0]
    h_feats_selective_options = h_feats_selective_options or [16, 32, 64, 128]
    n_layers_selective_options = n_layers_selective_options or [1, 2, 3]
    dropout_selective_range = dropout_selective_range or [0.0, 0.7]
    lr_selective_range = lr_selective_range or [1e-4, 1e-1]
    wd_selective_range = wd_selective_range or [1e-6, 1e-3]
    
    # Sample hyperparameters (including the permutation matrix index)
    fixed_matrix_datasets = ["cora", "citeseer", "pubmed"]
    
    if dataset_name.lower() in fixed_matrix_datasets:
        matrix_idx = 0
    else:
        matrix_idx = trial.suggest_int('matrix_idx', 0, (len(all_matrices)-1)) 
    
    p_add = trial.suggest_float('p_add', p_add_range[0], p_add_range[1])
    p_remove = trial.suggest_float('p_remove', p_remove_range[0], p_remove_range[1])
    print(temperature_range)
    temperature = trial.suggest_float('temperature', temperature_range[0], temperature_range[1])
    d_out = trial.suggest_float('d_out', 10, np.sqrt(g.number_of_nodes()))

    P_k = all_matrices[matrix_idx]
    num_graphs = 1
    
    # Use best GCN parameters for cold-start GCN
    h_feats_gcn = best_gcn_params['h_feats']
    n_layers_gcn = best_gcn_params['n_layers']
    dropout_p_gcn = best_gcn_params['dropout_p']
    model_lr_gcn = best_gcn_params['model_lr']
    wd_gcn = best_gcn_params['weight_decay']

    # Sample parameters for selective GCN
    h_feats_sel = trial.suggest_categorical('h_feats_selective', h_feats_selective_options)
    n_layers_sel = trial.suggest_categorical('n_layers_selective', n_layers_selective_options)
    dropout_p_sel = trial.suggest_float('dropout_p_selective', dropout_selective_range[0], dropout_selective_range[1])
    model_lr_sel = trial.suggest_float('model_lr_selective', lr_selective_range[0], lr_selective_range[1], log=True)
    wd_sel = trial.suggest_float('weight_decay_selective', wd_selective_range[0], wd_selective_range[1], log=True)


    # Run rewiring experiment
    stats_dict, results_list = run_bridge_experiment(
        g,
        P_k=P_k,
        h_feats_gcn=h_feats_gcn,
        n_layers_gcn=n_layers_gcn,
        dropout_p_gcn=dropout_p_gcn,
        model_lr_gcn=model_lr_gcn,
        wd_gcn=wd_gcn,
        h_feats_selective=h_feats_sel,
        n_layers_selective=n_layers_sel,
        dropout_p_selective=dropout_p_sel,
        model_lr_selective=model_lr_sel,
        wd_selective=wd_sel,
        temperature=temperature,
        p_add=p_add,
        p_remove=p_remove,
        d_out=d_out,
        num_graphs=num_graphs,
        device=device,
        num_splits=num_splits,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=False,
        dataset_name=dataset_name,
        do_hp=do_hp,
        do_self_loop=do_self_loop,
        do_residual_connections=do_residual_connections
    )

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
    
    return -stats_dict['val_acc_mean']  # Minimize negative validation accuracy



def objective_iterative_rewiring(
    trial: optuna.Trial,
    g: dgl.DGLGraph,
    best_gcn_params: Dict[str, Any],
    all_matrices: List[np.ndarray],
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    num_splits: int = 100,
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
    wd_selective_range: List[float] = None,
    n_rewire_iterations: int = 10,
    use_sgc: bool = True,
    sgc_k: int = 2
) -> float:
    """
    Objective function for optimizing iterative rewiring and selective GCN parameters with Optuna.
    
    Uses the best base GCN parameters for the cold-start GCN and optimizes the
    remaining parameters for iterative graph rewiring and the selective GCN.
    
    Args:
        trial: Optuna trial object
        g: Input graph
        best_gcn_params: Best hyperparameters for the base GCN
        all_matrices: List of permutation matrices to consider
        device: Device to perform computations on
        n_epochs: Maximum number of training epochs
        num_splits: Number of splits/repetitions for statistical significance
        early_stopping: Number of epochs to look back for early stopping
        do_hp: Whether to use higher-order polynomial filters
        do_self_loop: Whether to add self-loops
        do_residual_connections: Whether to use residual connections
        dataset_name: Name of the dataset
        temperature_range: Range for temperature values [min, max]
        p_add_range: Range for edge addition probability [min, max]
        p_remove_range: Range for edge removal probability [min, max]
        h_feats_selective_options: Options for hidden feature dimensions in selective GCN
        n_layers_selective_options: Options for number of layers in selective GCN
        dropout_selective_range: Range for dropout probability in selective GCN
        lr_selective_range: Range for learning rate in selective GCN
        wd_selective_range: Range for weight decay in selective GCN
        n_rewire_iterations: Number of rewiring iterations to perform
        use_sgc: Whether to use SGC for faster rewiring
        sgc_k: Number of propagation steps for SGC
        
    Returns:
        float: Negative validation accuracy (to be minimized)
    """
    # Use provided ranges or fall back to defaults
    temperature_range = temperature_range or [1e-5, 2.0]
    p_add_range = p_add_range or [0.0, 1.0]
    p_remove_range = p_remove_range or [0.0, 1.0]
    h_feats_selective_options = h_feats_selective_options or [16, 32, 64, 128]
    n_layers_selective_options = n_layers_selective_options or [1, 2, 3]
    dropout_selective_range = dropout_selective_range or [0.0, 0.7]
    lr_selective_range = lr_selective_range or [1e-4, 1e-1]
    wd_selective_range = wd_selective_range or [1e-6, 1e-3]
    
    # Sample hyperparameters (including the permutation matrix index)
    fixed_matrix_datasets = ["cora", "citeseer", "pubmed"]
    
    if dataset_name.lower() in fixed_matrix_datasets:
        matrix_idx = 0
    else:
        matrix_idx = trial.suggest_int('matrix_idx', 0, (len(all_matrices)-1)) 
    
    p_add = trial.suggest_float('p_add', p_add_range[0], p_add_range[1])
    p_remove = trial.suggest_float('p_remove', p_remove_range[0], p_remove_range[1])
    temperature = trial.suggest_float('temperature', temperature_range[0], temperature_range[1])
    d_out = trial.suggest_float('d_out', 10, np.sqrt(g.number_of_nodes()))

    P_k = all_matrices[matrix_idx]
    num_graphs = 1
    
    # Use best GCN parameters for cold-start GCN
    h_feats_gcn = best_gcn_params['h_feats']
    n_layers_gcn = best_gcn_params['n_layers']
    dropout_p_gcn = best_gcn_params['dropout_p']
    model_lr_gcn = best_gcn_params['model_lr']
    wd_gcn = best_gcn_params['weight_decay']

    # Sample parameters for selective GCN
    h_feats_sel = trial.suggest_categorical('h_feats_selective', h_feats_selective_options)
    n_layers_sel = trial.suggest_categorical('n_layers_selective', n_layers_selective_options)
    dropout_p_sel = trial.suggest_float('dropout_p_selective', dropout_selective_range[0], dropout_selective_range[1])
    model_lr_sel = trial.suggest_float('model_lr_selective', lr_selective_range[0], lr_selective_range[1], log=True)
    wd_sel = trial.suggest_float('weight_decay_selective', wd_selective_range[0], wd_selective_range[1], log=True)

    # Run rewiring experiment with iterative approach
    stats_dict, results_list = run_iterative_bridge_experiment(
        g,
        P_k=P_k,
        h_feats_gcn=h_feats_gcn,
        n_layers_gcn=n_layers_gcn,
        dropout_p_gcn=dropout_p_gcn,
        model_lr_gcn=model_lr_gcn,
        wd_gcn=wd_gcn,
        h_feats_selective=h_feats_sel,
        n_layers_selective=n_layers_sel,
        dropout_p_selective=dropout_p_sel,
        model_lr_selective=model_lr_sel,
        wd_selective=wd_sel,
        temperature=temperature,
        p_add=p_add,
        p_remove=p_remove,
        d_out=d_out,
        num_graphs=num_graphs,
        device=device,
        num_repeats=num_splits,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=False,
        dataset_name=dataset_name,
        do_hp=do_hp,
        do_self_loop=do_self_loop,
        do_residual_connections=do_residual_connections,
        use_sgc=use_sgc,
        n_rewire=n_rewire_iterations,
        K=sgc_k
    )

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
    
    # Store iterative rewiring specific information
    trial.set_user_attr('n_rewire_iterations', n_rewire_iterations)
    trial.set_user_attr('use_sgc', use_sgc)
    trial.set_user_attr('sgc_k', sgc_k)
    
    return -stats_dict['val_acc_mean']  # Minimize negative validation accuracy