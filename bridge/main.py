"""
Main entry point for the BRIDGE library.

BRIDGE (Block Rewiring from Inference-Derived Graph Ensembles) is a technique for
optimizing graph neural networks through graph rewiring.
"""

import os
import torch
import dgl
import numpy as np
import json
import yaml
import optuna
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse
from typing import List, Dict, Any

# Use absolute imports

from bridge.utils.dataset_processing import add_train_val_test_splits
from bridge.utils import set_seed, generate_all_symmetric_permutation_matrices, check_symmetry
from bridge.optimization import objective_gcn, objective_rewiring, objective_iterative_rewiring, train_and_evaluate_gcn
from bridge.datasets import SyntheticGraphDataset
from bridge.sensitivity.run_experiment import run_sensitivity_experiments
from bridge.rewiring import run_bridge_experiment, run_iterative_bridge_experiment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BRIDGE: Graph Rewiring and Sensitivity Analysis')
    
    # Experiment type
    parser.add_argument('--experiment_type', type=str, default='rewiring',
                        choices=['rewiring', 'sensitivity'],
                        help='Type of experiment to run')
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to config file (JSON or YAML)')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num_trials', type=int, default=300, help='Number of optimization trials')
    parser.add_argument('--num_splits', type=int, default=100, help='Number of splits for CI calculation')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment')
    
    # Model settings
    parser.add_argument('--do_hp', action='store_true', help='Use higher-order polynomial filters')
    parser.add_argument('--do_self_loop', action='store_true', help='Add self-loops to graphs')
    parser.add_argument('--do_residual', action='store_true', help='Use residual connections in GCN')
    parser.add_argument('--early_stopping', type=int, default=50, help='Early stopping patience')
    
    # Dataset settings
    parser.add_argument('--dataset_type', type=str, default='standard', 
                        choices=['standard', 'synthetic'], help='Type of dataset to use')
    parser.add_argument('--standard_datasets', nargs='+', default=['cora'],
                        help='List of standard datasets to use')
    
    # Synthetic dataset parameters
    parser.add_argument('--syn_nodes', type=int, default=3000, help='Number of nodes for synthetic dataset')
    parser.add_argument('--syn_classes', type=int, default=4, help='Number of classes for synthetic dataset')
    parser.add_argument('--syn_homophily', type=float, default=0.5, help='Homophily for synthetic dataset')
    parser.add_argument('--syn_degree', type=float, default=20, help='Mean degree for synthetic dataset')
    parser.add_argument('--syn_features', type=int, default=5, help='Number of features for synthetic dataset')
    
    # Optimization parameters
    parser.add_argument('--gcn_h_feats', nargs='+', type=int, default=[16, 32, 64, 128],
                        help='Hidden feature dimensions to try for GCN')
    parser.add_argument('--gcn_n_layers', nargs='+', type=int, default=[1, 2, 3],
                        help='Number of layers to try for GCN')
    parser.add_argument('--gcn_dropout_range', nargs=2, type=float, default=[0.0, 0.7],
                        help='Dropout range for GCN [min, max]')
    parser.add_argument('--lr_gcn_range', nargs=2, type=float, default=[1e-5, 0.1],
                        help='Learning rate range for GCN [min, max]')
    parser.add_argument('--wd_gcn_range', nargs=2, type=float, default=[1e-5, 0.1],
                        help='Weight decay range for GCN [min, max]')
    parser.add_argument('--temperature_range', nargs=2, type=float, default=[1e-5, 2.0],
                        help='Temperature range for softmax [min, max]')
    parser.add_argument('--p_add_range', nargs=2, type=float, default=[0.0, 1.0],
                        help='Probability range for adding edges [min, max]')
    parser.add_argument('--p_remove_range', nargs=2, type=float, default=[0.0, 1.0],
                        help='Probability range for removing edges [min, max]')
    parser.add_argument('--h_feats_selective_options', nargs='+', type=int, default=[16, 32, 64, 128],
                        help='Hidden feature dimensions to try for selective GCN')
    parser.add_argument('--n_layers_selective_options', nargs='+', type=int, default=[1, 2, 3],
                        help='Number of layers to try for selective GCN')
    parser.add_argument('--dropout_selective_range', nargs=2, type=float, default=[0.0, 0.7],
                        help='Dropout range for selective GCN [min, max]')
    parser.add_argument('--lr_selective_range', nargs=2, type=float, default=[1e-5, 0.1],
                        help='Learning rate range for selective GCN [min, max]')
    parser.add_argument('--wd_selective_range', nargs=2, type=float, default=[1e-5, 0.1],
                        help='Weight decay range for selective GCN [min, max]')
    
    # Iterative rewiring parameters
    parser.add_argument('--use_iterative_rewiring', action='store_true', 
                        help='Use iterative rewiring approach instead of single rewiring step')
    parser.add_argument('--n_rewire_iterations_range', nargs=2, type=int, default=[1, 20],
                      help='Range of rewiring iterations to try during optimization [min, max]')
    parser.add_argument('--use_sgc', action='store_true',
                        help='Use SGC for faster rewiring in iterative approach')
    parser.add_argument('--sgc_K_options', type=int, default=[1, 2, 3],
                        help='Number of propagation steps for SGC in iterative rewiring')
    parser.add_argument('--sgc_wd_range', type=int, default=[1e-5, 0.1],
                        help='Number of propagation steps for SGC in iterative rewiring')
    parser.add_argument('--sgc_lr_range', type=int, default=[1e-5, 0.1],
                        help='Number of propagation steps for SGC in iterative rewiring')
    
    # Symmetry checking
    parser.add_argument('--check_symmetry', action='store_false', help='Check and enforce graph symmetry')
    
    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if file_ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif file_ext == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")
    
    return config


def update_args_from_config(args, config):
    """
    Update command-line arguments with values from the config file.
    Command-line arguments take precedence over config file values.
    
    Args:
        args: Command-line arguments
        config: Configuration dictionary
        
    Returns:
        argparse.Namespace: Updated arguments
    """
    args_dict = vars(args)
    
    # Only update args that weren't explicitly set on the command line
    # and exist in the config
    default_args = parse_args()
    default_args_dict = vars(default_args)
    
    for key, value in config.items():
        if key in args_dict:
            # Check if this argument was explicitly provided on the command line
            if args_dict[key] == default_args_dict[key]:  # If it's still the default value
                args_dict[key] = value
    
    return argparse.Namespace(**args_dict)


def run_rewiring_experiment(args):
    """Run the graph rewiring optimization."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = args.experiment_name or input('Name of Experiment: ')
    results_dir = f"results/rewiring/{experiment_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the configuration for reproducibility
    config_dict = vars(args)
    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    if args.check_symmetry:
        sym_dir = f"{results_dir}/sym"
        asym_dir = f"{results_dir}/asym"
        os.makedirs(sym_dir, exist_ok=True)
        os.makedirs(asym_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = {}
    
    # Load datasets
    datasets = []
    
    if args.dataset_type == 'standard':
        for dataset_name in args.standard_datasets:
            try:
                if dataset_name.lower() == 'cora':
                    datasets.append(dgl.data.CoraGraphDataset(force_reload=True))
                elif dataset_name.lower() == 'citeseer':
                    datasets.append(dgl.data.CiteseerGraphDataset(force_reload=True))
                elif dataset_name.lower() == 'pubmed':
                    datasets.append(dgl.data.PubmedGraphDataset(force_reload=True))
                elif dataset_name.lower() == 'actor':
                    datasets.append(dgl.data.ActorDataset(force_reload=True))
                elif dataset_name.lower() == 'chameleon':
                    datasets.append(dgl.data.ChameleonDataset(force_reload=True))
                elif dataset_name.lower() == 'squirrel':
                    datasets.append(dgl.data.SquirrelDataset(force_reload=True))
                elif dataset_name.lower() == 'wisconsin':
                    datasets.append(dgl.data.WisconsinDataset(force_reload=True))
                elif dataset_name.lower() == 'cornell':
                    datasets.append(dgl.data.CornellDataset(force_reload=True))
                elif dataset_name.lower() == 'texas':
                    datasets.append(dgl.data.TexasDataset(force_reload=True))
                elif dataset_name.lower() == 'minesweeper':
                    datasets.append(dgl.data.MinesweeperDataset(force_reload=True))
                elif dataset_name.lower() == 'tolokers':
                    datasets.append(dgl.data.TolokersDataset(force_reload=True))
                else:
                    print(f"Unknown dataset: {dataset_name}")
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
    
    elif args.dataset_type == 'synthetic':
        # Create synthetic datasets
        homophily_values = [args.syn_homophily]
        for h in homophily_values:
            datasets.append(SyntheticGraphDataset(
                n=args.syn_nodes,
                k=args.syn_classes,
                h=h,
                d_mean=args.syn_degree,
                sigma_intra_scalar=0.1,
                sigma_inter_scalar=-0.05,
                tau_scalar=1,
                eta_scalar=1,
                in_feats=args.syn_features
            ))
    
    # Process each dataset
    for dataset in datasets:
        try:
            g = dataset[0]
            #add train val test masks 
            g = add_train_val_test_splits(g, split_ratio=0.6, num_splits=args.num_splits)
            dataset_name = dataset.name.strip('_v2')
            exit_sym_loop = False
            
            while not exit_sym_loop:
                print(f"\n{'='*50}")
                print(f"Processing {dataset_name} Dataset")
                print(f"{'='*50}")
                
                # Move graph to device
                g = g.to(device)
                if args.do_self_loop:
                    g = dgl.remove_self_loop(g)
                    g = dgl.add_self_loop(g)
                
                # Set HP mode based on dataset
                do_hp = args.do_hp
                if do_hp and dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
                    do_hp = False
                    print(f"Disabling HP mode for {dataset_name} (not recommended for this dataset)")
                
                # Print dataset statistics
                print(f"\nDataset Statistics:")
                print(f"Number of nodes: {g.num_nodes()}")
                print(f"Number of edges: {g.num_edges()}")
                print(f"Number of features: {g.ndata['feat'].shape[1]}")
                print(f"Number of classes: {len(torch.unique(g.ndata['label']))}")
                print(f"HP mode: {do_hp}")
                print(f"Self-loops: {args.do_self_loop}")
                print(f"Residual connections: {args.do_residual}")
                print(f"Using iterative rewiring: {args.use_iterative_rewiring}")
                if args.use_iterative_rewiring:
                    print(f"  - Rewiring iterations: {args.n_rewire_iterations_range}")
                    print(f"  - Using SGC: {args.use_sgc}")
                    if args.use_sgc:
                        print(f"  - SGC propagation steps: {args.sgc_K_options}")
                        print(f"  - SGC weight decay: {args.sgc_wd_range}")
                        print(f"  - SGC learning rate: {args.sgc_lr_range}")
                
                # Create dataset-specific study names
                gcn_study_name = f"gcn-{dataset_name}-{timestamp}"
                rewiring_study_name = f"rewiring-{dataset_name}-{timestamp}"
                
                print(f"\n=== Stage 1: Optimizing Base GCN for {dataset_name} ===")
                
                # Create and setup the objective function for GCN optimization
                def gcn_objective(trial):
                    return objective_gcn(
                        trial, 
                        g, 
                        device=device,
                        n_epochs=1000,
                        num_splits=args.num_splits,
                        early_stopping=args.early_stopping,
                        do_hp=do_hp,
                        do_residual_connections=args.do_residual,
                        dataset_name=dataset_name,
                        h_feats_options=args.gcn_h_feats,
                        n_layers_options=args.gcn_n_layers,
                        dropout_range=args.gcn_dropout_range,
                        lr_range=args.lr_gcn_range,
                        weight_decay_range=args.wd_gcn_range,
                    )
                
                # Create and run study for GCN optimization
                gcn_study = optuna.create_study(
                    study_name=gcn_study_name,
                    storage=f"sqlite:///{results_dir}/gcn_study.db",
                    direction='minimize',
                    load_if_exists=True
                )
                
                gcn_study.optimize(gcn_objective, n_trials=args.num_trials)
                
                # Get best GCN parameters
                best_gcn_params = gcn_study.best_params
                print("\nBest GCN parameters:", best_gcn_params)
                print("Best GCN validation accuracy:", -gcn_study.best_value)
                print("Best GCN test accuracy:", gcn_study.best_trial.user_attrs['test_acc'])
                
                print(f"\n=== Stage 2: Optimizing Rewiring & Selective GCN for {dataset_name} ===")
                
                # Get the number of classes
                k = len(torch.unique(g.ndata['label']))
                
                # Generate all possible symmetric permutation matrices
                all_matrices = generate_all_symmetric_permutation_matrices(k)
                
                # Create and setup the objective function for rewiring optimization
                def rewiring_objective(trial):
                    if args.use_iterative_rewiring:
                        return objective_iterative_rewiring(
                            trial, 
                            g, 
                            best_gcn_params, 
                            all_matrices,
                            device=device,
                            n_epochs=1000,
                            num_splits=args.num_splits,
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
                            wd_selective_range=args.wd_selective_range,
                            n_rewire_iterations_range=args.n_rewire_iterations_range,
                            use_sgc=args.use_sgc,
                            sgc_K_options=args.sgc_K_options,
                            sgc_wd_range=args.sgc_wd_range,
                            sgc_lr_range=args.sgc_lr_range
                        )
                    else:
                        return objective_rewiring(
                            trial, 
                            g, 
                            best_gcn_params, 
                            all_matrices,
                            device=device,
                            n_epochs=1000,
                            num_splits=args.num_splits,
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
                
                # Create and run study for rewiring optimization
                rewiring_study = optuna.create_study(
                    study_name=rewiring_study_name,
                    storage=f"sqlite:///{results_dir}/gcn_study.db",
                    direction='minimize',
                    load_if_exists=True
                )
                
                rewiring_study.optimize(rewiring_objective, n_trials=args.num_trials)
                
                # Get best rewiring parameters
                best_rewiring_params = rewiring_study.best_params
                best_rewiring_attributes = rewiring_study.best_trial.user_attrs
                print("\nBest rewiring parameters:", best_rewiring_params)
                print("Best rewiring attributes:", best_rewiring_attributes)
                print("Best rewiring validation accuracy:", -rewiring_study.best_value)
                
                # Apply best rewiring strategy to the graph
                matrix_idx = best_rewiring_params.get('matrix_idx', best_rewiring_attributes.get('matrix_idx'))
                P_k = all_matrices[matrix_idx]
                p_add = best_rewiring_params.get('p_add', best_rewiring_attributes.get('p_add'))
                p_remove = best_rewiring_params.get('p_remove', best_rewiring_attributes.get('p_remove'))
                temperature = best_rewiring_params.get('temperature', best_rewiring_attributes.get('temperature'))
                d_out = best_rewiring_params.get('d_out', best_rewiring_attributes.get('d_out'))
                
                # Select GCN hyperparameters
                h_feats_gcn = best_gcn_params['h_feats']
                n_layers_gcn = best_gcn_params['n_layers']
                dropout_p_gcn = best_gcn_params['dropout_p']
                model_lr_gcn = best_gcn_params['model_lr']
                wd_gcn = best_gcn_params['weight_decay']
                
                # Select selective GCN hyperparameters
                h_feats_sel = best_rewiring_params.get('h_feats_selective', best_rewiring_attributes.get('h_feats_selective'))
                n_layers_sel = best_rewiring_params.get('n_layers_selective', best_rewiring_attributes.get('n_layers_selective'))
                dropout_p_sel = best_rewiring_params.get('dropout_p_selective', best_rewiring_attributes.get('dropout_p_selective'))
                model_lr_sel = best_rewiring_params.get('model_lr_selective', best_rewiring_attributes.get('model_lr_selective'))
                wd_sel = best_rewiring_params.get('weight_decay_selective', best_rewiring_attributes.get('weight_decay_selective'))
                n_rewire_iterations = best_rewiring_params.get('n_rewire_iterations', best_rewiring_attributes.get('n_rewire_iterations'))

                if args.use_iterative_rewiring:
                    sgc_K = best_rewiring_params.get('sgc_K', best_rewiring_attributes.get('sgc_K'))
                    sgc_wd = best_rewiring_params.get('sgc_wd', best_rewiring_attributes.get('sgc_wd'))
                    sgc_lr = best_rewiring_params.get('sgc_lr', best_rewiring_attributes.get('sgc_lr'))
                
                # Run final experiment with best parameters

                # baseline GCN
                print("Running baseline GCN experiment...")
                (mean_train_gcn, mean_val_gcn, mean_test_gcn, train_ci_gcn, val_ci_gcn, test_ci_gcn) = train_and_evaluate_gcn(
                    g = g,
                    h_feats = h_feats_gcn,
                    n_layers = n_layers_gcn,
                    dropout_p = dropout_p_gcn,
                    model_lr = model_lr_gcn,
                    weight_decay = wd_gcn,
                    n_epochs = 1000,
                    early_stopping = args.early_stopping,
                    device = device,
                    num_splits = args.num_splits,
                    log_training = False,
                    do_hp = do_hp,
                    do_self_loop = args.do_self_loop,
                    do_residual_connections = args.do_residual,
                    dataset_name = dataset_name,
                    )

                # rewiring
                if args.use_iterative_rewiring:
                    # Run iterative rewiring experiment
                    print(f"Running iterative rewiring experiment with {n_rewire_iterations} iterations...")
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
                        num_graphs=1,
                        device=device,
                        num_repeats=args.num_splits,
                        n_epochs=1000,
                        early_stopping=args.early_stopping,
                        log_training=False,
                        dataset_name=dataset_name,
                        do_hp=do_hp,
                        do_self_loop=args.do_self_loop,
                        do_residual_connections=args.do_residual,
                        use_sgc=args.use_sgc,
                        n_rewire=n_rewire_iterations,
                        sgc_K=sgc_K,
                        sgc_wd=sgc_wd,
                        sgc_lr=sgc_lr
                    )
                else:
                    # Run standard rewiring experiment
                    print("Running standard rewiring experiment...")
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
                        num_graphs=1,
                        device=device,
                        num_splits=args.num_splits,
                        n_epochs=1000,
                        early_stopping=args.early_stopping,
                        log_training=False,
                        dataset_name=dataset_name,
                        do_hp=do_hp,
                        do_self_loop=args.do_self_loop,
                        do_residual_connections=args.do_residual
                    )
                
                # Calculate improvement
                baseline_test_acc = mean_test_gcn
                final_test_acc = stats_dict['test_acc_mean']
                improvement = (final_test_acc - baseline_test_acc) / baseline_test_acc * 100
                
                # Store results for this dataset
                dataset_results = {
                    'base_gcn': {
                        'params': best_gcn_params,
                        'validation_accuracy': -mean_val_gcn,
                        'test_accuracy': mean_test_gcn,
                        'test_accuracy_ci': (test_ci_gcn[0], test_ci_gcn[1]),
                    },
                    'rewiring': {
                        'params': best_rewiring_params,
                        'validation_accuracy': -rewiring_study.best_value,
                        'test_accuracy_mean': stats_dict['test_acc_mean'],
                        'test_accuracy_ci': stats_dict['test_acc_ci']
                    },
                    'improvement_percentage': improvement,
                    'graph_stats': {
                        'original': stats_dict['original_stats'],
                        'rewired': stats_dict['rewired_stats']
                    },
                    'dataset_stats': {
                        'num_nodes': g.num_nodes(),
                        'num_edges': g.num_edges(),
                        'num_features': g.ndata['feat'].shape[1],
                        'num_classes': len(torch.unique(g.ndata['label']))
                    },
                    'iterative_rewiring': args.use_iterative_rewiring,
                }
                
                # Add additional iterative rewiring info if used
                if args.use_iterative_rewiring:
                    # Extract rewiring history from the first result
                    if results_list and 'rewiring_history' in results_list[0]:
                        dataset_results['rewiring_history'] = results_list[0]['rewiring_history']
                
                all_results[dataset_name] = dataset_results
                
                # Save individual dataset results
                if args.check_symmetry:
                    if check_symmetry(g):
                        with open(f"{sym_dir}/{dataset_name}_results.json", 'w') as f:
                            json.dump(dataset_results, f, indent=2)
                    else:
                        with open(f"{asym_dir}/{dataset_name}_results.json", 'w') as f:
                            json.dump(dataset_results, f, indent=2)
                else:
                    with open(f"{results_dir}/{dataset_name}_results.json", 'w') as f:
                        json.dump(dataset_results, f, indent=2)
                
                print(f"\nResults for {dataset_name}:")
                print(f"Base GCN Test Accuracy: {baseline_test_acc:.4f}")
                print(f"Final Test Accuracy: {final_test_acc:.4f} ± "
                      f"{stats_dict['test_acc_ci'][1] - final_test_acc:.4f}")
                print(f"Improvement: {improvement:.2f}%")
                
                if not args.check_symmetry or check_symmetry(g):
                    exit_sym_loop = True
                else:
                    # Symmetrize the graph and continue with symmetric version
                    g = dgl.to_bidirected(g.cpu(), copy_ndata=True).to(device)
                    dataset_name = f'{dataset_name}_sym'
        
        except Exception as e:
            print(f"\nError processing {dataset_name}: {str(e)}")
            # Log the error
            with open(f"{results_dir}/error_log.txt", 'a') as f:
                f.write(f"Error processing {dataset_name}: {str(e)}\n")
            all_results[dataset_name] = {'error': str(e)}
    
    # Save complete results
    with open(f"{results_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary table
    summary_data = []
    for dataset_name, results in all_results.items():
        if 'error' not in results:
            summary_data.append({
                'Dataset': dataset_name,
                'Base GCN Accuracy': results['base_gcn']['test_accuracy'],
                'Final Accuracy': results['rewiring']['test_accuracy_mean'],
                'CI (±)': results['rewiring']['test_accuracy_ci'][1] - results['rewiring']['test_accuracy_mean'],
                'Improvement (%)': results['improvement_percentage'],
                'Nodes': results['dataset_stats']['num_nodes'],
                'Edges': results['dataset_stats']['num_edges'],
                'Classes': results['dataset_stats']['num_classes'],
                'Iterative': "Yes" if results.get('iterative_rewiring', False) else "No"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{results_dir}/summary.csv", index=False)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    # Print average improvement across all datasets
    if not summary_df.empty and 'Improvement (%)' in summary_df.columns:
        mean_improvement = summary_df['Improvement (%)'].mean()
        std_improvement = summary_df['Improvement (%)'].std()
        print(f"\nAverage improvement across all datasets: {mean_improvement:.2f}% ± {std_improvement:.2f}%")
    else:
        print("\nNo valid results to compute average improvement.")


def main():
    """Main entry point for the BRIDGE toolkit."""
    args = parse_args()
    
    # Load configuration from file if provided
    if args.config:
        config = load_config(args.config)
        args = update_args_from_config(args, config)
    
    # Run appropriate experiment type
    if args.experiment_type == 'rewiring':
        run_rewiring_experiment(args)
    elif args.experiment_type == 'sensitivity':
        if args.config:
            results_dir = run_sensitivity_experiments(args.config)
            print(f"Sensitivity analysis completed. Results saved to {results_dir}")
        else:
            print("Error: Config file is required for sensitivity analysis experiments.")
            print("Please provide a configuration file using the --config argument.")
    else:
        print(f"Error: Unknown experiment type '{args.experiment_type}'")
        print("Valid options are: 'rewiring', 'sensitivity'")


if __name__ == "__main__":
    main()