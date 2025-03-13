"""
Experiment runner for sensitivity analysis in graph neural networks.

This module provides functions for running comprehensive sensitivity analysis
experiments on synthetic graphs with controlled properties. It builds on the
theoretical SNR framework to evaluate when MPNNs outperform non-relational models.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import datetime
from pathlib import Path
from tqdm import tqdm
import dgl

# Import from BRIDGE package
from bridge.sensitivity import (
    LinearGCN, TwoLayerGCN,
    create_feature_generator,
    run_sensitivity_experiment,
    run_multi_graph_experiment,
    plot_snr_vs_homophily
)
from bridge.datasets.synthetic import SyntheticGraphDataset


def save_config(config, save_path):
    """Save configuration to a file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.endswith('.json'):
            json.dump(config, f, indent=4)
        elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {save_path}")


def load_config(config_path):
    """Load configuration from a JSON or YAML file."""
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


def setup_dataset(config):
    """Create synthetic dataset based on configuration parameters."""
    n = config.get('num_nodes', 3000)
    k = config.get('num_classes', 4)
    h = config.get('homophily', 0.5)
    d_mean = config.get('mean_degree', 20)
    in_feats = config.get('feature_dim', 5)
    sigma_intra_scalar = config.get('intra_class_cov', 0.1)
    sigma_inter_scalar = config.get('inter_class_cov', -0.05)
    tau_scalar = config.get('global_cov', 1.0)
    eta_scalar = config.get('noise_cov', 1.0)
    
    # Create dataset
    dataset = SyntheticGraphDataset(
        n=n,
        k=k,
        h=h,
        d_mean=d_mean,
        sigma_intra_scalar=sigma_intra_scalar,
        sigma_inter_scalar=sigma_inter_scalar,
        tau_scalar=tau_scalar,
        eta_scalar=eta_scalar,
        in_feats=in_feats
    )
    
    return dataset[0], in_feats


def setup_feature_generator(config):
    """Create feature generator from configuration."""
    in_feats = config.get('feature_dim', 5)
    scale = config.get('cov_scale', 1e-4)
    
    sigma_intra_scalar = config.get('intra_class_cov', 0.1)
    sigma_inter_scalar = config.get('inter_class_cov', -0.05)
    tau_scalar = config.get('global_cov', 1.0)
    eta_scalar = config.get('noise_cov', 1.0)
    
    # Define feature covariance matrices
    dtype = torch.float64
    sigma_intra = torch.eye(in_feats, dtype=dtype) * sigma_intra_scalar * scale
    sigma_inter = torch.eye(in_feats, dtype=dtype) * sigma_inter_scalar * scale
    tau = torch.eye(in_feats, dtype=dtype) * tau_scalar * scale
    eta = torch.eye(in_feats, dtype=dtype) * eta_scalar * scale
    
    # Create feature generator
    return create_feature_generator(sigma_intra, sigma_inter, tau, eta)


def create_model(model_type, in_feats, num_classes, hidden_dim):
    """Create a model based on specified type."""
    if model_type.lower() == 'linear_gcn':
        return LinearGCN(in_feats, hidden_dim, num_classes).double()
    elif model_type.lower() == 'two_layer_gcn':
        return TwoLayerGCN(in_feats, hidden_dim, num_classes).double()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_homophily_sweep_experiment(config, results_dir):
    """Run experiment with varying homophily values."""
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup parameters
    homophily_values = np.linspace(
        config.get('homophily_min', 0.1),
        config.get('homophily_max', 0.9),
        config.get('homophily_steps', 20)
    )
    in_feats = config.get('feature_dim', 5)
    num_nodes = config.get('num_nodes', 500)
    num_classes = config.get('num_classes', 2)
    mean_degree = config.get('mean_degree', 20)
    num_samples = config.get('num_samples', 5)
    hidden_dim = config.get('hidden_dim', 16)
    model_type = config.get('model_type', 'linear_gcn')
    
    # Setup feature generator
    feature_generator = setup_feature_generator(config)
    
    # Function to create graph dataset
    def graph_generator(num_nodes, num_classes, homophily, mean_degree):
        dataset = SyntheticGraphDataset(
            n=num_nodes,
            k=num_classes,
            h=homophily,
            d_mean=mean_degree,
            sigma_intra_scalar=config.get('intra_class_cov', 0.1),
            sigma_inter_scalar=config.get('inter_class_cov', -0.05),
            tau_scalar=config.get('global_cov', 1.0),
            eta_scalar=config.get('noise_cov', 1.0),
            in_feats=in_feats
        )
        return dataset[0]
    
    # Model constructor function
    def model_constructor(in_feats, num_classes):
        return create_model(model_type, in_feats, num_classes, hidden_dim)
    
    print(f"Running homophily sweep experiment with {len(homophily_values)} values...")
    
    # Run the experiment
    multi_results = run_multi_graph_experiment(
        graph_generator=graph_generator,
        model_constructor=model_constructor,
        feature_generator=feature_generator,
        in_feats=in_feats,
        num_nodes=num_nodes,
        num_classes=num_classes,
        homophily_values=homophily_values,
        mean_degree=mean_degree,
        num_samples=num_samples,
        num_acc_repeats=config.get('num_acc_repeats', 1),
        num_monte_carlo_samples=config.get('num_monte_carlo_samples', 100),
        lr=config.get('learning_rate', 0.01),
        weight_decay=config.get('weight_decay', 1e-3),
        num_epochs=config.get('num_epochs', 200),
        sigma_intra=torch.eye(in_feats, dtype=torch.float64) * config.get('intra_class_cov', 0.1) * config.get('cov_scale', 1e-4),
        sigma_inter=torch.eye(in_feats, dtype=torch.float64) * config.get('inter_class_cov', -0.05) * config.get('cov_scale', 1e-4),
        tau=torch.eye(in_feats, dtype=torch.float64) * config.get('global_cov', 1.0) * config.get('cov_scale', 1e-4),
        eta=torch.eye(in_feats, dtype=torch.float64) * config.get('noise_cov', 1.0) * config.get('cov_scale', 1e-4)
    )
    
    # Extract results for plotting
    homophily_list = np.array([h[0] for h in multi_results["homophily_list"]])
    snr_mc_means = np.array([snr[0] for snr in multi_results["estimated_snr_mc_list"]])
    snr_mc_stds = np.array([snr[1] for snr in multi_results["estimated_snr_mc_list"]])
    snr_theorem_means = np.array([snr[0] for snr in multi_results["estimated_snr_theorem_val_list"]])
    snr_theorem_stds = np.array([snr[1] for snr in multi_results["estimated_snr_theorem_val_list"]])
    acc_means = np.array([acc[0] for acc in multi_results["acc_list"]])
    acc_stds = np.array([acc[1] for acc in multi_results["acc_list"]])
    
    # Save results
    results_data = {
        'homophily_values': homophily_values.tolist(),
        'homophily_measured': homophily_list.tolist(),
        'snr_mc_means': snr_mc_means.tolist(),
        'snr_mc_stds': snr_mc_stds.tolist(),
        'snr_theorem_means': snr_theorem_means.tolist(),
        'snr_theorem_stds': snr_theorem_stds.tolist(),
        'acc_means': acc_means.tolist(),
        'acc_stds': acc_stds.tolist()
    }
    
    with open(os.path.join(results_dir, 'homophily_sweep_results.json'), 'w') as f:
        json.dump(results_data, f, indent=4)
    
    # Get FNN baseline (non-relational)
    # Run a single experiment with a graph but don't use graph structure
    print("Computing FNN baseline...")
    g_baseline = graph_generator(num_nodes, num_classes, 0.5, mean_degree)
    g_no_edge = dgl.graph(([], []), num_nodes=num_nodes)
    g_no_edge.ndata['label'] = g_baseline.ndata['label']
    
    model = model_constructor(in_feats=in_feats, num_classes=num_classes)
    
    fnn_results = run_sensitivity_experiment(
        model=model,
        graph=g_no_edge,  # Graph with no edges
        feature_generator=feature_generator,
        in_feats=in_feats,
        num_acc_repeats=config.get('num_acc_repeats', 1),
        num_monte_carlo_samples=config.get('num_monte_carlo_samples', 100),
        lr=config.get('learning_rate', 0.01),
        weight_decay=config.get('weight_decay', 1e-3),
        num_epochs=config.get('num_epochs', 200),
        sigma_intra=torch.eye(in_feats, dtype=torch.float64) * config.get('intra_class_cov', 0.1) * config.get('cov_scale', 1e-4),
        sigma_inter=torch.eye(in_feats, dtype=torch.float64) * config.get('inter_class_cov', -0.05) * config.get('cov_scale', 1e-4),
        tau=torch.eye(in_feats, dtype=torch.float64) * config.get('global_cov', 1.0) * config.get('cov_scale', 1e-4),
        eta=torch.eye(in_feats, dtype=torch.float64) * config.get('noise_cov', 1.0) * config.get('cov_scale', 1e-4)
    )
    
    fnn_acc_mean = fnn_results["mean_test_acc"]
    fnn_acc_std = 0.0  # Approximate from multiple runs
    
    # Visualize results
    print("Generating plots...")
    fig = plot_snr_vs_homophily(
        homophily_values=homophily_list,
        snr_mc_means=snr_mc_means,
        snr_mc_stds=snr_mc_stds,
        snr_theorem_means=snr_theorem_means,
        snr_theorem_stds=snr_theorem_stds,
        accuracy_means=acc_means,
        accuracy_stds=acc_stds,
        fnn_acc_mean=fnn_acc_mean,
        fnn_acc_std=fnn_acc_std,
        title=f"SNR and Test Accuracy vs Edge Homophily ({model_type})",
        save_path=os.path.join(results_dir, 'snr_vs_homophily.png')
    )
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, 'snr_vs_homophily.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Experiment completed. Results saved to {results_dir}")


def run_single_graph_experiment(config, results_dir):
    """Run experiment on a single graph with specified properties."""
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup parameters
    in_feats = config.get('feature_dim', 5)
    hidden_dim = config.get('hidden_dim', 16)
    model_type = config.get('model_type', 'linear_gcn')
    
    # Create graph
    g, in_feats = setup_dataset(config)
    
    # Setup feature generator
    feature_generator = setup_feature_generator(config)
    
    # Create model
    model = create_model(model_type, in_feats, len(torch.unique(g.ndata['label'])), hidden_dim)
    
    print(f"Running single graph experiment with homophily={config.get('homophily', 0.5)}...")
    
    # Run the experiment
    results = run_sensitivity_experiment(
        model=model,
        graph=g,
        feature_generator=feature_generator,
        in_feats=in_feats,
        num_acc_repeats=config.get('num_acc_repeats', 10),
        num_monte_carlo_samples=config.get('num_monte_carlo_samples', 50),
        sigma_intra=torch.eye(in_feats, dtype=torch.float64) * config.get('intra_class_cov', 0.1) * config.get('cov_scale', 1e-4),
        sigma_inter=torch.eye(in_feats, dtype=torch.float64) * config.get('inter_class_cov', -0.05) * config.get('cov_scale', 1e-4),
        tau=torch.eye(in_feats, dtype=torch.float64) * config.get('global_cov', 1.0) * config.get('cov_scale', 1e-4),
        eta=torch.eye(in_feats, dtype=torch.float64) * config.get('noise_cov', 1.0) * config.get('cov_scale', 1e-4)
    )
    
    # Save results
    results_data = {
        'homophily': config.get('homophily', 0.5),
        'monte_carlo_snr': results['estimated_snr_mc'].item() if isinstance(results['estimated_snr_mc'], torch.Tensor) else results['estimated_snr_mc'],
        'theorem_snr': results['estimated_snr_theorem'].item() if isinstance(results['estimated_snr_theorem'], torch.Tensor) else results['estimated_snr_theorem'],
        'test_accuracy': results['mean_test_acc'],
        'test_loss': results['mean_test_loss']
    }
    
    with open(os.path.join(results_dir, 'single_graph_results.json'), 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"Experiment completed. Results saved to {results_dir}")


def run_sensitivity_experiments(config_path):
    """Run the sensitivity analysis experiments based on configuration."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup results directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = config.get('experiment_name', 'sensitivity_analysis')
    results_dir = os.path.join('results', 'sensitivity', f"{experiment_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config for reproducibility
    save_config(config, os.path.join(results_dir, 'config.json'))
    
    run_homophily_sweep_experiment(config, results_dir)
    return results_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sensitivity analysis experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    
    args = parser.parse_args()
    
    run_sensitivity_experiments(args.config)