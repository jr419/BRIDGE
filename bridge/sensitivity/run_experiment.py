# bridge/sensitivity/run_experiment.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import yaml
import datetime
from pathlib import Path
from tqdm import tqdm
import dgl
from scipy import stats # For statistical tests

# Import from BRIDGE package
from bridge.sensitivity import (
    create_feature_generator,
    estimate_snr_monte_carlo,
    estimate_snr_from_sensitivities,
    estimate_sensitivity_autograd, # For S, N, T
    create_sensitivity_model,
    create_fnn_baseline_model,
    normalize_backbone_type,
    # New/modified utility functions will be imported from .utils
    # New visualization functions will be imported from .visualization
)
from bridge.datasets.synthetic import SyntheticGraphDataset
from bridge.utils.homophily import class_bottlenecking_score as higher_order_homophily # Renaming for clarity
from bridge.utils.graph_utils import set_seed
from bridge.utils.matrix_utils import compute_confidence_interval

# Import or define new utility functions here or in bridge.sensitivity.utils
# For example:
# from .utils import (
# calculate_sensitivity_condition,
# calculate_node_level_snr,
# calculate_bottlenecking_score,
# train_model, # Assuming a shared training utility
# evaluate_model, # Assuming a shared evaluation utility
# node_level_evaluate
# )
# from .visualization import (
# plot_local_sensitivity_validation,
# plot_snr_ratio_analysis,
# plot_bottlenecking_snr_scatter,
# plot_graph_wide_snr_validation, # Modify existing plot_snr_vs_homophily
# plot_snr_accuracy_correlation
# )

# --- Helper functions from bridge.sensitivity.utils ---
# (These would ideally be in bridge.sensitivity.utils.py)

def train_model_sensitivity(
    model: torch.nn.Module,
    graph: dgl.DGLGraph, # May not be used by FNN
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor = None,
    n_epochs: int = 100, # Paper uses 100 epochs for this part
    lr: float = 0.01,
    weight_decay: float = 1e-3, # Default from paper's GCN setup
    device: str = "cpu",
    patience: int = 1000,  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement threshold
) -> torch.nn.Module:
    """Trains a model for sensitivity analysis experiments with early stopping."""
    model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    
    if val_mask is not None:
        val_mask = val_mask.to(device)
        use_early_stopping = True
    else:
        use_early_stopping = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(n_epochs):
        # Training step
        optimizer.zero_grad()
        logits = model(graph, features)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation step for early stopping
        if use_early_stopping:
            model.eval()
            with torch.no_grad():
                val_logits = model(graph, features)
                val_loss = criterion(val_logits[val_mask], labels[val_mask]).item()
            
            # Check for improvement
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                #print(f"Early stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.4f})")
                break
            
            model.train()  # Switch back to training mode
    
    # Restore best model if early stopping was used
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

import torch.nn.functional as F

def evaluate_node_CE(
    model: torch.nn.Module,
    graph: dgl.DGLGraph, # May not be used by FNN
    features: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu"
) -> tuple[float, torch.Tensor]:
    """Evaluates a model, returning the cross-entropy loss and node-level cross-entropy."""
    model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        
        
        predictions_all = logits.argmax(dim=1)
        node_acc = (predictions_all==labels)
        node_losses = -F.cross_entropy(logits, labels, reduction='none')
        CE_loss = node_losses[mask].mean().item()  # Mean loss over the masked nodes
        
        masked_predictions = predictions_all[mask]
        masked_labels = labels[mask]
        accuracy = (masked_predictions == masked_labels).float().mean().item()
        
    return accuracy, CE_loss, node_acc, node_losses

def calculate_sensitivity_condition_check(S_sens, N_sens, T_sens, rho):
    """Checks the sensitivity condition: S > rho*N + (1-rho)*T for each node and output feature."""
    # Assuming sensitivities are [num_nodes, out_feats, in_feats, in_feats]
    # We need to sum over in_feats (q) for S_i,p,q,q (Eq. 12) [cite: 112]
    S_diag_sum = torch.diagonal(S_sens, dim1=-2, dim2=-1).sum(dim=-1) # Shape: [num_nodes, out_feats]
    N_diag_sum = torch.diagonal(N_sens, dim1=-2, dim2=-1).sum(dim=-1) # Shape: [num_nodes, out_feats]
    T_diag_sum = torch.diagonal(T_sens, dim1=-2, dim2=-1).sum(dim=-1) # Shape: [num_nodes, out_feats]
    
    condition_met = S_diag_sum > (rho * N_diag_sum + (1 - rho) * T_diag_sum)
    return condition_met # Shape: [num_nodes, out_feats]

def calculate_local_bottleneck_score(graph, p_order, labels, device):
    """Calculates h_i^{l,l} as per Eq. 15"""
    bottleneck_scores = higher_order_homophily(
        p=p_order,
        g=graph,
        self_loops=False,
        fix_d=True,
        sym=False,
        device=device
    )
    return bottleneck_scores


def resolve_backbone_specs(model_params):
    """Normalize configured backbone specs while preserving legacy n_layers configs."""
    default_hidden = model_params.get('hidden_feats', 64)
    default_layers = model_params.get('n_layers', 1)
    default_alpha = model_params.get('alpha', 0.1)
    default_lambda = model_params.get('lambda', model_params.get('lambda_', 0.5))
    default_bias = model_params.get('bias', True)

    raw_backbones = model_params.get('backbones')
    if not raw_backbones:
        raw_backbones = [{'name': 'vanilla_gcn', 'type': 'vanilla_gcn'}]

    specs = []
    for raw_spec in raw_backbones:
        if isinstance(raw_spec, str):
            raw_spec = {'name': raw_spec, 'type': raw_spec}

        backbone_type = normalize_backbone_type(raw_spec.get('type', raw_spec.get('name', 'vanilla_gcn')))
        name = raw_spec.get('name', backbone_type)
        n_layers = raw_spec.get('n_layers', default_layers)
        hidden_feats = raw_spec.get('hidden_feats', default_hidden)
        alpha = raw_spec.get('alpha', default_alpha)
        lambda_value = raw_spec.get('lambda', raw_spec.get('lambda_', default_lambda))
        bias = raw_spec.get('bias', default_bias)

        specs.append({
            'name': name,
            'type': backbone_type,
            'n_layers': int(n_layers),
            'hidden_feats': int(hidden_feats),
            'alpha': float(alpha),
            'lambda': float(lambda_value),
            'bias': bool(bias),
        })
    return specs


def backbone_result_fields(backbone_spec):
    """Metadata columns written to node- and graph-level result rows."""
    return {
        'backbone': backbone_spec['name'],
        'backbone_type': backbone_spec['type'],
        'alpha': backbone_spec['alpha'] if backbone_spec['type'] in {'initial_residual_gcn', 'gcnii'} else None,
        'lambda': backbone_spec['lambda'] if backbone_spec['type'] == 'gcnii' else None,
        'n_layers': backbone_spec['n_layers'],
        'hidden_feats': backbone_spec['hidden_feats'],
    }


def run_full_sensitivity_experiment(config_path):
    """
    Main function to run the comprehensive sensitivity analysis.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Experiment setup from config
    n_nodes = config['sbm_params']['n_nodes']
    mean_degree = config['sbm_params']['mean_degree']
    in_feats = config['feature_params']['dim']
    num_classes = config['sbm_params']['num_classes'] # Assuming k classes for SBM

    homophily_values = np.linspace(
        config['sbm_params']['h_min'],
        config['sbm_params']['h_max'],
        config['sbm_params']['h_steps']
    )

    num_total_runs = config['experiment_params']['num_total_runs'] # 100 independent runs
    num_mc_simulations = config['experiment_params']['num_mc_simulations'] # 300
    num_mc_inner_samples = config['experiment_params']['num_mc_inner_samples'] # 300
    n_epochs = config['model_params']['n_epochs'] # 100
    lr = config['model_params']['lr']
    wd = config['model_params']['weight_decay']
    rho = config['experiment_params']['rho'] # local noise ratio, Psi_qq / (Phi_qq + Psi_qq)

    # Check if single_graph mode is enabled
    single_graph_mode = config.get('single_graph', False)

    # Covariance matrices for feature generation and theoretical SNR
    # These should be created based on config['feature_params']
    sigma_intra_val = config['feature_params']['sigma_intra']
    sigma_inter_val = config['feature_params']['sigma_inter']
    phi_val = config['feature_params']['phi_global'] # Global shift covariance
    psi_val = config['feature_params']['psi_noise'] # Node noise covariance
    
    # Create diagonal covariance matrices as per paper's setup for IID features [cite: 113, 304]
    sigma_intra = torch.eye(in_feats, device=device) * sigma_intra_val
    sigma_inter = torch.eye(in_feats, device=device) * sigma_inter_val
    phi_cov = torch.eye(in_feats, device=device) * phi_val
    psi_cov = torch.eye(in_feats, device=device) * psi_val

    feature_generator = create_feature_generator(sigma_intra.cpu(), sigma_inter.cpu(), phi_cov.cpu(), psi_cov.cpu(), dtype=torch.double)
    
    backbone_specs = resolve_backbone_specs(config['model_params'])

    print(f"DEBUG: Config loaded - n_layers: {config['model_params'].get('n_layers', 'NOT_FOUND')}")
    print(f"DEBUG: Using backbones: {[spec['name'] for spec in backbone_specs]}")
    print(f"DEBUG: Single graph mode: {config.get('single_graph', False)}")

    node_results_list = []
    graph_results_list = []
    stored_graphs = {}  # Store graphs for single_graph mode

    for h_idx, h in enumerate(tqdm(homophily_values, desc="Homophily Sweep")):
        # Generate single graph for this homophily if in single_graph mode
        if single_graph_mode:
            sbm_dataset = SyntheticGraphDataset(
                n=n_nodes,
                k=num_classes, # k=2 for 2-block SBM
                h=h,
                d_mean=mean_degree,
                sigma_intra_scalar=sigma_intra_val, # These are for feature gen within dataset,
                sigma_inter_scalar=sigma_inter_val, # ensure they align or use external generator
                tau_scalar=phi_val,
                eta_scalar=psi_val,
                in_feats=in_feats,
                sym=True
            )
            shared_graph = sbm_dataset[0].to(device)
            shared_graph.ndata['train_mask'] = shared_graph.ndata['train_mask'][:,0] if shared_graph.ndata['train_mask'].ndim > 1 else shared_graph.ndata['train_mask']
            shared_graph.ndata['val_mask'] = shared_graph.ndata['val_mask'][:,0] if shared_graph.ndata['val_mask'].ndim > 1 else shared_graph.ndata['val_mask']
            shared_graph.ndata['test_mask'] = shared_graph.ndata['test_mask'][:,0] if shared_graph.ndata['test_mask'].ndim > 1 else shared_graph.ndata['test_mask']
            
            shared_labels = shared_graph.ndata['label'].to(device)
            # Store the graph structure and labels for saving later
            stored_graphs[f'h_{h:.3f}'] = {
                'graph': shared_graph.cpu(),
                'labels': shared_labels.cpu(),
                'homophily': h,
                'n_nodes': n_nodes,
                'mean_degree': mean_degree,
                'num_classes': num_classes
            }

        for run_idx in range(num_total_runs):
            # 1. Generate Graph (2-block planted partition SBM)
            if single_graph_mode:
                # Use the shared graph for all runs at this homophily
                graph = shared_graph
                labels = shared_labels
            else:
                # Generate new graph for each run (original behavior)
                sbm_dataset = SyntheticGraphDataset(
                    n=n_nodes,
                    k=num_classes, # k=2 for 2-block SBM
                    h=h,
                    d_mean=mean_degree,
                    sigma_intra_scalar=sigma_intra_val, # These are for feature gen within dataset,
                    sigma_inter_scalar=sigma_inter_val, # ensure they align or use external generator
                    tau_scalar=phi_val,
                    eta_scalar=psi_val,
                    in_feats=in_feats,
                    sym=True
                )
                graph = sbm_dataset[0].to(device)
                graph.ndata['train_mask'] = graph.ndata['train_mask'][:,0] if graph.ndata['train_mask'].ndim > 1 else graph.ndata['train_mask']
                graph.ndata['val_mask'] = graph.ndata['val_mask'][:,0] if graph.ndata['val_mask'].ndim > 1 else graph.ndata['val_mask']
                graph.ndata['test_mask'] = graph.ndata['test_mask'][:,0] if graph.ndata['test_mask'].ndim > 1 else graph.ndata['test_mask']
                
                labels = graph.ndata['label'].to(device)

            # Generate features (always new for each run)
            num_graph_nodes = graph.num_nodes()
            features = feature_generator(num_graph_nodes, in_feats, labels.cpu(), num_mu_samples=1)[:,:,0].to(device)
            graph.ndata['feat'] = features

            edgeless_graph = dgl.graph(([], []), num_nodes=num_graph_nodes).to(device)
            edgeless_graph.ndata['label'] = labels
            bottleneck_cache = {}

            for backbone_spec in backbone_specs:
                result_fields = backbone_result_fields(backbone_spec)

                gcn_model = create_sensitivity_model(
                    backbone_spec['type'],
                    in_feats,
                    backbone_spec['hidden_feats'],
                    num_classes,
                    n_layers=backbone_spec['n_layers'],
                    alpha=backbone_spec['alpha'],
                    lambda_=backbone_spec['lambda'],
                    bias=backbone_spec['bias'],
                ).double().to(device)
                fnn_model = create_fnn_baseline_model(
                    in_feats,
                    backbone_spec['hidden_feats'],
                    num_classes,
                    n_layers=backbone_spec['n_layers'],
                    bias=backbone_spec['bias'],
                ).double().to(device)

                gcn_model = train_model_sensitivity(
                    gcn_model, graph, features, labels,
                    graph.ndata['train_mask'], graph.ndata['val_mask'],
                    n_epochs, lr, wd, device
                )
                fnn_model = train_model_sensitivity(
                    fnn_model, edgeless_graph, features, labels,
                    graph.ndata['train_mask'], graph.ndata['val_mask'],
                    n_epochs, lr, wd, device
                )

                gcn_test_acc, gcn_test_loss, gcn_node_acc, gcn_node_ce = evaluate_node_CE(
                    gcn_model, graph, features, labels, graph.ndata['test_mask'], device
                )
                fnn_test_acc, fnn_test_loss, fnn_node_acc, fnn_node_ce = evaluate_node_CE(
                    fnn_model, edgeless_graph, features, labels, graph.ndata['test_mask'], device
                )

                gcn_model.eval()
                S_sens_gcn = estimate_sensitivity_autograd(gcn_model, graph, in_feats, labels, "signal", device=device)
                N_sens_gcn = estimate_sensitivity_autograd(gcn_model, graph, in_feats, labels, "noise", device=device)
                T_sens_gcn = estimate_sensitivity_autograd(gcn_model, graph, in_feats, labels, "global", device=device)

                gcn_snr_theorem = estimate_snr_from_sensitivities(
                    S_sens_gcn, N_sens_gcn, T_sens_gcn,
                    sigma_intra, sigma_inter, phi_cov, psi_cov,
                    device=device
                )
                gcn_snr_mc = estimate_snr_monte_carlo(
                    gcn_model, graph, in_feats, labels.cpu(),
                    num_mc_simulations, feature_generator, device, num_mc_inner_samples
                )

                S_sens_fnn = estimate_sensitivity_autograd(fnn_model, edgeless_graph, in_feats, labels, "signal", device=device)
                N_sens_fnn = estimate_sensitivity_autograd(fnn_model, edgeless_graph, in_feats, labels, "noise", device=device)
                T_sens_fnn = estimate_sensitivity_autograd(fnn_model, edgeless_graph, in_feats, labels, "global", device=device)

                fnn_snr_theorem = estimate_snr_from_sensitivities(
                    S_sens_fnn, N_sens_fnn, T_sens_fnn,
                    sigma_intra, sigma_inter, phi_cov, psi_cov,
                    device=device
                )
                fnn_snr_mc = estimate_snr_monte_carlo(
                    fnn_model, edgeless_graph, in_feats, labels.cpu(),
                    num_mc_simulations, feature_generator, device, num_mc_inner_samples
                )

                sensitivity_condition_satisfied = calculate_sensitivity_condition_check(
                    S_sens_gcn, N_sens_gcn, T_sens_gcn, rho
                )

                p_order = backbone_spec['n_layers']
                if p_order not in bottleneck_cache:
                    bottleneck_cache[p_order] = calculate_local_bottleneck_score(graph, p_order, labels, device)
                local_bottleneck_scores = bottleneck_cache[p_order]
                graph_level_ho_homophily = local_bottleneck_scores.mean()

                for node_idx in range(num_graph_nodes):
                    node_data = {
                        **result_fields,
                        'homophily_h': h,
                        'run_idx': run_idx,
                        'node_idx': node_idx,
                        'label': labels[node_idx].item(),
                        'degree': graph.in_degrees()[node_idx].item(),
                        'is_graph_level': False,
                        'single_graph_mode': single_graph_mode,
                        'gcn_S_sensitivity_trace': torch.diagonal(S_sens_gcn[node_idx], dim1=-2, dim2=-1).sum().item(),
                        'gcn_N_sensitivity_trace': torch.diagonal(N_sens_gcn[node_idx], dim1=-2, dim2=-1).sum().item(),
                        'gcn_T_sensitivity_trace': torch.diagonal(T_sens_gcn[node_idx], dim1=-2, dim2=-1).sum().item(),
                        'gcn_sensitivity_condition_met': sensitivity_condition_satisfied[node_idx].any().item(),
                        'gcn_snr_theorem_node': gcn_snr_theorem[node_idx].mean().item(),
                        'gcn_snr_mc_node': gcn_snr_mc[node_idx].mean().item(),
                        'fnn_snr_theorem_node': fnn_snr_theorem[node_idx].mean().item(),
                        'fnn_snr_mc_node': fnn_snr_mc[node_idx].mean().item(),
                        'gcn_ce_node': gcn_node_ce[node_idx].item(),
                        'fnn_ce_node': fnn_node_ce[node_idx].item(),
                        'gcn_accuracy_node': gcn_node_acc[node_idx].item(),
                        'fnn_accuracy_node': fnn_node_acc[node_idx].item(),
                        'gcn_bottleneck_score_node': local_bottleneck_scores[node_idx].item(),
                    }
                    node_results_list.append(node_data)

                graph_data_entry = {
                    **result_fields,
                    'homophily_h': h,
                    'run_idx': run_idx,
                    'is_graph_level': True,
                    'single_graph_mode': single_graph_mode,
                    'gcn_avg_S_sensitivity': torch.diagonal(S_sens_gcn, dim1=-2, dim2=-1).sum(dim=(-1,-2)).mean().item(),
                    'gcn_avg_N_sensitivity': torch.diagonal(N_sens_gcn, dim1=-2, dim2=-1).sum(dim=(-1,-2)).mean().item(),
                    'gcn_avg_T_sensitivity': torch.diagonal(T_sens_gcn, dim1=-2, dim2=-1).sum(dim=(-1,-2)).mean().item(),
                    'gcn_avg_snr_theorem': gcn_snr_theorem.mean().item(),
                    'gcn_avg_snr_mc': gcn_snr_mc.mean().item(),
                    'fnn_avg_snr_theorem': fnn_snr_theorem.mean().item(),
                    'fnn_avg_snr_mc': fnn_snr_mc.mean().item(),
                    'gcn_test_loss_graph': gcn_test_loss,
                    'fnn_test_loss_graph': fnn_test_loss,
                    'gcn_test_accuracy_graph': gcn_test_acc,
                    'fnn_test_accuracy_graph': fnn_test_acc,
                    'gcn_higher_order_homophily_graph': graph_level_ho_homophily.item()
                }
                graph_results_list.append(graph_data_entry)

    #save as json
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path(config.get('results_dir', 'results'))
    results_dir = results_base / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save stored graphs if in single_graph mode
    if single_graph_mode:
        graphs_dir = results_dir / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        
        for graph_key, graph_data in stored_graphs.items():
            graph_path = graphs_dir / f"graph_{graph_key}_{timestamp}.pt"
            torch.save(graph_data, graph_path)
        
        # Save graph metadata
        graph_metadata = {
            'single_graph_mode': True,
            'num_graphs': len(stored_graphs),
            'homophily_values': [stored_graphs[key]['homophily'] for key in stored_graphs.keys()],
            'graph_files': [f"graph_{key}_{timestamp}.pt" for key in stored_graphs.keys()],
            'experiment_timestamp': timestamp
        }
        with open(graphs_dir / f"graph_metadata_{timestamp}.json", 'w') as f:
            json.dump(graph_metadata, f, indent=4)
    
    # Save node-level results dictionary as json
    node_df = pd.DataFrame(node_results_list)
    node_df.to_csv(results_dir / f"node_results_{timestamp}.csv", index=False)
    # Save graph-level results dictionary as json
    graph_df = pd.DataFrame(graph_results_list)
    graph_df.to_csv(results_dir / f"graph_results_{timestamp}.csv", index=False)
    # Save combined results as a single DataFrame
    df_results = pd.concat([node_df, graph_df], ignore_index=True)
    df_results.to_csv(results_dir / f"combined_results_{timestamp}.csv", index=False)
    # Save the configuration used for the experiment
    with open(results_dir / f"config_{timestamp}.yaml", 'w') as f:
        yaml.dump(config, f)
    # Save the raw node-level results as a JSON file
    node_results_json = node_df.to_dict(orient='records')
    with open(results_dir / f"node_results_{timestamp}.json", 'w') as f:
        json.dump(node_results_json, f, indent=4)
    # Save the raw graph-level results as a JSON file
    graph_results_json = graph_df.to_dict(orient='records')
    with open(results_dir / f"graph_results_{timestamp}.json", 'w') as f:
        json.dump(graph_results_json, f, indent=4)
    
    from .paper_plots import generate_paper_sensitivity_plots_from_frames

    generated_plots = generate_paper_sensitivity_plots_from_frames(
        node_df,
        graph_df,
        results_dir / "paper_plots",
        results_dir=results_dir,
        primary_backbone=backbone_specs[0]['name'],
        h_value=0.5,
        mean_degree=mean_degree,
    )
    
    # plot_snr_accuracy_correlation(graph_df, results_dir / f"plot5_snr_accuracy_correlation_{timestamp}.png")
    
    print(f"Experiment results saved to {results_dir}")
    print(f"Generated {len(generated_plots)} paper-style plots in {results_dir / 'paper_plots'}")
    if single_graph_mode:
        print(f"Generated graphs saved to {results_dir / 'graphs'}")
    
    return results_dir

if __name__ == "__main__":
    # This would be run with: python -m bridge.sensitivity.run_experiment --config path_to_your_config.yaml
    import argparse
    parser = argparse.ArgumentParser(description="Run Comprehensive MPNN Sensitivity Analysis Framework Experiments")
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment configuration YAML file.')
    args = parser.parse_args()
    
    run_full_sensitivity_experiment(args.config)
