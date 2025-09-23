"""
Pipeline for graph rewiring and neural network training.

This module provides the main pipeline for rewiring graphs to optimize
the performance of graph neural networks.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
from tqdm import trange
from typing import Tuple, List, Dict, Union, Optional, Any
import copy

from ..models import GCN, SGC
from ..training import train, get_metric_type
from ..utils import (
    set_seed, check_symmetry,
    class_bottlenecking_score, self_bottlenecking_score, total_bottlenecking_score,
    compute_confidence_interval, estimate_iid_variances
)
from .operations import create_rewired_graph
from .sdrf import sdrf_rewire
from .digl import digl_rewired

def create_model(
    model_type: str,
    in_feats: int,
    h_feats: int,
    out_feats: int,
    n_layers: int,
    dropout_p: float,
    do_residual_connections: bool = False,
    do_hp: bool = False,
    device: str = 'cpu'
):
    model_type = model_type.upper()
    if model_type == 'GCN':
        model = GCN(
            in_feats,
            h_feats,
            out_feats,
            n_layers,
            dropout_p,
            residual_connection=do_residual_connections,
            do_hp=do_hp
        ).to(device)
        return model
    # SGC can be supported via explicit construction elsewhere if needed
    raise ValueError(f"Unsupported model type: {model_type}. Only 'GCN' is supported in the standard pipeline.")


def run_bridge_pipeline(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    d_out: float = 10,
    num_graphs: int = 1,
    device: Union[str, torch.device] = 'cpu',
    seed: int = 0,
    log_training: bool = False,
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
    test_mask: Optional[torch.Tensor] = None,
    dataset_name: str = 'unknown',
    do_hp: bool = False,
    do_self_loop: bool = False,
    do_residual_connections: bool = False
) -> Dict[str, Any]:
    """
    Run the BRIDGE (Block Rewiring from Inference-Derived Graph Ensembles) pipeline.
    
    The pipeline consists of:
    1. Training a base GCN on the original graph
    2. Using the trained GCN to infer node classes (hard argmax)
    3. Computing an optimal block matrix for rewiring
    4. Rewiring the graph based on the optimal block matrix (full resampling)
    5. Training a standard GCN on the rewired graph
    
    Returns cold-start (original) and rewired model metrics, plus graph stats.
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Move graph to the device
    g = g.to(device)
    feat = g.ndata['feat']
    labels = g.ndata['label']
    k = len(torch.unique(labels))
    n_nodes = g.num_nodes()  # original number of nodes

    # Use provided masks or default to graph masks
    train_mask = train_mask if train_mask is not None else g.ndata['train_mask'].bool()
    val_mask   = val_mask   if val_mask   is not None else g.ndata['val_mask'].bool()
    test_mask  = test_mask  if test_mask  is not None else g.ndata['test_mask'].bool()

    ########################################################################
    # 1) Log Original Graph Statistics
    ########################################################################
    def compute_graph_stats(graph, labels=None):
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        mean_degree = graph.in_degrees().float().mean().item()
        mean_class_bottlenecking_score = class_bottlenecking_score(n_layers_mpnn+1, graph, do_hp=do_hp).mean().item()
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'mean_degree': mean_degree,
            'mean_class_bottlenecking_score': mean_class_bottlenecking_score,
        }
        return stats

    original_stats = compute_graph_stats(g, labels)
    if log_training:
        print(f"Original Graph Stats: {original_stats}")

    ########################################################################
    # 2) Train Cold-Start GCN on the Original Graph
    ########################################################################
    in_feats = feat.shape[1]
    out_feats = int(labels.max().item()) + 1
    
    model_cold = GCN(
        in_feats, h_feats_mpnn, out_feats, n_layers_mpnn,
        dropout_p_mpnn, residual_connection=do_residual_connections, do_hp=do_hp
    ).to(device)
    
    train_acc_cold, val_acc_cold, test_acc_cold, model_cold = train(
        g,
        model_cold,
        train_mask,
        val_mask,
        test_mask,
        model_lr=model_lr_mpnn,
        optimizer_weight_decay=wd_mpnn,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=log_training,
        metric_type=get_metric_type(dataset_name)
    )

    # Get predicted classes (hard predictions)
    model_cold.eval()
    with torch.no_grad():
        logits = model_cold(g, feat)
        pred = logits.argmax(dim=1)
        Z_pred = torch.zeros(n_nodes, out_feats, device=device)
        Z_pred.scatter_(1, pred.unsqueeze(1), 1.0)

    # Ensure no empty classes
    unique_counts = pred.bincount(minlength=out_feats)
    empty_classes = torch.where(unique_counts == 0)[0]
    for i, empty_cls in enumerate(empty_classes):
        pred[i] = empty_cls

    ########################################################################
    # 3) Compute B_opt and 4) Create Rewired Graph(s)
    ########################################################################
    pred = pred.to(device)
    Z_pred = Z_pred.to(device)

    pi = Z_pred.cpu().numpy().sum(0) / n_nodes
    pi = np.clip(pi, 1e-5, None)
    Pi_inv = np.diag(1/pi)
    B_opt = (d_out/k) * Pi_inv @ P_k @ Pi_inv
    B_opt_tensor = torch.tensor(B_opt, dtype=torch.float32, device=device)

    g = g.to(device)

    A_old = g.adj().cpu().to_dense()
    if not check_symmetry(g):
        g_rewired = create_rewired_graph(
            g=g.to(device),
            B_opt_tensor=B_opt_tensor.to(device),
            pred=pred.to(device),
            Z_pred=Z_pred,
            device=device,
            sym_type='asymetric'
        )
    else:
        g_rewired = create_rewired_graph(
            g=g.to(device),
            B_opt_tensor=B_opt_tensor.to(device),
            pred=pred.to(device),
            Z_pred=Z_pred,
            device=device,
            sym_type='upper'
        )

    # Add self-loops if requested
    if do_self_loop:
        g_rewired = dgl.remove_self_loop(g_rewired)
        g_rewired = dgl.add_self_loop(g_rewired)

    A_ext = g_rewired.adj().cpu().to_dense()
    edges_added   = ((A_ext > 0.5) & (A_old < 0.5)).sum().item()
    edges_removed = ((A_ext < 0.5) & (A_old > 0.5)).sum().item()
    
    rewired_stats = compute_graph_stats(g_rewired, g_rewired.ndata['label'])
    rewired_stats.update({
        'edges_added': edges_added,
        'edges_removed': edges_removed
    })
    if log_training:
        print(f"Rewired Graph Stats: {rewired_stats}")

    ########################################################################
    # 5) Train Standard GCN on Rewired Graph
    ########################################################################
    model_rw = GCN(
        in_feats, h_feats_mpnn, out_feats, n_layers_mpnn,
        dropout_p_mpnn, residual_connection=do_residual_connections, do_hp=do_hp
    ).to(device)

    train_acc_rw, val_acc_rw, test_acc_rw, model_rw = train(
        g_rewired,
        model_rw,
        train_mask,
        val_mask,
        test_mask,
        model_lr=model_lr_mpnn,
        optimizer_weight_decay=wd_mpnn,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=log_training,
        metric_type=get_metric_type(dataset_name)
    )

    results = {
        'cold_start': {
            'train_acc': train_acc_cold,
            'val_acc': val_acc_cold,
            'test_acc': test_acc_cold,
        },
        'rewired': {
            'train_acc': train_acc_rw,
            'val_acc': val_acc_rw,
            'test_acc': test_acc_rw,
        },
        'original_stats': original_stats,
        'rewired_stats': rewired_stats
    }
    return results


def run_bridge_experiment(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    d_out: float = 10,
    num_graphs: int = 1,
    device: Union[str, torch.device] = 'cpu',
    num_splits: int = 100,
    log_training: bool = False,
    dataset_name: str = 'unknown',
    do_hp: bool = False,
    do_self_loop: bool = False,
    do_residual_connections: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run the rewiring pipeline multiple times and average the results.
    Returns aggregated stats for the rewired model.
    """
    test_acc_list = []
    val_acc_list = []
    results_list = []
    
    # Check for multiple splits
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    has_multiple_splits = len(train_mask.shape) > 1
    num_splits = train_mask.shape[1] if has_multiple_splits else num_splits
    
    # Lists to track statistics
    edges_added_list = []
    edges_removed_list = []
    original_density_list = []
    rewired_density_list = []
    original_class_bottlenecking_list = []
    rewired_class_bottlenecking_list = []
    original_degree_list = []
    rewired_degree_list = []
    
    for split_idx in trange(num_splits):
        # Get masks for this split/repeat
        if has_multiple_splits:
            current_train_mask = train_mask[:, split_idx]
            current_val_mask = val_mask[:, split_idx]
            current_test_mask = test_mask[:, split_idx]
        else:
            current_train_mask = train_mask
            current_val_mask = val_mask
            current_test_mask = test_mask
            
        results = run_bridge_pipeline(
            g,
            P_k=P_k, 
            h_feats_mpnn=h_feats_mpnn,
            n_layers_mpnn=n_layers_mpnn,
            dropout_p_mpnn=dropout_p_mpnn,
            model_lr_mpnn=model_lr_mpnn,
            wd_mpnn=wd_mpnn,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            d_out=d_out,
            num_graphs=num_graphs,
            device=device,
            seed=split_idx,
            log_training=log_training,
            train_mask=current_train_mask,
            val_mask=current_val_mask,
            test_mask=current_test_mask,
            dataset_name=dataset_name,
            do_hp=do_hp,
            do_self_loop=do_self_loop,
            do_residual_connections=do_residual_connections
        )
        
        # Store results and statistics
        test_acc_list.append(results['rewired']['test_acc'])
        val_acc_list.append(results['rewired']['val_acc'])
        results_list.append(results)
        
        edges_added_list.append(results['rewired_stats']['edges_added'])
        edges_removed_list.append(results['rewired_stats']['edges_removed'])
        
        original_density = results['original_stats']['num_edges'] / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
        rewired_density = results['rewired_stats']['num_edges'] / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
        
        original_density_list.append(original_density)
        rewired_density_list.append(rewired_density)
        original_class_bottlenecking_list.append(results['original_stats']['mean_class_bottlenecking_score'])
        rewired_class_bottlenecking_list.append(results['rewired_stats']['mean_class_bottlenecking_score'])
        original_degree_list.append(results['original_stats']['mean_degree'])
        rewired_degree_list.append(results['rewired_stats']['mean_degree'])

    # Compute statistics
    def compute_stats(data_list):
        mean, lower, upper = compute_confidence_interval(data_list)
        return {'mean': mean, 'ci': (lower, upper)}
    
    stats_dict = {
        'test_acc': compute_stats(test_acc_list),
        'val_acc': compute_stats(val_acc_list),
        'edges_added': compute_stats(edges_added_list),
        'edges_removed': compute_stats(edges_removed_list),
        'original_stats': {
            'density': compute_stats(original_density_list),
            'class_bottlenecking': compute_stats(original_class_bottlenecking_list),
            'degree': compute_stats(original_degree_list)
        },
        'rewired_stats': {
            'density': compute_stats(rewired_density_list),
            'class_bottlenecking': compute_stats(rewired_class_bottlenecking_list),
            'degree': compute_stats(rewired_degree_list)
        }
    }
    
    # Format the output to match the expected structure
    formatted_stats = {
        'test_acc_mean': stats_dict['test_acc']['mean'],
        'test_acc_ci': stats_dict['test_acc']['ci'],
        'val_acc_mean': stats_dict['val_acc']['mean'],
        'val_acc_ci': stats_dict['val_acc']['ci'],
        'edges_added_mean': stats_dict['edges_added']['mean'],
        'edges_added_ci': stats_dict['edges_added']['ci'],
        'edges_removed_mean': stats_dict['edges_removed']['mean'],
        'edges_removed_ci': stats_dict['edges_removed']['ci'],
        'original_stats': {
            'density_mean': stats_dict['original_stats']['density']['mean'],
            'density_ci': stats_dict['original_stats']['density']['ci'],
            'class_bottlenecking_mean': stats_dict['original_stats']['class_bottlenecking']['mean'],
            'class_bottlenecking_ci': stats_dict['original_stats']['class_bottlenecking']['ci'],
            'degree_mean': stats_dict['original_stats']['degree']['mean'],
            'degree_ci': stats_dict['original_stats']['degree']['ci']
        },
        'rewired_stats': {
            'density_mean': stats_dict['rewired_stats']['density']['mean'],
            'density_ci': stats_dict['rewired_stats']['density']['ci'],
            'class_bottlenecking_mean': stats_dict['rewired_stats']['class_bottlenecking']['mean'],
            'class_bottlenecking_ci': stats_dict['rewired_stats']['class_bottlenecking']['ci'],
            'degree_mean': stats_dict['rewired_stats']['degree']['mean'],
            'degree_ci': stats_dict['rewired_stats']['degree']['ci']
        }
    }

    return formatted_stats, results_list


def run_iterative_bridge_pipeline(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    model_type: str = 'GCN',
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    d_out: float = 10,
    num_graphs: int = 1,
    device: Union[str, torch.device] = 'cpu',
    seed: int = 0,
    log_training: bool = False,
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
    test_mask: Optional[torch.Tensor] = None,
    dataset_name: str = 'unknown',
    do_hp: bool = False,
    do_self_loop: bool = False,
    do_residual_connections: bool = False,
    use_sgc: bool = True,
    n_rewire: int = 10,
    sgc_K: int = 2,
    sgc_lr: float = 1e-2,
    sgc_wd: float = 1e-4,
    rewiring_method: str = "bridge",
    tau: float = 0.1,
    sdrf_iterations: int = 1,
    c_plus: float = 0.0,
    digl_diffusion_type: str = 'ppr',
    digl_alpha: float = 0.15,
    digl_k: int = 64,
    digl_t: float = 5.0,
    digl_epsilon: float = 0.0001,
    simulated_acc: Optional[float] = None
) -> Dict[str, Any]:
    """
    Iterative BRIDGE pipeline.
    Trains a standard model on the final rewired graph.
    """
    set_seed(seed)

    g = g.to(device)
    feat = g.ndata['feat']
    labels = g.ndata['label']
    k = len(torch.unique(labels))
    n_nodes = g.num_nodes()

    train_mask = train_mask if train_mask is not None else g.ndata['train_mask'].bool()
    val_mask = val_mask if val_mask is not None else g.ndata['val_mask'].bool()
    test_mask = test_mask if test_mask is not None else g.ndata['test_mask'].bool()

    g_original = g.clone()

    if log_training:
        print(f"Using model type: {model_type.upper()}")

    def compute_graph_stats(graph,l):
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        mean_degree = graph.in_degrees().float().mean().item()
        mean_class_bottlenecking_score = class_bottlenecking_score(l, graph, do_hp=do_hp).mean().item()
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'mean_degree': mean_degree,
            'mean_class_bottlenecking_score': mean_class_bottlenecking_score,
        }
        return stats

    original_stats = compute_graph_stats(g, n_layers_mpnn + 1)

    g_rewired = g.clone()

    if rewiring_method == 'sdrf':
        g_rewired = sdrf_rewire(
            g_in=g_rewired,
            tau=tau,
            n_iterations=sdrf_iterations,
            c_plus=c_plus)
    elif rewiring_method == 'digl':
        g_rewired = digl_rewired(
            g=g_rewired,
            method=digl_diffusion_type,
            alpha=digl_alpha,
            k = digl_k,
            t=digl_t,
            eps=digl_epsilon)

    in_feats = feat.shape[1]
    out_feats = int(labels.max().item()) + 1

    rewiring_history = []

    # First (cold-start) model for predictions
    model = create_model(
        model_type=model_type,
        in_feats=in_feats,
        h_feats=h_feats_mpnn,
        out_feats=out_feats,
        n_layers=n_layers_mpnn,
        dropout_p=dropout_p_mpnn,
        do_residual_connections=do_residual_connections,
        do_hp=do_hp,
        device=device
    )

    train_acc_cold, val_acc_cold, test_acc_cold, model = train(
        g_rewired.to(device),
        model,
        train_mask,
        val_mask,
        test_mask,
        model_lr=model_lr_mpnn,
        optimizer_weight_decay=wd_mpnn,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=log_training,
        metric_type=get_metric_type(dataset_name)
    )

    for iter_idx in range(n_rewire):
        if simulated_acc is not None:
            noise_fraction = (1.0-simulated_acc)/(1.0-1.0/k)
            pred = labels.clone()
            num_noisy_nodes = int(noise_fraction * n_nodes)
            noise_indices = torch.randperm(n_nodes, device=device)[:num_noisy_nodes]
            noise_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            noise_mask[noise_indices] = True
            if num_noisy_nodes > 0:
                random_labels = torch.randint(0, out_feats, (num_noisy_nodes,), device=device)
                pred[noise_mask] = random_labels
            Z_pred = torch.zeros(n_nodes, out_feats, device=device)
            Z_pred.scatter_(1, pred.unsqueeze(1), 1.0)
        else:
            model.eval()
            with torch.no_grad():
                logits = model(g_rewired.to(device), feat.to(device))
                pred = logits.argmax(dim=1)
                Z_pred = torch.zeros(n_nodes, out_feats, device=device)
                Z_pred.scatter_(1, pred.unsqueeze(1), 1.0)

        unique_counts = pred.bincount(minlength=out_feats)
        empty_classes = torch.where(unique_counts == 0)[0]
        for i, empty_cls in enumerate(empty_classes):
            pred[i] = empty_cls

        pi = Z_pred.cpu().numpy().sum(0) / n_nodes
        Pi_inv = np.diag(1/(pi+1e-8))
        B_opt = (d_out/k) * Pi_inv @ P_k @ Pi_inv
        B_opt_tensor = torch.tensor(B_opt, dtype=torch.float32, device=device)
        
        A_old = g_rewired.adj().to_dense().cpu()

        if rewiring_method == 'sdrf' or rewiring_method == 'digl':
            g_rewired = g_rewired.to(device)
        else:
            g_rewired = create_rewired_graph(
                g=g_rewired.to(device),
                B_opt_tensor=B_opt_tensor.to(device),
                pred=pred.to(device),
                Z_pred=Z_pred,
                device=device,
                sym_type='upper'
            )

        if do_self_loop:
            g_rewired = dgl.remove_self_loop(g_rewired)
            g_rewired = dgl.add_self_loop(g_rewired)
        
        A_new = g_rewired.adj().to_dense().cpu()
        edges_added = ((A_new > 0.5) & (A_old < 0.5)).sum().item()
        edges_removed = ((A_new < 0.5) & (A_old > 0.5)).sum().item()
        
        current_stats = compute_graph_stats(g_rewired, n_layers_mpnn + 1)
        current_stats.update({
            'edges_added': edges_added,
            'edges_removed': edges_removed,
            'iteration': iter_idx + 1,
        })
        rewiring_history.append(copy.deepcopy(current_stats))

        if log_training:
            print(f"Iteration {iter_idx+1} Stats ({model_type}) | Edges: {current_stats['num_edges']} | Added: {edges_added} | Removed: {edges_removed}")

        if rewiring_method in ('sdrf','digl'):
            break

    # Final training on rewired graph (standard model)
    model_final = create_model(
        model_type=model_type,
        in_feats=in_feats,
        h_feats=h_feats_mpnn,
        out_feats=out_feats,
        n_layers=n_layers_mpnn,
        dropout_p=dropout_p_mpnn,
        do_residual_connections=do_residual_connections,
        do_hp=do_hp,
        device=device
    )

    train_acc_rw, val_acc_rw, test_acc_rw, model_final = train(
        g_rewired.to(device),
        model_final,
        train_mask,
        val_mask,
        test_mask,
        model_lr=model_lr_mpnn,
        optimizer_weight_decay=wd_mpnn,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=log_training,
        metric_type=get_metric_type(dataset_name)
    )

    results = {
        'cold_start': {
            'train_acc': train_acc_cold,
            'val_acc': val_acc_cold,
            'test_acc': test_acc_cold,
        },
        'rewired': {
            'train_acc': train_acc_rw,
            'val_acc': val_acc_rw,
            'test_acc': test_acc_rw,
        },
        'original_stats': original_stats,
        'rewired_stats': rewiring_history[-1] if rewiring_history else compute_graph_stats(g_rewired, n_layers_mpnn + 1),
        'rewiring_history': rewiring_history,
        'model_type': model_type,
    }
    return results


def run_iterative_bridge_experiment(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    model_type: str = 'GCN',
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    d_out: float = 10,
    num_graphs: int = 1,
    device: Union[str, torch.device] = 'cpu',
    num_repeats: int = 10,
    log_training: bool = False,
    dataset_name: str = 'unknown',
    do_hp: bool = False,
    do_self_loop: bool = False,
    do_residual_connections: bool = False,
    use_sgc: bool = True,
    n_rewire: int = 10,
    sgc_K: int = 2,
    sgc_lr: float = 1e-2,
    sgc_wd: float = 1e-4,
    rewiring_method: str = "bridge",
    tau: float = 0.1,
    sdrf_iterations: int = 1,
    c_plus: float = 0.0,
    digl_diffusion_type: str = 'ppr',
    digl_alpha: float = 0.15,
    digl_k: int = 64,
    digl_t: float = 5.0,
    digl_epsilon: float = 0.0001,
    simulated_acc: Optional[float] = None
    
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Iterative rewiring experiment returning aggregated stats for the standard model trained on the final rewired graph.
    """
    test_acc_list = []
    val_acc_list = []
    results_list = []
    
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    has_multiple_splits = len(train_mask.shape) > 1
    num_splits = train_mask.shape[1] if has_multiple_splits else num_repeats
    
    edges_added_list = []
    edges_removed_list = []
    original_density_list = []
    rewired_density_list = []
    original_class_bottlenecking_list = []
    rewired_class_bottlenecking_list = []
    original_degree_list = []
    rewired_degree_list = []
    
    for split_idx in trange(num_splits):
        if has_multiple_splits:
            current_train_mask = train_mask[:, split_idx]
            current_val_mask = val_mask[:, split_idx]
            current_test_mask = test_mask[:, split_idx]
        else:
            current_train_mask = train_mask
            current_val_mask = val_mask
            current_test_mask = test_mask
            
        results = run_iterative_bridge_pipeline(
            g,
            P_k=P_k,
            model_type=model_type,
            h_feats_mpnn=h_feats_mpnn,
            n_layers_mpnn=n_layers_mpnn,
            dropout_p_mpnn=dropout_p_mpnn,
            model_lr_mpnn=model_lr_mpnn,
            wd_mpnn=wd_mpnn,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            d_out=d_out,
            num_graphs=num_graphs,
            device=device,
            seed=split_idx,
            log_training=log_training,
            train_mask=current_train_mask,
            val_mask=current_val_mask,
            test_mask=current_test_mask,
            dataset_name=dataset_name,
            do_hp=do_hp,
            do_self_loop=do_self_loop,
            do_residual_connections=do_residual_connections,
            use_sgc=use_sgc,
            n_rewire=n_rewire,
            sgc_K=sgc_K,
            sgc_lr=sgc_lr,
            sgc_wd=sgc_wd,
            rewiring_method=rewiring_method,
            tau=tau,
            sdrf_iterations=sdrf_iterations,
            c_plus=c_plus,
            digl_diffusion_type=digl_diffusion_type,
            digl_alpha=digl_alpha,
            digl_k=digl_k,
            digl_t=digl_t,
            digl_epsilon=digl_epsilon,
            simulated_acc=simulated_acc
        )
        
        test_acc_list.append(results['rewired']['test_acc'])
        val_acc_list.append(results['rewired']['val_acc'])
        
        results_list.append(results)
        
        edges_added_list.append(results['rewired_stats']['edges_added'])
        edges_removed_list.append(results['rewired_stats']['edges_removed'])
        
        original_density = results['original_stats']['num_edges'] / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
        rewired_density = results['rewired_stats']['num_edges'] / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
        
        original_density_list.append(original_density)
        rewired_density_list.append(rewired_density)
        original_class_bottlenecking_list.append(results['original_stats']['mean_class_bottlenecking_score'])
        rewired_class_bottlenecking_list.append(results['rewired_stats']['mean_class_bottlenecking_score'])
        original_degree_list.append(results['original_stats']['mean_degree'])
        rewired_degree_list.append(results['rewired_stats']['mean_degree'])

    def compute_stats(data_list):
        mean, lower, upper = compute_confidence_interval(data_list)
        return {'mean': mean, 'ci': (lower, upper)}
    
    stats_dict = {
        'model_type': model_type,
        'test_acc': compute_stats(test_acc_list),
        'val_acc': compute_stats(val_acc_list),
        'edges_added': compute_stats(edges_added_list),
        'edges_removed': compute_stats(edges_removed_list),
        'original_stats': {
            'density': compute_stats(original_density_list),
            'class_bottlenecking': compute_stats(original_class_bottlenecking_list),
            'degree': compute_stats(original_degree_list)
        },
        'rewired_stats': {
            'density': compute_stats(rewired_density_list),
            'class_bottlenecking': compute_stats(rewired_class_bottlenecking_list),
            'degree': compute_stats(rewired_degree_list)
        }
    }
    
    formatted_stats = {
        'model_type': model_type,
        'test_acc_mean': stats_dict['test_acc']['mean'],
        'test_acc_ci': stats_dict['test_acc']['ci'],
        'val_acc_mean': stats_dict['val_acc']['mean'],
        'val_acc_ci': stats_dict['val_acc']['ci'],
        'edges_added_mean': stats_dict['edges_added']['mean'],
        'edges_added_ci': stats_dict['edges_added']['ci'],
        'edges_removed_mean': stats_dict['edges_removed']['mean'],
        'edges_removed_ci': stats_dict['edges_removed']['ci'],
        'original_stats': {
            'density_mean': stats_dict['original_stats']['density']['mean'],
            'density_ci': stats_dict['original_stats']['density']['ci'],
            'class_bottlenecking_mean': stats_dict['original_stats']['class_bottlenecking']['mean'],
            'class_bottlenecking_ci': stats_dict['original_stats']['class_bottlenecking']['ci'],
            'degree_mean': stats_dict['original_stats']['degree']['mean'],
            'degree_ci': stats_dict['original_stats']['degree']['ci']
        },
        'rewired_stats': {
            'density_mean': stats_dict['rewired_stats']['density']['mean'],
            'density_ci': stats_dict['rewired_stats']['density']['ci'],
            'class_bottlenecking_mean': stats_dict['rewired_stats']['class_bottlenecking']['mean'],
            'class_bottlenecking_ci': stats_dict['rewired_stats']['class_bottlenecking']['ci'],
            'degree_mean': stats_dict['rewired_stats']['degree']['mean'],
            'degree_ci': stats_dict['rewired_stats']['degree']['ci']
        }
    }

    return formatted_stats, results_list
