"""
Pipeline for graph rewiring and neural network training.

This module provides the main pipeline for rewiring graphs to optimize
the performance of graph neural networks.
"""

import torch
import torch.nn.functional as F
import dgl
import numpy as np
from tqdm import trange
from typing import Tuple, List, Dict, Union, Optional, Any

from ..models import GCN, SelectiveGCN
from ..training import train, train_selective, get_metric_type
from ..utils import (
    set_seed, check_symmetry, local_homophily, compute_confidence_interval
)
from .operations import create_rewired_graph


def run_bridge_pipeline(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    h_feats_gcn: int = 64,
    n_layers_gcn: int = 2,
    dropout_p_gcn: float = 0.5,
    model_lr_gcn: float = 1e-3,
    wd_gcn: float = 0.0,
    h_feats_selective: int = 64,
    n_layers_selective: int = 2,
    dropout_p_selective: float = 0.5,
    model_lr_selective: float = 1e-3,
    wd_selective: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    temperature: float = 1.0,
    p_add: float = 0.1,
    p_remove: float = 0.1,
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
    2. Using the trained GCN to infer node classes
    3. Computing an optimal block matrix for rewiring
    4. Rewiring the graph based on the optimal block matrix
    5. Training a selective GCN on both the original and rewired graphs
    
    Args:
        g: Input graph
        P_k: Permutation matrix for rewiring
        h_feats_gcn: Hidden feature dimension for the base GCN
        n_layers_gcn: Number of layers for the base GCN
        dropout_p_gcn: Dropout probability for the base GCN
        model_lr_gcn: Learning rate for the base GCN
        wd_gcn: Weight decay for the base GCN
        h_feats_selective: Hidden feature dimension for the selective GCN
        n_layers_selective: Number of layers for the selective GCN
        dropout_p_selective: Dropout probability for the selective GCN
        model_lr_selective: Learning rate for the selective GCN
        wd_selective: Weight decay for the selective GCN
        n_epochs: Maximum number of training epochs
        early_stopping: Number of epochs to look back for early stopping
        temperature: Temperature for softmax when computing class probabilities
        p_add: Probability of adding new edges during rewiring
        p_remove: Probability of removing existing edges during rewiring
        d_out: Desired output mean degree
        num_graphs: Number of rewired graphs to generate
        device: Device to perform computations on
        seed: Random seed for reproducibility
        log_training: Whether to print training progress
        train_mask: Boolean mask indicating training nodes
        val_mask: Boolean mask indicating validation nodes
        test_mask: Boolean mask indicating test nodes
        dataset_name: Name of the dataset
        do_hp: Whether to use high-pass filters
        do_self_loop: Whether to add self-loops
        do_residual_connections: Whether to use residual connections
        
    Returns:
        Dict[str, Any]: Results of the rewiring pipeline, including:
            - cold_start: Results for the base GCN
            - selective: Results for the selective GCN
            - original_stats: Statistics for the original graph
            - rewired_stats: Statistics for the rewired graph
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
        mean_local_homophily = local_homophily(n_layers_selective, graph, do_hp=do_hp).mean().item()
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'mean_degree': mean_degree,
            'mean_local_homophily': mean_local_homophily,
        }
        return stats

    original_stats = compute_graph_stats(g, labels)
    if log_training:
        print(f"Original Graph Stats: {original_stats}")

    ########################################################################
    # 2) Train Cold-Start GCN on the *Original* Graph
    ########################################################################
    in_feats = feat.shape[1]
    out_feats = int(labels.max().item()) + 1
    
    model_cold = GCN(
        in_feats, h_feats_gcn, out_feats, n_layers_gcn,
        dropout_p_gcn, residual_connection=do_residual_connections, do_hp=do_hp
    ).to(device)
    
    train_acc_cold, val_acc_cold, test_acc_cold, model_cold = train(
        g,
        model_cold,
        train_mask,
        val_mask,
        test_mask,
        model_lr=model_lr_gcn,
        optimizer_weight_decay=wd_gcn,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        log_training=log_training,
        metric_type=get_metric_type(dataset_name)
    )
    # Get predicted class distribution (softmax of logits/temperature)
    model_cold.eval()
    with torch.no_grad():
        logits = model_cold(g, feat)
        Z_pred = F.softmax(logits / temperature, dim=1)  # shape: [n_nodes, out_feats]
        pred = Z_pred.argmax(dim=1)                      # [n_nodes]

    # Ensure no empty classes: if any class is empty, fill it artificially
    unique_counts = pred.bincount(minlength=out_feats)
    empty_classes = torch.where(unique_counts == 0)[0]
    for i, empty_cls in enumerate(empty_classes):
        pred[i] = empty_cls

    ########################################################################
    # 3) Compute B_opt from predicted classes and old adjacency
    ########################################################################
    pred = pred.to(device)
    Z_pred = Z_pred.to(device)

    pi = Z_pred.cpu().numpy().sum(0) / n_nodes
    Pi_inv = np.diag(1/pi)
    B_opt = (d_out/k) * Pi_inv @ P_k @ Pi_inv
    B_opt_tensor = torch.tensor(B_opt, dtype=torch.float32, device=device)
    A_old = g.cpu().adj().to_dense()

    g = g.to(device)
    g_list = [g]
    
    ########################################################################
    # 4) Create Rewired Graphs
    ########################################################################
    for _ in range(num_graphs):
        if not check_symmetry(g):
            g_rewired = create_rewired_graph(
                g=g.to(device),
                B_opt_tensor=B_opt_tensor.to(device),
                pred=pred.to(device),
                Z_pred=Z_pred,
                p_add=p_add,
                p_remove=p_remove,
                device=device,
                sym_type='asymetric'
            )
        else:
            g_rewired = create_rewired_graph(
                g=g.to(device),
                B_opt_tensor=B_opt_tensor.to(device),
                pred=pred.to(device),
                Z_pred=Z_pred,
                p_add=p_add,
                p_remove=p_remove,
                device=device,
                sym_type='upper'
            )
            
        # Add self-loops if requested
        if do_self_loop:
            g_rewired = dgl.remove_self_loop(g_rewired)
            g_rewired = dgl.add_self_loop(g_rewired)
            
        g_list.append(g_rewired.to(device))

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
    # 5) Prepare Graph List for Selective GCN
    ########################################################################
    # Compute local homophily for each graph, using predicted labels on old nodes
    lh_list = []
    for i, g_i in enumerate(g_list):
        lh_tensor = local_homophily(
            n_layers_selective+1, g_i.to(device), y=pred.to(device), do_hp=do_hp
        )
        rewired_stats.update({
            f'predicted_mean_homophily{"_hp" if do_hp else ""}_{i}': torch.mean(lh_tensor).item()
        })
        lh_list.append(lh_tensor)

    lh_stack = torch.stack(lh_list)
    node_mask = lh_stack.argmax(dim=0)  # for each node, which graph is better?

    for g_i in g_list:
        g_i.ndata['mask'] = node_mask.to(device)
    rewired_stats.update({'selective_mask': str(node_mask.bincount(minlength=len(g_list)).cpu().numpy())})
    
    ########################################################################
    # 6) Train Selective GCN
    ########################################################################
    model_selective = SelectiveGCN(
        in_feats, h_feats_selective, out_feats,
        n_layers_selective, dropout_p_selective,
        residual_connection=do_residual_connections, do_hp=do_hp
    ).to(device)

    train_acc_sel, val_acc_sel, test_acc_sel, model_selective = train_selective(
        g_list,
        model_selective,
        train_mask,
        val_mask,
        test_mask,
        model_lr=model_lr_selective,
        optimizer_weight_decay=wd_selective,
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
        'selective': {
            'train_acc': train_acc_sel,
            'val_acc': val_acc_sel,
            'test_acc': test_acc_sel,
        },
        'original_stats': original_stats,
        'rewired_stats': rewired_stats
    }
    return results


def run_bridge_experiment(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    h_feats_gcn: int = 64,
    n_layers_gcn: int = 2,
    dropout_p_gcn: float = 0.5,
    model_lr_gcn: float = 1e-3,
    wd_gcn: float = 0.0,
    h_feats_selective: int = 64,
    n_layers_selective: int = 2,
    dropout_p_selective: float = 0.5,
    model_lr_selective: float = 1e-3,
    wd_selective: float = 0.0,
    n_epochs: int = 1000,
    early_stopping: int = 50,
    temperature: float = 1.0,
    p_add: float = 0.1,
    p_remove: float = 0.1,
    d_out: float = 10,
    num_graphs: int = 1,
    device: Union[str, torch.device] = 'cpu',
    num_repeats: int = 10,
    log_training: bool = False,
    dataset_name: str = 'unknown',
    do_hp: bool = False,
    do_self_loop: bool = False,
    do_residual_connections: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run the rewiring pipeline multiple times and average the results.
    
    This function runs the rewiring pipeline for multiple trials or data splits
    and computes the mean performance and confidence intervals.
    
    Args:
        g: Input graph
        P_k: Permutation matrix for rewiring
        h_feats_gcn: Hidden feature dimension for the base GCN
        n_layers_gcn: Number of layers for the base GCN
        dropout_p_gcn: Dropout probability for the base GCN
        model_lr_gcn: Learning rate for the base GCN
        wd_gcn: Weight decay for the base GCN
        h_feats_selective: Hidden feature dimension for the selective GCN
        n_layers_selective: Number of layers for the selective GCN
        dropout_p_selective: Dropout probability for the selective GCN
        model_lr_selective: Learning rate for the selective GCN
        wd_selective: Weight decay for the selective GCN
        n_epochs: Maximum number of training epochs
        early_stopping: Number of epochs to look back for early stopping
        temperature: Temperature for softmax when computing class probabilities
        p_add: Probability of adding new edges during rewiring
        p_remove: Probability of removing existing edges during rewiring
        d_out: Desired output mean degree
        num_graphs: Number of rewired graphs to generate
        device: Device to perform computations on
        num_repeats: Number of times to repeat the experiment
        log_training: Whether to print training progress
        dataset_name: Name of the dataset
        do_hp: Whether to use high-pass filters
        do_self_loop: Whether to add self-loops
        do_residual_connections: Whether to use residual connections
        
    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]:
            - Dictionary of aggregated statistics with means and confidence intervals
            - List of individual trial results
    """
    test_acc_list = []
    val_acc_list = []
    results_list = []
    
    # Check for multiple splits
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    has_multiple_splits = len(train_mask.shape) > 1
    num_splits = train_mask.shape[1] if has_multiple_splits else num_repeats
    
    # Lists to track statistics
    edges_added_list = []
    edges_removed_list = []
    original_density_list = []
    rewired_density_list = []
    original_homophily_list = []
    rewired_homophily_list = []
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
            h_feats_gcn=h_feats_gcn,
            n_layers_gcn=n_layers_gcn,
            dropout_p_gcn=dropout_p_gcn,
            model_lr_gcn=model_lr_gcn,
            wd_gcn=wd_gcn,
            h_feats_selective=h_feats_selective,
            n_layers_selective=n_layers_selective,
            dropout_p_selective=dropout_p_selective,
            model_lr_selective=model_lr_selective,
            wd_selective=wd_selective,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            temperature=temperature,
            p_remove=p_remove,
            p_add=p_add,
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
        test_acc_list.append(results['selective']['test_acc'])
        val_acc_list.append(results['selective']['val_acc'])
        results_list.append(results)
        
        edges_added_list.append(results['rewired_stats']['edges_added'])
        edges_removed_list.append(results['rewired_stats']['edges_removed'])
        
        original_density = results['original_stats']['num_edges'] / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
        rewired_density = results['rewired_stats']['num_edges'] / (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2)
        
        original_density_list.append(original_density)
        rewired_density_list.append(rewired_density)
        original_homophily_list.append(results['original_stats']['mean_local_homophily'])
        rewired_homophily_list.append(results['rewired_stats']['mean_local_homophily'])
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
            'homophily': compute_stats(original_homophily_list),
            'degree': compute_stats(original_degree_list)
        },
        'rewired_stats': {
            'density': compute_stats(rewired_density_list),
            'homophily': compute_stats(rewired_homophily_list),
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
            'homophily_mean': stats_dict['original_stats']['homophily']['mean'],
            'homophily_ci': stats_dict['original_stats']['homophily']['ci'],
            'degree_mean': stats_dict['original_stats']['degree']['mean'],
            'degree_ci': stats_dict['original_stats']['degree']['ci']
        },
        'rewired_stats': {
            'density_mean': stats_dict['rewired_stats']['density']['mean'],
            'density_ci': stats_dict['rewired_stats']['density']['ci'],
            'homophily_mean': stats_dict['rewired_stats']['homophily']['mean'],
            'homophily_ci': stats_dict['rewired_stats']['homophily']['ci'],
            'degree_mean': stats_dict['rewired_stats']['degree']['mean'],
            'degree_ci': stats_dict['rewired_stats']['degree']['ci']
        }
    }

    return formatted_stats, results_list
