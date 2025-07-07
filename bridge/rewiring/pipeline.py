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

from ..models import GCN, SelectiveGCN, SGC, SelectiveSGC
from ..training import train, train_selective, get_metric_type
from ..utils import (
    set_seed, check_symmetry, local_homophily, local_autophily, local_total_connectivity,
    compute_confidence_interval, estimate_iid_variances
)
from .operations import create_rewired_graph

def run_bridge_pipeline(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
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
        h_feats_mpnn: Hidden feature dimension for the base GCN
        n_layers_mpnn: Number of hidden layers for the base GCN
        dropout_p_mpnn: Dropout probability for the base GCN
        model_lr_mpnn: Learning rate for the base GCN
        wd_mpnn: Weight decay for the base GCN
        h_feats_selective: Hidden feature dimension for the selective GCN
        n_layers_selective: Number of hidden layers for the selective GCN
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
        mean_local_homophily = local_homophily(n_layers_selective+1, graph, do_hp=do_hp).mean().item()
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
    #clip pi to avoid division by zero
    pi = np.clip(pi, 1e-5, None)
    Pi_inv = np.diag(1/pi)
    B_opt = (d_out/k) * Pi_inv @ P_k @ Pi_inv
    B_opt_tensor = torch.tensor(B_opt, dtype=torch.float32, device=device)

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
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
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
    num_splits: int = 100,
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
        h_feats_mpnn: Hidden feature dimension for the base GCN
        n_layers_mpnn: Number of hidden layers for the base GCN
        dropout_p_mpnn: Dropout probability for the base GCN
        model_lr_mpnn: Learning rate for the base GCN
        wd_mpnn: Weight decay for the base GCN
        h_feats_selective: Hidden feature dimension for the selective GCN
        n_layers_selective: Number of hidden layers for the selective GCN
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
        num_splits: Number of times to repeat the experiment
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
    num_splits = train_mask.shape[1] if has_multiple_splits else num_splits
    
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
            h_feats_mpnn=h_feats_mpnn,
            n_layers_mpnn=n_layers_mpnn,
            dropout_p_mpnn=dropout_p_mpnn,
            model_lr_mpnn=model_lr_mpnn,
            wd_mpnn=wd_mpnn,
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


def calculate_accuracy(pred, labels, mask, device):
    """Calculate accuracy ensuring all tensors are on the same device."""
    pred = pred.to(device)
    labels = labels.to(device) 
    mask = mask.to(device)
    return ((pred == labels)[mask]).float().mean().item()

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GATv2Conv, GINConv, SAGEConv,GraphConv
from typing import Optional, Callable, Union

class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) implementation.
    
    This implementation supports variable depth and optional residual connections.
    
    Args:
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        n_layers: Number of hidden GCN layers
        dropout_p: Dropout probability
        activation: Activation function to use (default: F.relu)
        bias: Whether to use bias in GraphConv layers
        residual_connection: Deprecated flag for residual connections (not used in this implementation)
        do_hp: Depreciated flag for High Pass Graph Convolution (not used in this implementation)
    """
    def __init__(
        self, 
        in_feats: int, 
        h_feats: int, 
        out_feats: int, 
        n_layers: int, 
        dropout_p: float, 
        activation: Callable = F.relu, 
        bias: bool = True, 
        residual_connection: bool = False,
        do_hp: bool = False
    ):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        self.do_hp = do_hp
        
        # Input layer
        self.layers.append(GraphConv(in_feats, h_feats, bias=bias, allow_zero_in_degree=True))
        
        # Hidden layers (if any)
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(h_feats, h_feats, bias=bias, allow_zero_in_degree=True))
     
        # Output layer
        self.layers.append(GraphConv(h_feats, out_feats, bias=bias, allow_zero_in_degree=True))

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GCN model.
        
        Args:
            g: Input graph
            features: Node feature matrix
            
        Returns:
            torch.Tensor: Node embeddings
        """
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:  # no activation & dropout on the output layer
                h = self.activation(h)
                h = self.dropout(h)
        return h

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) implementation.
    This implementation supports variable depth, multiple attention heads, and optional residual connections.
    
    Args:
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        n_layers: Number of hidden GAT layers
        dropout_p: Dropout probability
        heads: Number of attention heads for hidden layers (default: 8)
        out_heads: Number of attention heads for output layer (default: 1)
        activation: Activation function to use (default: F.relu)
        feat_drop: Feature dropout rate (default: 0.0)
        attn_drop: Attention dropout rate (default: 0.0)
        negative_slope: Negative slope for LeakyReLU (default: 0.2)
        residual_connection: Whether to use residual connections (default: False)
        do_hp: Deprecated flag for High Pass Graph Convolution (not used)
    """
    def __init__(
        self,
        in_feats: int,
        h_feats: int,
        out_feats: int,
        n_layers: int,
        dropout_p: float,
        heads: int = 3,
        out_heads: int = 1,
        activation: Callable = F.relu,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        negative_slope: float = 0.2,
        residual_connection: bool = False,
        do_hp: bool = False
    ):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        self.n_layers = n_layers
        self.heads = heads
        self.out_heads = out_heads
        
        # Input layer
        self.layers.append(GATv2Conv(
            in_feats, h_feats, heads,
            feat_drop=feat_drop, attn_drop=attn_drop,
            negative_slope=negative_slope, residual=residual_connection,
            allow_zero_in_degree=True
        ))
        
        # Hidden layers (if any)
        for _ in range(n_layers - 1):
            self.layers.append(GATv2Conv(
                h_feats * heads, h_feats, heads,
                feat_drop=feat_drop, attn_drop=attn_drop,
                negative_slope=negative_slope, residual=residual_connection,
                allow_zero_in_degree=True
            ))
        
        # Output layer
        self.layers.append(GATv2Conv(
            h_feats * heads, out_feats, out_heads,
            feat_drop=feat_drop, attn_drop=attn_drop,
            negative_slope=negative_slope, residual=False,
            allow_zero_in_degree=True
        ))

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GAT model.
        
        Args:
            g: Input graph
            features: Node feature matrix
            
        Returns:
            torch.Tensor: Node embeddings
        """
        h = features
        
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            
            if i != len(self.layers) - 1:  # not the output layer
                # Flatten multi-head attention outputs
                h = h.flatten(1)
                h = self.activation(h)
                h = self.dropout(h)
            else:  # output layer
                # Average attention heads for output
                h = h.mean(1)
                
        return h


class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) implementation.
    This implementation uses MLPs as the aggregation function.
    
    Args:
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        n_layers: Number of hidden GIN layers
        dropout: Dropout probability
        aggregator_type: Type of aggregator ('sum', 'mean', 'max')
        learn_eps: Whether to learn the epsilon parameter
        activation: Activation function to use (default: F.relu)
        residual_connection: Whether to use residual connections (default: False)
        do_hp: Deprecated flag for High Pass Graph Convolution (not used)
    """
    def __init__(
        self,
        in_feats: int,
        h_feats: int,
        out_feats: int,
        n_layers: int,
        dropout: float,
        aggregator_type: str = 'sum',
        learn_eps: bool = False,
        activation: Callable = F.relu,
        residual_connection: bool = False,
        do_hp: bool = False
    ):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        
        def create_mlp(input_dim, output_dim):
            """Create a 2-layer MLP"""
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
        
        # Input layer
        mlp = create_mlp(in_feats, h_feats)
        self.layers.append(GINConv(
            mlp, aggregator_type, learn_eps=learn_eps
        ))
        
        # Hidden layers (if any)
        for _ in range(n_layers - 1):
            mlp = create_mlp(h_feats, h_feats)
            self.layers.append(GINConv(
                mlp, aggregator_type, learn_eps=learn_eps
            ))
        
        # Output layer
        mlp = create_mlp(h_feats, out_feats)
        self.layers.append(GINConv(
            mlp, aggregator_type, learn_eps=learn_eps
        ))

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GIN model.
        
        Args:
            g: Input graph
            features: Node feature matrix
            
        Returns:
            torch.Tensor: Node embeddings
        """
        h = features
        
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            
            if i != len(self.layers) - 1:  # no activation & dropout on the output layer
                h = self.activation(h)
                h = self.dropout(h)
                
        return h


class GraphSAGE(nn.Module):
    """
    GraphSAGE implementation.
    This implementation supports different aggregation functions and variable depth.
    
    Args:
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        n_layers: Number of hidden SAGE layers
        dropout: Dropout probability
        aggregator_type: Type of aggregator ('mean', 'mpnn', 'pool', 'lstm')
        feat_drop: Feature dropout rate (default: 0.0)
        bias: Whether to use bias in linear layers
        norm: Normalization function to use (default: None)
        activation: Activation function to use (default: F.relu)
        residual_connection: Whether to use residual connections (default: False)
        do_hp: Deprecated flag for High Pass Graph Convolution (not used)
    """
    def __init__(
        self,
        in_feats: int,
        h_feats: int,
        out_feats: int,
        n_layers: int,
        dropout: float,
        aggregator_type: str = 'mean',
        feat_drop: float = 0.0,
        bias: bool = True,
        norm: Optional[Callable] = None,
        activation: Callable = F.relu,
        residual_connection: bool = False,
        do_hp: bool = False
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        
        # Input layer
        self.layers.append(SAGEConv(
            in_feats, h_feats, aggregator_type,
            feat_drop=feat_drop, bias=bias, norm=norm
        ))
        
        # Hidden layers (if any)
        for _ in range(n_layers - 1):
            self.layers.append(SAGEConv(
                h_feats, h_feats, aggregator_type,
                feat_drop=feat_drop, bias=bias, norm=norm
            ))
        
        # Output layer
        self.layers.append(SAGEConv(
            h_feats, out_feats, aggregator_type,
            feat_drop=feat_drop, bias=bias, norm=norm
        ))

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GraphSAGE model.
        
        Args:
            g: Input graph
            features: Node feature matrix
            
        Returns:
            torch.Tensor: Node embeddings
        """
        h = features
        
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            
            if i != len(self.layers) - 1:  # no activation & dropout on the output layer
                h = self.activation(h)
                h = self.dropout(h)
                
        return h


# Updated factory function to work with the new models
def create_model(model_type: str, in_feats: int, h_feats: int, out_feats: int, 
                 n_layers: int, dropout_p: float, do_residual_connections: bool = False, 
                 do_hp: bool = False, device: str = 'cpu'):
    """
    Factory function to create different model types.
    
    Args:
        model_type: Type of model ('GCN', 'GAT', 'GIN', 'GraphSAGE')
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension
        out_feats: Output feature dimension
        n_layers: Number of layers
        dropout_p: Dropout probability
        do_residual_connections: Whether to use residual connections
        do_hp: Whether to use high-pass filters
        device: Device to place model on
    
    Returns:
        Model instance
    """
    model_type = model_type.upper()
    
    if model_type == 'GCN':
        model = GCN(
            in_feats, h_feats, out_feats, n_layers,
            dropout_p, residual_connection=do_residual_connections, do_hp=do_hp
        ).to(device)
    
    elif model_type == 'GAT':
        # GAT typically uses attention heads
        num_heads = 3 if n_layers > 1 else 1  # Use multiple heads for hidden layers
        model = GAT(
            in_feats, h_feats, out_feats, n_layers,
            dropout_p, heads=num_heads, 
            residual_connection=do_residual_connections, do_hp=do_hp
        ).to(device)
    
    elif model_type == 'GIN':
        model = GIN(
            in_feats, h_feats, out_feats, n_layers,
            dropout=dropout_p, residual_connection=do_residual_connections, do_hp=do_hp
        ).to(device)
    
    elif model_type == 'GRAPHSAGE':
        model = GraphSAGE(
            in_feats, h_feats, out_feats, n_layers,
            dropout=dropout_p, aggregator_type='mean',
            residual_connection=do_residual_connections, do_hp=do_hp
        ).to(device)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: GCN, GAT, GIN, GRAPHSAGE")
    
    return model

def run_iterative_bridge_pipeline(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    model_type: str = 'GCN',  # New parameter for model type
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
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
    do_residual_connections: bool = False,
    use_sgc: bool = True,
    n_rewire: int = 10,
    sgc_K: int = 2,
    sgc_lr: float = 1e-2,
    sgc_wd: float = 1e-4,
    simulated_acc: Optional[float] = None
) -> Dict[str, Any]:
    """
    Run an iterative version of the BRIDGE pipeline that performs multiple rounds of rewiring.
    
    This function repeatedly:
    1. Classifies nodes using a specified model type for the first iteration
    2. For subsequent iterations, uses the same model type on both the original and current rewired graph
    3. Rewires the graph based on the predicted classes
    4. Repeats the process n_rewire times
    5. Trains a final selective model on the original and final rewired graph
    
    Args:
        g: Input graph
        P_k: Permutation matrix for rewiring
        model_type: Type of model to use ('GCN', 'GAT', 'GIN', 'GraphSAGE')
        h_feats_mpnn: Hidden feature dimension for the base model
        n_layers_mpnn: Number of hidden layers for the base model
        dropout_p_mpnn: Dropout probability for the base model
        model_lr_mpnn: Learning rate for the base model
        wd_mpnn: Weight decay for the base model
        h_feats_selective: Hidden feature dimension for the selective model
        n_layers_selective: Number of hidden layers for the selective model
        dropout_p_selective: Dropout probability for the selective model
        model_lr_selective: Learning rate for the selective model
        wd_selective: Weight decay for the selective model
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
        use_sgc: Whether to use SGC for classification (faster) or standard model
        n_rewire: Number of rewiring iterations
        sgc_K: Number of propagation steps for SGC
        sgc_lr: Learning rate specifically for the SGC model in first iteration
        sgc_wd: Weight decay specifically for the SGC model in first iteration
        simulated_acc: Optional float between 0 and 1 representing the accuracy of simulated predictions. 
                                   If provided, skips model training and uses noisy ground truth labels instead.
                                   E.g., 0.8 means 80% of predictions are correct, 20% are random noise.
        
    Returns:
        Dict[str, Any]: Results of the rewiring pipeline, including:
            - cold_start: Results for the base model
            - selective: Results for the selective model
            - original_stats: Statistics for the original graph
            - rewired_stats: Statistics for the rewired graph
            - rewiring_history: Statistics at each rewiring step
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
    val_mask = val_mask if val_mask is not None else g.ndata['val_mask'].bool()
    test_mask = test_mask if test_mask is not None else g.ndata['test_mask'].bool()

    # Store original graph for final training and selective model
    g_original = g.clone()
    
    # Log model type being used
    if log_training:
        print(f"Using model type: {model_type.upper()}")
    
    ########################################################################
    # 1) Log Original Graph Statistics
    ########################################################################

    def compute_graph_stats(graph,l):
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        mean_degree = graph.in_degrees().float().mean().item()
        mean_local_homophily = local_homophily(l, graph, do_hp=do_hp).mean().item()
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'mean_degree': mean_degree,
            'mean_local_homophily': mean_local_homophily,
        }
        return stats

    original_stats = compute_graph_stats(g, n_layers_mpnn + 1)
    
    if log_training:
        print(f"Original Graph Stats: {original_stats}")
    
    # Initialize graph for iterative rewiring
    g_rewired = g.clone()
    
    # Initialize in_feats and out_feats
    in_feats = feat.shape[1]
    out_feats = int(labels.max().item()) + 1
    
    ########################################################################
    # 2) Iterative Rewiring Process
    ########################################################################
    
    # Keep track of statistics over all iterations
    rewiring_history = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_train_acc = 0.0
    best_iter_idx = 0
    
    for iter_idx in range(n_rewire):
        if simulated_acc is not None:
            # Add label noise for troubleshooting
            # simulated_acc represents the accuracy (1 - noise_fraction*(1-1/k))
            # noise will be right 1/k of the time, so need to adjust noise_fraction accordingly
            noise_fraction = (1.0-simulated_acc)/(1.0-1.0/k) #1.0 - ( k/(k-1) * simulated_acc - 1/(k-1))
            
            # Start with true labels
            pred = labels.clone()
            
            # Create a mask for nodes that will get noisy labels
            num_noisy_nodes = int(noise_fraction * n_nodes)
            noise_indices = torch.randperm(n_nodes, device=device)[:num_noisy_nodes]
            noise_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            noise_mask[noise_indices] = True
            
            # Assign random labels to noisy nodes
            if num_noisy_nodes > 0:
                random_labels = torch.randint(0, out_feats, (num_noisy_nodes,), device=device)
                pred[noise_mask] = random_labels
            
            # Create corresponding Z_pred (one-hot encoding)
            Z_pred = torch.zeros(n_nodes, out_feats, device=device)
            Z_pred.scatter_(1, pred.unsqueeze(1), 1.0)
            
            if log_training:
                print(f"Added {noise_mask.sum().item()} noisy labels in iteration {iter_idx+1} "
                      f"(accuracy: {simulated_acc:.2f})")
        
        else:
            if iter_idx == 0:
                # Create model using the specified model type
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
            elif iter_idx == 1:
                # Create model using the specified model type with selective parameters
                model = create_model(
                    model_type=model_type,
                    in_feats=in_feats,
                    h_feats=h_feats_selective,
                    out_feats=out_feats,
                    n_layers=n_layers_selective,
                    dropout_p=dropout_p_selective,
                    do_residual_connections=do_residual_connections,
                    do_hp=do_hp,
                    device=device
                )

                _, _, _, model = train(
                    g_rewired.to(device),
                    model,
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
            
                
            # Get predicted class distribution
            model.eval()
            with torch.no_grad():
                logits = model(g_rewired.to(device), feat.to(device))
                # Create corresponding Z_pred (one-hot encoding)
                pred = logits.argmax(dim=1)  
                Z_pred = torch.zeros(n_nodes, out_feats, device=device)
                Z_pred.scatter_(1, pred.unsqueeze(1), 1.0)
        
        train_acc = calculate_accuracy(pred, labels, train_mask, device)
        val_acc = calculate_accuracy(pred, labels, val_mask, device)
        test_acc = calculate_accuracy(pred, labels, test_mask, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_iter_idx = iter_idx

        # Ensure no empty classes: if any class is empty, fill it artificially
        unique_counts = pred.bincount(minlength=out_feats)
        empty_classes = torch.where(unique_counts == 0)[0]
        for i, empty_cls in enumerate(empty_classes):
            pred[i] = empty_cls
        
        ########################################################################
        # 3) Compute B_opt from predicted classes and rewire graph
        ########################################################################

        pi = Z_pred.cpu().numpy().sum(0) / n_nodes
        Pi_inv = np.diag(1/(pi+1e-8))
        B_opt = (d_out/k) * Pi_inv @ P_k @ Pi_inv
        B_opt_tensor = torch.tensor(B_opt, dtype=torch.float32, device=device)
        
        A_old = g_rewired.adj().to_dense().cpu()
        
        # Create rewired graph for this iteration
        # if not check_symmetry(g_rewired):
        #     g_rewired = create_rewired_graph(
        #         g=g_rewired.to(device),
        #         B_opt_tensor=B_opt_tensor.to(device),
        #         pred=pred.to(device),
        #         Z_pred=Z_pred,
        #         p_add=p_add,
        #         p_remove=p_remove,
        #         device=device,
        #         sym_type='asymetric'
        #     )
        # else:
        g_rewired = create_rewired_graph(
            g=g_rewired.to(device),
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
        
        # Compute statistics for current rewiring
        A_new = g_rewired.adj().to_dense().cpu()
        edges_added = ((A_new > 0.5) & (A_old < 0.5)).sum().item()
        edges_removed = ((A_new < 0.5) & (A_old > 0.5)).sum().item()
        
        current_stats = compute_graph_stats(g_rewired, n_layers_selective + 1)
        current_stats.update({
            'edges_added': edges_added,
            'edges_removed': edges_removed,
            'iteration': iter_idx + 1,
            'train_accuracy': copy.deepcopy(train_acc),
            'test_accuracy': copy.deepcopy(test_acc),
            'val_accuracy': copy.deepcopy(val_acc),
        })
        
        rewiring_history.append(copy.deepcopy(current_stats))
        train_acc_list.append(copy.deepcopy(train_acc))
        val_acc_list.append(copy.deepcopy(val_acc))
        test_acc_list.append(copy.deepcopy(test_acc))
        
        if log_training:
            print(f"Iteration {iter_idx+1} Stats ({model_type}): Test Accuracy: {test_acc:.4f}, "
                f"Mean Homophily: {current_stats['mean_local_homophily']:.4f}, "
                f"Edges: {current_stats['num_edges']}, "
                f"Added: {edges_added}, Removed: {edges_removed}")
            
        
    

    # Compile results
    results = {
        'cold_start': {
            'train_acc': train_acc_cold,
            'val_acc': val_acc_cold,
            'test_acc': test_acc_cold,
        },
        'selective': {
            'train_acc': best_train_acc,
            'val_acc': best_val_acc,
            'test_acc': best_test_acc,
            'best_iter': best_iter_idx + 1,  # +1 to match human
        },
        'original_stats': original_stats,
        'rewired_stats': rewiring_history[best_iter_idx],  # Use stats from the best iteration
        'rewiring_history': rewiring_history,
        'model_type': model_type,  # Add model type to results
        'train_acc_list': train_acc_list,
        'val_acc_list': val_acc_list,
        'test_acc_list': test_acc_list,
    }
    return results


def run_iterative_bridge_experiment(
    g: dgl.DGLGraph,
    P_k: np.ndarray,
    model_type: str = 'GCN',  # New parameter for model type
    h_feats_mpnn: int = 64,
    n_layers_mpnn: int = 2,
    dropout_p_mpnn: float = 0.5,
    model_lr_mpnn: float = 1e-3,
    wd_mpnn: float = 0.0,
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
    do_residual_connections: bool = False,
    use_sgc: bool = True,
    n_rewire: int = 10,
    sgc_K: int = 2,
    sgc_lr: float = 1e-2,
    sgc_wd: float = 1e-4,
    simulated_acc: Optional[float] = None
    
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run the iterative rewiring pipeline multiple times and average the results.
    
    This function runs the iterative rewiring pipeline for multiple trials or data splits
    and computes the mean performance and confidence intervals.
    
    Args:
        g: Input graph
        P_k: Permutation matrix for rewiring
        model_type: Type of model to use ('GCN', 'GAT', 'GIN', 'GraphSAGE')
        h_feats_mpnn: Hidden feature dimension for the base model
        n_layers_mpnn: Number of hidden layers for the base model
        dropout_p_mpnn: Dropout probability for the base model
        model_lr_mpnn: Learning rate for the base model
        wd_mpnn: Weight decay for the base model
        h_feats_selective: Hidden feature dimension for the selective model
        n_layers_selective: Number of hidden layers for the selective model
        dropout_p_selective: Dropout probability for the selective model
        model_lr_selective: Learning rate for the selective model
        wd_selective: Weight decay for the selective model
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
        use_sgc: Whether to use SGC for classification (faster) or standard model
        n_rewire: Number of rewiring iterations
        sgc_K: Number of propagation steps for SGC
        sgc_lr: Learning rate specifically for the SGC model in first iteration
        sgc_wd: Weight decay specifically for the SGC model in first iteration
        simulated_acc: Optional float between 0 and 1 representing the accuracy of simulated predictions. 
                                   If provided, skips model training and uses noisy ground truth labels instead.
                                   E.g., 0.8 means 80% of predictions are correct, 20% are random noise.
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
            
        results = run_iterative_bridge_pipeline(
            g,
            P_k=P_k,
            model_type=model_type,  # Pass model type
            h_feats_mpnn=h_feats_mpnn,
            n_layers_mpnn=n_layers_mpnn,
            dropout_p_mpnn=dropout_p_mpnn,
            model_lr_mpnn=model_lr_mpnn,
            wd_mpnn=wd_mpnn,
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
            do_residual_connections=do_residual_connections,
            use_sgc=use_sgc,
            n_rewire=n_rewire,
            sgc_K=sgc_K,
            sgc_lr=sgc_lr,
            sgc_wd=sgc_wd,
            simulated_acc=simulated_acc
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
    
    # Format the output
    formatted_stats = {
        'model_type': model_type,  # Add model type to results
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