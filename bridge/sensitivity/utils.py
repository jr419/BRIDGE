"""
Utility functions for sensitivity and SNR analysis experiments.

This module provides helper functions for running experiments,
computing statistics, and working with graph data.
"""

import torch
import torch.nn as nn
import dgl
import numpy as np
from typing import Dict, Tuple, List, Union, Optional, Any, Callable
from tqdm import tqdm

from .snr import estimate_snr_monte_carlo, estimate_snr_theorem_autograd


def homophily(g: dgl.DGLGraph) -> float:
    """
    Compute edge homophily of a graph.
    
    Edge homophily is defined as the fraction of edges in the graph
    that connect nodes of the same class.
    
    Args:
        g: Input graph with node labels in g.ndata['label']
        
    Returns:
        The edge homophily score (between 0 and 1)
    """
    A = g.adjacency_matrix().to_dense()
    y = g.ndata['label']
    h = sum([(A[y==i][:,y==i]).sum() for i in range(g.ndata['label'].max()+1)])/(A.sum())
    return h.item()


def create_train_test_split(g: dgl.DGLGraph, train_ratio: float = 0.6) -> None:
    """
    Create train/test split for a graph.
    
    This function adds 'train_mask' and 'test_mask' to g.ndata.
    
    Args:
        g: The input graph
        train_ratio: Fraction of nodes to use for training
    """
    num_nodes = g.num_nodes()
    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    g.ndata['train_mask'] = train_mask
    g.ndata['test_mask'] = test_mask


def train_model(
    model: nn.Module,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    n_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 1e-3,
    verbose: bool = False
) -> nn.Module:
    """
    Train a GNN model on the given graph and features.
    
    Args:
        model: The neural network model to train
        graph: The input graph
        features: Node features
        labels: Node labels
        train_mask: Boolean mask for training nodes
        n_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        verbose: Whether to print training progress
        
    Returns:
        The trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in (range(n_epochs) if not verbose else tqdm(range(n_epochs), desc="Training")):
        optimizer.zero_grad()
        logits = model(graph, features)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    return model


def evaluate_model(
    model: nn.Module,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[float, float]:
    """
    Evaluate a GNN model on the given graph and features.
    
    Args:
        model: The neural network model to evaluate
        graph: The input graph
        features: Node features
        labels: Node labels
        mask: Boolean mask for nodes to evaluate
        
    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        logits = model(graph, features)
        predictions = logits[mask].argmax(dim=1)
        correct = (predictions == labels[mask]).sum().item()
        accuracy = correct / mask.sum().item()
        loss = criterion(logits[mask], labels[mask]).item()
    
    return accuracy, loss


def node_level_evaluate(
    model: nn.Module,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate a GNN model on the given graph and return node-level accuracy.
    
    Args:
        model: The neural network model to evaluate
        graph: The input graph
        features: Node features
        labels: Node labels
        
    Returns:
        Binary tensor indicating correct (1) or incorrect (0) prediction for each node
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(graph, features)
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels).float()
    
    return correct


def get_sample_statistics(
    values: List[float], 
    remove_nan: bool = True
) -> Tuple[float, float]:
    """
    Compute mean and standard deviation from a list of values.
    
    Args:
        values: List of values
        remove_nan: Whether to remove NaN values before computing statistics
        
    Returns:
        Tuple of (mean, std)
    """
    if remove_nan:
        values = [v for v in values if not np.isnan(v)]
    
    if not values:
        return 0.0, 0.0
    
    return np.mean(values), np.std(values)


def append_or_mean(lst: List[float], value: float) -> None:
    """
    Append a value to a list, or the mean of the list if value is NaN.
    
    Args:
        lst: List to append to
        value: Value to append (or NaN)
    """
    if not np.isnan(value):
        lst.append(value)
    else:
        if lst:
            lst.append(np.mean(lst))
        else:
            lst.append(0.0)


def run_sensitivity_experiment(
    model: nn.Module,
    graph: dgl.DGLGraph,
    feature_generator: Callable,
    in_feats: int,
    num_acc_repeats: int = 100,
    num_monte_carlo_samples: int = 100,
    num_epochs: int = 200,
    lr: float = 0.01, 
    weight_decay: float = 1e-3,
    sigma_intra: Optional[torch.Tensor] = None,
    sigma_inter: Optional[torch.Tensor] = None,
    tau: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    device: str = "cuda",
    do_mean: bool = True
) -> Dict[str, Any]:
    """
    Run a comprehensive sensitivity analysis experiment.
    
    This function:
    1. Trains a model multiple times with different feature realizations
    2. Estimates SNR using Monte Carlo and theorem-based approaches
    3. Computes accuracy and other metrics
    
    Args:
        model: The neural network model to evaluate
        graph: The input graph
        feature_generator: Function to generate features
        in_feats: Number of input features
        num_acc_repeats: Number of training repetitions for accuracy estimation
        num_monte_carlo_samples: Number of samples for Monte Carlo SNR estimation
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        sigma_intra: Intra-class covariance matrix (for theorem-based SNR)
        sigma_inter: Inter-class covariance matrix (for theorem-based SNR)
        tau: Global shift covariance matrix (for theorem-based SNR)
        eta: Noise covariance matrix (for theorem-based SNR)
        device: Device to compute on
        do_mean: Whether to return node-averaged metrics (True) or node-level metrics (False)
        
    Returns:
        Dictionary with experiment results:
        - estimated_snr_mc: Monte Carlo SNR estimate
        - estimated_snr_theorem: Theorem-based SNR estimate
        - mean_test_acc: Mean test accuracy
        - mean_test_loss: Mean test loss
        - homophily: Graph homophily
    """
    labels = graph.ndata['label']
    num_nodes = graph.num_nodes()
    num_classes = len(torch.unique(labels))
    
    # Create train/test split if not present
    if 'train_mask' not in graph.ndata or 'test_mask' not in graph.ndata:
        create_train_test_split(graph)
    
    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']
    
    # Prepare accumulators
    if do_mean:
        mean_test_acc = 0.0
        mean_test_loss = 0.0
    else:
        mean_test_acc = torch.zeros(num_nodes)
        mean_test_loss = 0.0
    
    # Run multiple training iterations
    for _ in tqdm(range(num_acc_repeats), desc="Training repetitions"):
        # Generate a new random feature set for training
        features = feature_generator(num_nodes, in_feats, labels, num_mu_samples=1)[:,:,0]
        
        # Initialize a new model instance
        model_instance = type(model)(in_feats, *model.weight1.shape).double()
        
        # Train the model
        train_model(model_instance, graph, features, labels, train_mask, num_epochs, lr, weight_decay)
        
        # Evaluate on test set
        if do_mean:
            test_acc, test_loss = evaluate_model(model_instance, graph, features, labels, test_mask)
            mean_test_acc += test_acc
            mean_test_loss += test_loss
        else:
            node_correct = node_level_evaluate(model_instance, graph, features, labels)
            mean_test_acc += node_correct
            _, test_loss = evaluate_model(model_instance, graph, features, labels, test_mask)
            mean_test_loss += test_loss
    
    # Average the results
    mean_test_acc = mean_test_acc / num_acc_repeats
    mean_test_loss = mean_test_loss / num_acc_repeats
    
    # Estimate SNR using Monte Carlo
    model_instance.eval()
    estimated_snr_mc = estimate_snr_monte_carlo(
        model_instance, graph, in_feats, labels,
        num_montecarlo_simulations=num_monte_carlo_samples,
        feature_generator=feature_generator,
        device=device,
        inner_samples=num_monte_carlo_samples
    )
    
    # Estimate SNR using theorem if covariance matrices are provided
    if all(x is not None for x in [sigma_intra, sigma_inter, tau, eta]):
        estimated_snr_theorem = estimate_snr_theorem_autograd(
            model_instance, graph, in_feats, labels,
            sigma_intra, sigma_inter, tau, eta,
            device=device
        )
    else:
        # Placeholder if covariance matrices aren't provided
        if do_mean:
            estimated_snr_theorem = torch.tensor(0.0)
        else:
            estimated_snr_theorem = torch.zeros_like(estimated_snr_mc)
    
    # Compute graph homophily
    graph_homophily = homophily(graph)
    
    if do_mean:
        # Average SNR across nodes
        estimated_snr_mc = torch.mean(estimated_snr_mc)
        estimated_snr_theorem = torch.mean(estimated_snr_theorem)
    
    return {
        "estimated_snr_mc": estimated_snr_mc,
        "estimated_snr_theorem": estimated_snr_theorem,
        "mean_test_acc": mean_test_acc,
        "mean_test_loss": mean_test_loss,
        "homophily": graph_homophily
    }


def run_multi_graph_experiment(
    graph_generator: Callable,
    model_constructor: Callable,
    feature_generator: Callable,
    in_feats: int,
    num_nodes: int,
    num_classes: int,
    homophily_values: List[float],
    mean_degree: int = 10,
    num_samples: int = 5,
    **experiment_kwargs
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Run sensitivity analysis on multiple graphs with varying homophily.
    
    Args:
        graph_generator: Function to generate a graph given parameters
        model_constructor: Function to construct a model given in_feats
        feature_generator: Function to generate features
        in_feats: Number of input features
        num_nodes: Number of nodes in generated graphs
        num_classes: Number of classes in generated graphs
        homophily_values: List of homophily values to test
        mean_degree: Mean degree for generated graphs
        num_samples: Number of graph samples per homophily value
        experiment_kwargs: Additional arguments for run_sensitivity_experiment
        
    Returns:
        Dictionary with lists of (mean, std) tuples for each metric:
        - estimated_snr_mc_list: Monte Carlo SNR estimates
        - estimated_snr_theorem_val_list: Theorem-based SNR estimates
        - acc_list: Test accuracy values
        - loss_list: Test loss values
        - homophily_list: Actual homophily values
    """
    # Initialize result lists
    estimated_snr_mc_list = []
    estimated_snr_theorem_val_list = []
    acc_list = []
    loss_list = []
    homophily_list = []
    
    # For each homophily value
    for h in tqdm(homophily_values, desc="Homophily values"):
        # Lists to store results for this homophily
        estimated_snr_mc_samples = []
        estimated_snr_theorem_samples = []
        acc_samples = []
        loss_samples = []
        homophily_samples = []
        
        # Generate and analyze multiple graph samples
        for i in range(num_samples):
            print(f"Running sample {i+1}/{num_samples} for homophily {h:.2f}")
            print('='*40+'\n')
            # Generate graph
            g = graph_generator(num_nodes=num_nodes, num_classes=num_classes, 
                             homophily=h, mean_degree=mean_degree)
            
            # Make sure there are no self-loops
            g = dgl.remove_self_loop(g)
            #g = dgl.add_self_loop(g)
            
            # Create model
            model = model_constructor(in_feats=in_feats, num_classes=num_classes)
            
            # Run experiment
            results = run_sensitivity_experiment(
                model=model,
                graph=g,
                feature_generator=feature_generator,
                in_feats=in_feats,
                **experiment_kwargs
            )
            
            # Extract results
            # Convert pytorch tensors to numpy
            estimated_snr_mc = results["estimated_snr_mc"].cpu().numpy()
            estimated_snr_theorem = results["estimated_snr_theorem"].cpu().numpy()
            mean_test_acc = results["mean_test_acc"]
            mean_test_loss = results["mean_test_loss"]
            graph_homophily = results["homophily"]
            
            # Store results, handling NaN values
            append_or_mean(estimated_snr_mc_samples, estimated_snr_mc)
            append_or_mean(estimated_snr_theorem_samples, estimated_snr_theorem)
            append_or_mean(acc_samples, mean_test_acc)
            append_or_mean(loss_samples, mean_test_loss)
            append_or_mean(homophily_samples, graph_homophily)
        
        # Compute statistics for this homophily value
        estimated_snr_mc_list.append(get_sample_statistics(estimated_snr_mc_samples))
        estimated_snr_theorem_val_list.append(get_sample_statistics(estimated_snr_theorem_samples))
        acc_list.append(get_sample_statistics(acc_samples))
        loss_list.append(get_sample_statistics(loss_samples))
        homophily_list.append(get_sample_statistics(homophily_samples))
    
    return {
        "estimated_snr_mc_list": estimated_snr_mc_list,
        "estimated_snr_theorem_val_list": estimated_snr_theorem_val_list,
        "acc_list": acc_list,
        "loss_list": loss_list,
        "homophily_list": homophily_list
    }
