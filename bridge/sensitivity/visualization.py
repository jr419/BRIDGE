"""
Visualization utilities for sensitivity analysis.

This module provides functions for visualizing sensitivity analysis
results, including SNR and accuracy across different graph properties.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def compute_sensitivity_threshold(
    homophily_values: np.ndarray,
    snr_theorem_values: np.ndarray,
    snr_fnn_threshold: float,
    method: str = 'crossing'
) -> Tuple[float, float]:
    """
    Compute the homophily threshold where SNR crosses the FNN threshold.
    
    Args:
        homophily_values: Array of homophily values
        snr_theorem_values: Array of SNR values from theorem
        snr_fnn_threshold: SNR threshold of FNN to compare against
        method: Method to determine threshold ('crossing' or 'condition')
        
    Returns:
        Tuple of (min_threshold, max_threshold)
    """
    if method == 'crossing':
        # Find where SNR crosses the threshold
        above_threshold = snr_theorem_values > snr_fnn_threshold
        
        # Find transition points (from below to above or vice versa)
        transitions = np.where(np.diff(above_threshold.astype(int)))[0]
        
        if len(transitions) >= 2:
            min_idx, max_idx = transitions[0], transitions[-1]
            return homophily_values[min_idx], homophily_values[max_idx + 1]
        elif len(transitions) == 1:
            idx = transitions[0]
            return homophily_values[idx], homophily_values[idx + 1]
    
    # Default: return full range
    return homophily_values[0], homophily_values[-1]


def plot_snr_vs_homophily(
    homophily_values: np.ndarray,
    snr_mc_means: np.ndarray,
    snr_mc_stds: np.ndarray,
    snr_theorem_means: np.ndarray,
    snr_theorem_stds: np.ndarray,
    accuracy_means: np.ndarray,
    accuracy_stds: np.ndarray,
    fnn_acc_mean: float = 0.5,
    fnn_acc_std: float = 0.05,
    snr_fnn_threshold: Optional[float] = None,
    title: str = 'SNR and Test Accuracy vs Edge Homophily',
    factor_std: float = 0.5,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a plot showing SNR and accuracy against homophily.
    
    Args:
        homophily_values: Array of homophily values
        snr_mc_means: Array of means for Monte Carlo SNR
        snr_mc_stds: Array of standard deviations for Monte Carlo SNR
        snr_theorem_means: Array of means for theorem-based SNR
        snr_theorem_stds: Array of standard deviations for theorem-based SNR
        accuracy_means: Array of means for test accuracy
        accuracy_stds: Array of standard deviations for test accuracy
        fnn_acc_mean: Mean accuracy of FNN (for horizontal line)
        fnn_acc_std: Standard deviation of FNN accuracy
        snr_fnn_threshold: SNR threshold of FNN to compare against
        title: Plot title
        factor_std: Factor to multiply standard deviations for confidence bands
        figsize: Figure size (width, height)
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        The matplotlib Figure object
    """
    # Sort by homophily if needed
    if not np.all(np.diff(homophily_values) >= 0):
        argsort_inds = np.argsort(homophily_values)
        homophily_values = homophily_values[argsort_inds]
        snr_mc_means = snr_mc_means[argsort_inds]
        snr_mc_stds = snr_mc_stds[argsort_inds]
        snr_theorem_means = snr_theorem_means[argsort_inds]
        snr_theorem_stds = snr_theorem_stds[argsort_inds]
        accuracy_means = accuracy_means[argsort_inds]
        accuracy_stds = accuracy_stds[argsort_inds]
    
    # Plot setup
    plt.rcParams.update({'font.size': 12})
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Colors
    color_mc = 'tab:blue'
    color_theorem = 'tab:orange'
    color_acc = 'tab:green'
    
    # Left y-axis: SNR
    ax1.set_xlabel('Edge Homophily')
    ax1.set_ylabel('SNR Estimate', color=color_mc)
    
    # Plot Monte Carlo SNR
    ln1, = ax1.plot(
        homophily_values, snr_mc_means,
        label='Monte Carlo SNR', color=color_mc, marker='o', linestyle='-'
    )
    ax1.fill_between(
        homophily_values,
        snr_mc_means - factor_std*snr_mc_stds,
        snr_mc_means + factor_std*snr_mc_stds,
        color=color_mc, alpha=0.2
    )
    
    # Plot Theorem-based SNR
    ln2, = ax1.plot(
        homophily_values, snr_theorem_means,
        label='Sensitivity-based SNR', color=color_theorem, marker='s', linestyle='-'
    )
    ax1.fill_between(
        homophily_values,
        snr_theorem_means - factor_std*snr_theorem_stds,
        snr_theorem_means + factor_std*snr_theorem_stds,
        color=color_theorem, alpha=0.2
    )
    
    ax1.tick_params(axis='y', labelcolor=color_mc)
    
    # Right y-axis: Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color=color_acc)
    
    # Plot GNN accuracy
    ln3, = ax2.plot(
        homophily_values, accuracy_means,
        label='GNN Test Acc', color=color_acc, linestyle='-', marker='d'
    )
    ax2.fill_between(
        homophily_values,
        accuracy_means - factor_std*accuracy_stds,
        accuracy_means + factor_std*accuracy_stds,
        color=color_acc, alpha=0.15
    )
    
    # Plot FNN accuracy as horizontal line
    ln4 = ax2.axhline(
        y=fnn_acc_mean, color='green', linestyle='--', linewidth=1, 
        label='FNN Test Accuracy'
    )
    ax2.fill_between(
        [0, 1],
        fnn_acc_mean - factor_std*fnn_acc_std, 
        fnn_acc_mean + factor_std*fnn_acc_std,
        color='green', alpha=0.15, label='FNN ±1 STD'
    )
    
    # Add threshold lines if provided
    if snr_fnn_threshold is not None:
        v_min, v_max = compute_sensitivity_threshold(
            homophily_values, snr_theorem_means, snr_fnn_threshold
        )
        ln5 = ax1.axvline(
            x=v_min, color='red', linestyle='-.', linewidth=1,
            label='Predicted GNN/FNN Threshold'
        )
        ax1.axvline(x=v_max, color='red', linestyle='-.', linewidth=1)
        
        # Add to legends list
        lines = [ln1, ln2, ln3, ln4, ln5]
    else:
        lines = [ln1, ln2, ln3, ln4]
    
    # Combine legends
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best', fontsize=10)
    
    # Title and grid
    ax2.tick_params(axis='y', labelcolor=color_acc)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sensitivity_vs_graph_property(
    property_values: np.ndarray,
    signal_sensitivity: np.ndarray,
    noise_sensitivity: np.ndarray,
    global_sensitivity: np.ndarray,
    property_name: str = 'Homophily',
    title: str = 'Sensitivity vs Graph Property',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot different sensitivity types against a graph property.
    
    Args:
        property_values: Array of graph property values (e.g., homophily)
        signal_sensitivity: Array of signal sensitivity values
        noise_sensitivity: Array of noise sensitivity values
        global_sensitivity: Array of global sensitivity values
        property_name: Name of the graph property
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        The matplotlib Figure object
    """
    # Sort by property if needed
    if not np.all(np.diff(property_values) >= 0):
        argsort_inds = np.argsort(property_values)
        property_values = property_values[argsort_inds]
        signal_sensitivity = signal_sensitivity[argsort_inds]
        noise_sensitivity = noise_sensitivity[argsort_inds]
        global_sensitivity = global_sensitivity[argsort_inds]
    
    # Plot setup
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors
    color_signal = 'tab:blue'
    color_noise = 'tab:red'
    color_global = 'tab:orange'
    
    # Plot sensitivities
    ax.plot(
        property_values, signal_sensitivity,
        label='Signal Sensitivity', color=color_signal, marker='o', linestyle='-'
    )
    ax.plot(
        property_values, noise_sensitivity,
        label='Noise Sensitivity', color=color_noise, marker='s', linestyle='-'
    )
    ax.plot(
        property_values, global_sensitivity,
        label='Global Sensitivity', color=color_global, marker='^', linestyle='-'
    )
    
    # Add labels and legend
    ax.set_xlabel(property_name)
    ax.set_ylabel('Sensitivity')
    ax.legend(loc='best')
    
    # Title and grid
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_node_level_snr(
    graph: Union['dgl.DGLGraph', 'nx.Graph'],
    snr_values: np.ndarray,
    node_positions: Optional[Dict] = None,
    title: str = 'Node-level SNR',
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot node-level SNR values on a graph.
    
    Args:
        graph: A DGL or NetworkX graph
        snr_values: Array of SNR values for each node
        node_positions: Dictionary of node positions (if None, layout is computed)
        title: Plot title
        cmap: Colormap for SNR values
        figsize: Figure size (width, height)
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        The matplotlib Figure object
    """
    try:
        import networkx as nx
        
        # Convert DGL graph to NetworkX if needed
        if hasattr(graph, 'to_networkx'):
            nx_graph = graph.to_networkx()
        else:
            nx_graph = graph
        
        # Compute layout if not provided
        if node_positions is None:
            node_positions = nx.spring_layout(nx_graph)
        
        # Plot setup
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw nodes with SNR-based colors
        nodes = nx.draw_networkx_nodes(
            nx_graph, node_positions, node_size=80,
            node_color=snr_values, cmap=plt.get_cmap(cmap),
            ax=ax
        )
        
        # Add colorbar
        plt.colorbar(nodes, ax=ax, label='SNR')
        
        # Draw edges
        nx.draw_networkx_edges(nx_graph, node_positions, alpha=0.2, ax=ax)
        
        # Title and layout
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    except ImportError:
        print("NetworkX is required for graph visualization. Please install with 'pip install networkx'.")
        return None
