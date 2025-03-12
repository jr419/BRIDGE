"""
BRIDGE Sensitivity Analysis Package

This package provides tools for analyzing the signal-to-noise ratio and sensitivity
of graph neural networks. It offers methods to understand when and how graph structure
affects model performance, based on the paper:

"The Limits of MPNNs: How Homophilic Bottlenecks Restrict the Signal-to-Noise Ratio in Message Passing"

Main components:
- Models: Linear and nonlinear GNN implementations
- SNR estimation: Methods to estimate signal-to-noise ratio
- Sensitivity analysis: Tools to analyze model sensitivity to different input perturbations
- Feature generation: Utilities for generating synthetic node features
- Visualization: Functions for visualizing sensitivity analysis results
"""

from .models import LinearGCN, TwoLayerGCN
from .snr import (
    estimate_snr_monte_carlo,
    estimate_snr_theorem,
    estimate_snr_theorem_autograd
)
from .sensitivity import (
    estimate_sensitivity_analytic,
    compute_jacobian,
    estimate_sensitivity_autograd
)
from .feature_gen import generate_features, create_feature_generator
from .utils import (
    homophily,
    train_model,
    evaluate_model,
    run_sensitivity_experiment,
    run_multi_graph_experiment
)
from .visualization import (
    plot_snr_vs_homophily,
    plot_sensitivity_vs_graph_property,
    plot_node_level_snr
)

__all__ = [
    # Models
    'LinearGCN', 'TwoLayerGCN',
    
    # SNR estimation
    'estimate_snr_monte_carlo', 'estimate_snr_theorem', 'estimate_snr_theorem_autograd',
    
    # Sensitivity analysis
    'estimate_sensitivity_analytic', 'compute_jacobian', 'estimate_sensitivity_autograd',
    
    # Feature generation
    'generate_features', 'create_feature_generator',
    
    # Utilities
    'homophily', 'train_model', 'evaluate_model',
    'run_sensitivity_experiment', 'run_multi_graph_experiment',
    
    # Visualization
    'plot_snr_vs_homophily', 'plot_sensitivity_vs_graph_property', 'plot_node_level_snr'
]
