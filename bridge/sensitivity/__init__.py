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

from .models import (
    LinearGCN,
    TwoLayerGCN,
    FNN,
    TwoLayerFNN,
    FeedForwardNN,
    VanillaGCN,
    InitialResidualGCN,
    GCNII,
    H2GCN,
    create_sensitivity_model,
    create_fnn_baseline_model,
    normalize_backbone_type,
)
from .sensitivity import (
    estimate_sensitivity_analytic,
    compute_jacobian,
    estimate_sensitivity_autograd
)
from .snr import (
    estimate_snr_monte_carlo,
    estimate_snr_theorem,
    estimate_snr_theorem_autograd,
    estimate_snr_from_sensitivities,
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
    plot_local_sensitivity_validation, plot_snr_ratio_analysis, plot_bottlenecking_snr_scatter, plot_graph_wide_snr_validation
)
from .paper_plots import (
    generate_paper_sensitivity_plots,
    generate_paper_sensitivity_plots_from_frames,
    load_sensitivity_results,
    average_node_iterations,
    average_graph_iterations,
    plot_paper_condition_improvement_boxplots,
)

__all__ = [
    # Models
    'LinearGCN', 'TwoLayerGCN', 'FNN', 'TwoLayerFNN',
    'FeedForwardNN', 'VanillaGCN', 'InitialResidualGCN', 'GCNII', 'H2GCN',
    'create_sensitivity_model', 'create_fnn_baseline_model', 'normalize_backbone_type',
    
    # SNR estimation
    'estimate_snr_monte_carlo', 'estimate_snr_theorem', 'estimate_snr_theorem_autograd',
    'estimate_snr_from_sensitivities',
    
    # Sensitivity analysis
    'estimate_sensitivity_analytic', 'compute_jacobian', 'estimate_sensitivity_autograd',
    
    # Feature generation
    'generate_features', 'create_feature_generator',
    
    # Utilities
    'homophily', 'train_model', 'evaluate_model',
    'run_sensitivity_experiment', 'run_multi_graph_experiment',
    
    # Visualization
    'plot_local_sensitivity_validation', 'plot_snr_ratio_analysis',
    'plot_bottlenecking_snr_scatter', 'plot_graph_wide_snr_validation',
    'generate_paper_sensitivity_plots', 'generate_paper_sensitivity_plots_from_frames',
    'load_sensitivity_results', 'average_node_iterations', 'average_graph_iterations',
    'plot_paper_condition_improvement_boxplots'
]
