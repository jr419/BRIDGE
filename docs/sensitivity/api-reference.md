---
layout: default
title: API Reference
parent: Sensitivity Analysis
nav_order: 1
---

# Sensitivity Analysis API Reference
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Models

### LinearGCN

```python
class LinearGCN(nn.Module)
```

A simple linear GCN layer implementing the operation h = Â @ x @ W, where Â is the normalized adjacency matrix.

**Parameters:**
- `in_feats` (int): Input feature dimension
- `hidden_feats` (int): Hidden feature dimension (unused in this implementation)
- `out_feats` (int): Output feature dimension

**Methods:**
- `forward(graph, features)`: Forward pass for the LinearGCN

### TwoLayerGCN

```python
class TwoLayerGCN(nn.Module)
```

A two-layer GCN with ReLU activation between layers.

**Parameters:**
- `in_feats` (int): Input feature dimension
- `hidden_feats` (int): Hidden feature dimension
- `out_feats` (int): Output feature dimension

**Methods:**
- `forward(graph, features)`: Forward pass for the TwoLayerGCN

## SNR Estimation

### estimate_snr_monte_carlo

```python
def estimate_snr_monte_carlo(model, graph, in_feats, labels, num_montecarlo_simulations, 
                             feature_generator, device="cpu", inner_samples=100, 
                             split_model_input_size=100)
```

Estimate the Signal-to-Noise Ratio (SNR) of an MPNN's outputs via Monte Carlo simulation.

**Parameters:**
- `model` (nn.Module): The neural network model
- `graph` (dgl.DGLGraph): The input graph
- `in_feats` (int): Number of input features
- `labels` (torch.Tensor): Node labels
- `num_montecarlo_simulations` (int): Number of outer loop iterations
- `feature_generator` (Callable): Function to generate features
- `device` (str): Device to compute on
- `inner_samples` (int): Number of feature samples for each mean vector
- `split_model_input_size` (int): Maximum batch size for processing samples

**Returns:**
- torch.Tensor: The estimated SNR for each node and output feature

### estimate_snr_theorem

```python
def estimate_snr_theorem(model, graph, labels, sigma_intra, sigma_inter, tau, eta)
```

Compute SNR estimate using the theoretical formula from the paper.

**Parameters:**
- `model` (nn.Module): The neural network model (must be LinearGCN)
- `graph` (dgl.DGLGraph): The input graph
- `labels` (torch.Tensor): Node labels
- `sigma_intra` (torch.Tensor): Intra-class covariance matrix
- `sigma_inter` (torch.Tensor): Inter-class covariance matrix
- `tau` (torch.Tensor): Global shift covariance matrix
- `eta` (torch.Tensor): Noise covariance matrix

**Returns:**
- torch.Tensor: The estimated SNR for each node (averaged across output features)

### estimate_snr_theorem_autograd

```python
def estimate_snr_theorem_autograd(model, graph, in_feats, labels, sigma_intra, 
                                 sigma_inter, tau, eta, device="cuda")
```

Compute SNR estimate using the theoretical formula with autograd sensitivities.

**Parameters:**
- `model` (nn.Module): The neural network model (any model that works with autograd)
- `graph` (dgl.DGLGraph): The input graph
- `in_feats` (int): Number of input features
- `labels` (torch.Tensor): Node labels
- `sigma_intra` (torch.Tensor): Intra-class covariance matrix
- `sigma_inter` (torch.Tensor): Inter-class covariance matrix
- `tau` (torch.Tensor): Global shift covariance matrix
- `eta` (torch.Tensor): Noise covariance matrix
- `device` (str): Device to compute on

**Returns:**
- torch.Tensor: The estimated SNR for each node (averaged across output features)

## Sensitivity Analysis

### estimate_sensitivity_analytic

```python
def estimate_sensitivity_analytic(model, graph, labels, sensitivity_type)
```

Estimate the sensitivity for a *Linear* GCN analytically (no autograd).

**Parameters:**
- `model` (nn.Module): A linear GCN model with a weight attribute
- `graph` (dgl.DGLGraph): The input graph
- `labels` (torch.Tensor): Node labels (used for signal sensitivity)
- `sensitivity_type` (str): Type of sensitivity to compute ("signal", "noise", or "global")

**Returns:**
- torch.Tensor: A sensitivity tensor of shape [num_nodes, num_classes, in_feats, in_feats]

### compute_jacobian

```python
def compute_jacobian(model, graph, x, device="cuda")
```

Compute the Jacobian matrix of the model with respect to the input.

**Parameters:**
- `model` (nn.Module): The neural network model
- `graph` (dgl.DGLGraph): The input graph
- `x` (torch.Tensor): Input features of shape (N, in_feats)
- `device` (str): Device to compute on

**Returns:**
- torch.Tensor: A tensor of shape (N, out_feats, N, in_feats) containing the Jacobian

### estimate_sensitivity_autograd

```python
def estimate_sensitivity_autograd(model, graph, in_feats, labels, sensitivity_type, device="cuda")
```

Estimate sensitivity using autograd-computed Jacobian.

**Parameters:**
- `model` (nn.Module): The neural network model
- `graph` (dgl.DGLGraph): The input graph
- `in_feats` (int): Number of input features
- `labels` (torch.Tensor): Node labels (used for signal sensitivity)
- `sensitivity_type` (str): Type of sensitivity to compute ("signal", "noise", or "global")
- `device` (str): Device to compute on

**Returns:**
- torch.Tensor: A sensitivity tensor of shape [num_nodes, num_classes, in_feats, in_feats]

## Feature Generation

### generate_features

```python
def generate_features(num_nodes, num_features, labels, inter_class_cov, intra_class_cov, 
                      global_cov, noise_cov, mu_repeats=1)
```

Generate synthetic node features with controlled covariance structure.

**Parameters:**
- `num_nodes` (int): Number of nodes
- `num_features` (int): Number of feature dimensions
- `labels` (np.ndarray): Node class labels
- `inter_class_cov` (np.ndarray): Covariance matrix between different classes
- `intra_class_cov` (np.ndarray): Covariance matrix within the same class
- `global_cov` (np.ndarray): Covariance matrix for the global shift
- `noise_cov` (np.ndarray): Covariance matrix for the node-specific noise
- `mu_repeats` (int): Number of feature realizations to generate for each class mean

**Returns:**
- np.ndarray: A numpy array of shape (num_nodes, num_features, mu_repeats)

### create_feature_generator

```python
def create_feature_generator(sigma_intra, sigma_inter, tau, eta, dtype=torch.float64)
```

Create a feature generator function with fixed covariance parameters.

**Parameters:**
- `sigma_intra` (torch.Tensor): Intra-class covariance matrix
- `sigma_inter` (torch.Tensor): Inter-class covariance matrix
- `tau` (torch.Tensor): Global shift covariance matrix
- `eta` (torch.Tensor): Noise covariance matrix
- `dtype` (torch.dtype): Torch data type for the output tensor

**Returns:**
- Callable: A function that generates features with signature: feature_generator(num_nodes, in_feats, labels, num_mu_samples)

## Experiment Utilities

### train_model

```python
def train_model(model, graph, features, labels, train_mask, n_epochs=200, 
                lr=0.01, weight_decay=1e-3, verbose=False)
```

Train a GNN model on the given graph and features.

**Parameters:**
- `model` (nn.Module): The neural network model to train
- `graph` (dgl.DGLGraph): The input graph
- `features` (torch.Tensor): Node features
- `labels` (torch.Tensor): Node labels
- `train_mask` (torch.Tensor): Boolean mask for training nodes
- `n_epochs` (int): Number of training epochs
- `lr` (float): Learning rate
- `weight_decay` (float): Weight decay for regularization
- `verbose` (bool): Whether to print training progress

**Returns:**
- nn.Module: The trained model

### evaluate_model

```python
def evaluate_model(model, graph, features, labels, mask)
```

Evaluate a GNN model on the given graph and features.

**Parameters:**
- `model` (nn.Module): The neural network model to evaluate
- `graph` (dgl.DGLGraph): The input graph
- `features` (torch.Tensor): Node features
- `labels` (torch.Tensor): Node labels
- `mask` (torch.Tensor): Boolean mask for nodes to evaluate

**Returns:**
- Tuple[float, float]: Tuple of (accuracy, loss)

### run_sensitivity_experiment

```python
def run_sensitivity_experiment(model, graph, feature_generator, in_feats, 
                             num_acc_repeats=100, num_monte_carlo_samples=100, 
                             num_epochs=200, lr=0.01, weight_decay=1e-3, 
                             sigma_intra=None, sigma_inter=None, 
                             tau=None, eta=None, device="cuda", do_mean=True)
```

Run a comprehensive sensitivity analysis experiment.

**Parameters:**
- `model` (nn.Module): The neural network model to evaluate
- `graph` (dgl.DGLGraph): The input graph
- `feature_generator` (Callable): Function to generate features
- `in_feats` (int): Number of input features
- `num_acc_repeats` (int): Number of training repetitions for accuracy estimation
- `num_monte_carlo_samples` (int): Number of samples for Monte Carlo SNR estimation
- `num_epochs` (int): Number of training epochs
- `lr` (float): Learning rate
- `weight_decay` (float): Weight decay for regularization
- `sigma_intra` (torch.Tensor): Intra-class covariance matrix (for theorem-based SNR)
- `sigma_inter` (torch.Tensor): Inter-class covariance matrix (for theorem-based SNR)
- `tau` (torch.Tensor): Global shift covariance matrix (for theorem-based SNR)
- `eta` (torch.Tensor): Noise covariance matrix (for theorem-based SNR)
- `device` (str): Device to compute on
- `do_mean` (bool): Whether to return node-averaged metrics (True) or node-level metrics (False)

**Returns:**
- Dict[str, Any]: Dictionary with experiment results

### run_multi_graph_experiment

```python
def run_multi_graph_experiment(graph_generator, model_constructor, feature_generator, 
                               in_feats, num_nodes, num_classes, homophily_values, 
                               mean_degree=10, num_samples=5, **experiment_kwargs)
```

Run sensitivity analysis on multiple graphs with varying homophily.

**Parameters:**
- `graph_generator` (Callable): Function to generate a graph given parameters
- `model_constructor` (Callable): Function to construct a model given in_feats
- `feature_generator` (Callable): Function to generate features
- `in_feats` (int): Number of input features
- `num_nodes` (int): Number of nodes in generated graphs
- `num_classes` (int): Number of classes in generated graphs
- `homophily_values` (List[float]): List of homophily values to test
- `mean_degree` (int): Mean degree for generated graphs
- `num_samples` (int): Number of graph samples per homophily value
- `experiment_kwargs`: Additional arguments for run_sensitivity_experiment

**Returns:**
- Dict[str, List[Tuple[float, float]]]: Dictionary with lists of (mean, std) tuples for each metric

## Visualization

### plot_snr_vs_homophily

```python
def plot_snr_vs_homophily(homophily_values, snr_mc_means, snr_mc_stds, 
                         snr_theorem_means, snr_theorem_stds, 
                         accuracy_means, accuracy_stds, 
                         fnn_acc_mean=0.5, fnn_acc_std=0.05, 
                         snr_fnn_threshold=None, 
                         title='SNR and Test Accuracy vs Edge Homophily', 
                         factor_std=0.5, figsize=(10, 6), save_path=None)
```

Create a plot showing SNR and accuracy against homophily.

**Parameters:**
- `homophily_values` (np.ndarray): Array of homophily values
- `snr_mc_means` (np.ndarray): Array of means for Monte Carlo SNR
- `snr_mc_stds` (np.ndarray): Array of standard deviations for Monte Carlo SNR
- `snr_theorem_means` (np.ndarray): Array of means for theorem-based SNR
- `snr_theorem_stds` (np.ndarray): Array of standard deviations for theorem-based SNR
- `accuracy_means` (np.ndarray): Array of means for test accuracy
- `accuracy_stds` (np.ndarray): Array of standard deviations for test accuracy
- `fnn_acc_mean` (float): Mean accuracy of FNN (for horizontal line)
- `fnn_acc_std` (float): Standard deviation of FNN accuracy
- `snr_fnn_threshold` (float): SNR threshold of FNN to compare against
- `title` (str): Plot title
- `factor_std` (float): Factor to multiply standard deviations for confidence bands
- `figsize` (Tuple[int, int]): Figure size (width, height)
- `save_path` (str): Path to save the figure (if None, figure is not saved)

**Returns:**
- plt.Figure: The matplotlib Figure object

### plot_sensitivity_vs_graph_property

```python
def plot_sensitivity_vs_graph_property(property_values, signal_sensitivity, 
                                     noise_sensitivity, global_sensitivity, 
                                     property_name='Homophily', 
                                     title='Sensitivity vs Graph Property', 
                                     figsize=(10, 6), save_path=None)
```

Plot different sensitivity types against a graph property.

**Parameters:**
- `property_values` (np.ndarray): Array of graph property values (e.g., homophily)
- `signal_sensitivity` (np.ndarray): Array of signal sensitivity values
- `noise_sensitivity` (np.ndarray): Array of noise sensitivity values
- `global_sensitivity` (np.ndarray): Array of global sensitivity values
- `property_name` (str): Name of the graph property
- `title` (str): Plot title
- `figsize` (Tuple[int, int]): Figure size (width, height)
- `save_path` (str): Path to save the figure (if None, figure is not saved)

**Returns:**
- plt.Figure: The matplotlib Figure object

### plot_node_level_snr

```python
def plot_node_level_snr(graph, snr_values, node_positions=None, 
                       title='Node-level SNR', cmap='viridis', 
                       figsize=(8, 8), save_path=None)
```

Plot node-level SNR values on a graph.

**Parameters:**
- `graph` (Union[dgl.DGLGraph, nx.Graph]): A DGL or NetworkX graph
- `snr_values` (np.ndarray): Array of SNR values for each node
- `node_positions` (Dict): Dictionary of node positions (if None, layout is computed)
- `title` (str): Plot title
- `cmap` (str): Colormap for SNR values
- `figsize` (Tuple[int, int]): Figure size (width, height)
- `save_path` (str): Path to save the figure (if None, figure is not saved)

**Returns:**
- plt.Figure: The matplotlib Figure object
