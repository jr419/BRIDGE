---
layout: default
title: Feature Generation
parent: API Reference
---

# Feature Generation
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The feature generation module provides functions for generating synthetic node features with controlled covariance structures between classes, global shifts, and node-specific noise. These functions are essential for sensitivity analysis and SNR estimation in graph neural networks.

## Feature Generation Functions

### generate_features

```python
def generate_features(
    num_nodes: int, 
    num_features: int, 
    labels: np.ndarray, 
    inter_class_cov: np.ndarray, 
    intra_class_cov: np.ndarray, 
    global_cov: np.ndarray, 
    noise_cov: np.ndarray, 
    mu_repeats: int = 1
) -> np.ndarray
```

Generates synthetic node features with controlled covariance structure. This function implements the feature generation model from the paper: X_i = μ_{y_i} + γ + ε_i.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_nodes` | int | Number of nodes |
| `num_features` | int | Number of feature dimensions |
| `labels` | np.ndarray | Node class labels (numpy array) |
| `inter_class_cov` | np.ndarray | Covariance matrix between different classes |
| `intra_class_cov` | np.ndarray | Covariance matrix within the same class |
| `global_cov` | np.ndarray | Covariance matrix for the global shift |
| `noise_cov` | np.ndarray | Covariance matrix for the node-specific noise |
| `mu_repeats` | int | Number of feature realizations to generate for each class mean |

#### Returns

| Return Type | Description |
|-------------|-------------|
| np.ndarray | A numpy array of shape (num_nodes, num_features, mu_repeats) containing the generated features |

#### Mathematical Model

The feature generation model is:

$$X_i = \mu_{y_i} + \gamma + \varepsilon_i$$

where:
- $\mu_{y_i}$ is the class-specific mean for the class of node i
- $\gamma$ is a global random vector (same for all nodes)
- $\varepsilon_i$ is node-specific noise

#### Example

```python
import numpy as np
from bridge.sensitivity import generate_features

# Define class labels for 10 nodes (3 classes)
labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

# Define simple covariance matrices for a 3-feature scenario
feature_dim = 3
intra_class_cov = 0.1 * np.eye(feature_dim)   # Strong within-class correlation
inter_class_cov = -0.05 * np.eye(feature_dim)  # Negative between-class correlation
global_cov = np.eye(feature_dim)              # Unit global variation
noise_cov = 0.5 * np.eye(feature_dim)         # Moderate node-specific noise

# Generate features (10 nodes, 3 features, 5 realizations)
features = generate_features(
    num_nodes=10,
    num_features=feature_dim,
    labels=labels,
    inter_class_cov=inter_class_cov,
    intra_class_cov=intra_class_cov,
    global_cov=global_cov,
    noise_cov=noise_cov,
    mu_repeats=5
)

print(f"Generated features shape: {features.shape}")  # Should be (10, 3, 5)

# Analyze features for the first realization
first_realization = features[:, :, 0]
print("Features for the first realization:")
print(first_realization)

# Compute mean feature vector for each class
for class_id in range(3):
    class_mask = (labels == class_id)
    class_features = first_realization[class_mask]
    class_mean = np.mean(class_features, axis=0)
    print(f"Class {class_id} mean feature vector: {class_mean}")
```

### create_feature_generator

```python
def create_feature_generator(
    sigma_intra: torch.Tensor,
    sigma_inter: torch.Tensor,
    tau: torch.Tensor,
    eta: torch.Tensor,
    dtype: torch.dtype = torch.float64
) -> Callable
```

Creates a feature generator function with fixed covariance parameters. This is a factory function that returns another function with a specific signature required for SNR estimation.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sigma_intra` | torch.Tensor | Intra-class covariance matrix |
| `sigma_inter` | torch.Tensor | Inter-class covariance matrix |
| `tau` | torch.Tensor | Global shift covariance matrix |
| `eta` | torch.Tensor | Noise covariance matrix |
| `dtype` | torch.dtype | Torch data type for the output tensor |

#### Returns

| Return Type | Description |
|-------------|-------------|
| Callable | A function that generates features with signature: feature_generator(num_nodes, in_feats, labels, num_mu_samples) |

#### Returned Function Signature

```python
def feature_generator_fixed(
    num_nodes: int, 
    in_feats: int, 
    labels: torch.Tensor, 
    num_mu_samples: int = 1
) -> torch.Tensor
```

#### Example

```python
import torch
from bridge.sensitivity import create_feature_generator

# Create covariance matrices
in_feats = 5
sigma_intra = 0.1 * torch.eye(in_feats)      # Strong intra-class correlation
sigma_inter = -0.05 * torch.eye(in_feats)    # Negative inter-class correlation
tau = torch.eye(in_feats)                    # Unit global variation
eta = 0.5 * torch.eye(in_feats)              # Moderate node-specific noise

# Create feature generator with these covariance matrices
feature_generator = create_feature_generator(
    sigma_intra=sigma_intra,
    sigma_inter=sigma_inter,
    tau=tau,
    eta=eta
)

# Use the generator to create features
num_nodes = 100
labels = torch.randint(0, 3, (num_nodes,))  # Random labels with 3 classes
features = feature_generator(
    num_nodes=num_nodes,
    in_feats=in_feats,
    labels=labels,
    num_mu_samples=10
)

print(f"Generated features shape: {features.shape}")  # Should be (100, 5, 10)
```

## Implementation Details

### Feature Generation Process

The feature generation follows these steps:

1. **Class Mean Generation**:
   - A large covariance matrix is constructed with block structure
   - Intra-class covariance (diagonal blocks): Controls similarity between nodes of same class
   - Inter-class covariance (off-diagonal blocks): Controls similarity between nodes of different classes
   - Class means are drawn from this multivariate normal distribution

2. **Global Shift Generation**:
   - A global random vector is generated for each repeat
   - This vector is shared across all nodes
   - Controlled by the global covariance matrix (tau)

3. **Node-Specific Noise**:
   - Independent random noise is generated for each node
   - Controlled by the noise covariance matrix (eta)

4. **Feature Combination**:
   - The final features are the sum of class means, global shift, and node-specific noise
   - The process is repeated for `mu_repeats` times to create multiple realizations

### Covariance Matrix Structure

The covariance matrices control different aspects of the feature distribution:

- **sigma_intra** (Intra-class covariance): Controls how similar nodes of the same class are to each other.
  - Higher values → stronger class signal
  - Typically positive for clear class separation

- **sigma_inter** (Inter-class covariance): Controls how similar nodes of different classes are to each other.
  - Negative values → class anti-correlation
  - Typically negative or zero for clearer class boundaries

- **tau** (Global covariance): Controls shared variations across all nodes.
  - Higher values → more dataset-wide noise
  - Affects all classes simultaneously

- **eta** (Node-specific covariance): Controls individual node variations.
  - Higher values → more per-node noise
  - Affects class separability

## Usage in Sensitivity Analysis

The feature generation functions are primarily used in sensitivity analysis to:

1. **SNR Estimation**: Generate multiple feature realizations for Monte Carlo SNR estimation
2. **Model Training**: Create training data with controlled properties
3. **Controllable Experiments**: Investigate the effect of feature properties on model performance

### Example with SNR Estimation

```python
from bridge.sensitivity import create_feature_generator, estimate_snr_monte_carlo
from bridge.models import LinearGCN

# Create covariance matrices
in_feats = 5
sigma_intra = 0.1 * torch.eye(in_feats)
sigma_inter = -0.05 * torch.eye(in_feats)
tau = torch.eye(in_feats)
eta = torch.eye(in_feats)

# Create feature generator
feature_generator = create_feature_generator(
    sigma_intra=sigma_intra,
    sigma_inter=sigma_inter,
    tau=tau,
    eta=eta
)

# Create a model
model = LinearGCN(in_feats=in_feats, hidden_feats=16, out_feats=3)

# Estimate SNR using Monte Carlo with the feature generator
snr_mc = estimate_snr_monte_carlo(
    model=model,
    graph=g,
    in_feats=in_feats,
    labels=g.ndata['label'],
    num_montecarlo_simulations=100,
    feature_generator=feature_generator,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

### Example with Comprehensive Experiment

```python
from bridge.sensitivity import run_sensitivity_experiment, create_feature_generator

# Create feature generator
feature_generator = create_feature_generator(
    sigma_intra=0.1 * torch.eye(in_feats),
    sigma_inter=-0.05 * torch.eye(in_feats),
    tau=torch.eye(in_feats),
    eta=torch.eye(in_feats)
)

# Run comprehensive experiment
results = run_sensitivity_experiment(
    model=model,
    graph=g,
    feature_generator=feature_generator,
    in_feats=in_feats,
    num_acc_repeats=10,
    num_monte_carlo_samples=100,
    sigma_intra=0.1 * torch.eye(in_feats),
    sigma_inter=-0.05 * torch.eye(in_feats),
    tau=torch.eye(in_feats),
    eta=torch.eye(in_feats)
)
```

## Relationship to SNR

The feature generation model directly corresponds to the SNR theoretical framework:

- The class means (μ) represent the signal component
- The global shift (γ) and node-specific noise (ε) represent the noise components
- The covariance parameters control the signal-to-noise ratio:
  - Higher intra-class vs. inter-class covariance → Higher SNR
  - Lower global and node-specific covariance → Higher SNR

## Related Components

- [estimate_snr_monte_carlo]({% link api-reference/snr.md %}): Uses feature generators for SNR estimation
- [run_sensitivity_experiment]({% link api-reference/sensitivity-analysis.md %}): Comprehensive experiment using feature generators
- [SyntheticGraphDataset]({% link api-reference/synthetic-dataset.md %}): Uses similar feature generation methodology for datasets
