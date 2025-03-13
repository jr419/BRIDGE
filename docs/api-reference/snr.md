---
layout: default
title: SNR Estimation
parent: API Reference
---

# Signal-to-Noise Ratio (SNR) Estimation
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The SNR estimation module provides functions for estimating the Signal-to-Noise Ratio (SNR) of Graph Neural Networks through Monte Carlo simulations and theoretical approaches. The SNR is a crucial metric for understanding when and how well graph neural networks can discriminate between different classes.

## Estimation Functions

### estimate_snr_monte_carlo

```python
def estimate_snr_monte_carlo(
    model: nn.Module,
    graph: dgl.DGLGraph,
    in_feats: int,
    labels: torch.Tensor,
    num_montecarlo_simulations: int,
    feature_generator: Callable,
    device: str = "cpu",
    inner_samples: int = 100,
    split_model_input_size: int = 100
) -> torch.Tensor
```

Estimates the Signal-to-Noise Ratio (SNR) of an MPNN's outputs via Monte Carlo simulation.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | The neural network model |
| `graph` | dgl.DGLGraph | The input graph |
| `in_feats` | int | Number of input features |
| `labels` | torch.Tensor | Node labels |
| `num_montecarlo_simulations` | int | Number of outer loop iterations (different mu samples) |
| `feature_generator` | Callable | Function to generate features with specific signature |
| `device` | str | Device to compute on |
| `inner_samples` | int | Number of feature samples for each mu |
| `split_model_input_size` | int | Maximum batch size for processing samples |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Tensor of shape [num_nodes, out_feats] containing the estimated SNR for each node and output feature |

#### Mathematical Definition

The SNR for each node i and output feature p is defined as:

$$\text{SNR}(H^{(\ell)}_{i,p}) = \frac{\text{Var}_\mu[\mathbb{E}[H^{(\ell)}_{i,p} | \mu]]}{\mathbb{E}_\mu[\text{Var}[H^{(\ell)}_{i,p} | \mu]]}$$

where:
- $\mu$ represents the class-specific mean vectors
- $H^{(\ell)}_{i,p}$ is the output of the GNN for node i and feature p
- $\text{Var}_\mu$ is the variance across different class means
- $\mathbb{E}_\mu$ is the expectation across different class means

#### Example

```python
import torch
import dgl
from bridge.sensitivity import estimate_snr_monte_carlo, create_feature_generator
from bridge.models import LinearGCN

# Create a model
model = LinearGCN(in_feats=5, hidden_feats=16, out_feats=3)

# Load a graph
g = dgl.data.CoraGraphDataset()[0]
labels = g.ndata['label']

# Create a feature generator with specific covariance parameters
feature_generator = create_feature_generator(
    sigma_intra=0.1 * torch.eye(5),
    sigma_inter=-0.05 * torch.eye(5),
    tau=torch.eye(5),
    eta=torch.eye(5)
)

# Estimate SNR using Monte Carlo
snr_mc = estimate_snr_monte_carlo(
    model=model,
    graph=g,
    in_feats=5,
    labels=labels,
    num_montecarlo_simulations=100,
    feature_generator=feature_generator,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Print average SNR
print(f"Average Monte Carlo SNR: {snr_mc.mean().item():.4f}")
```

### estimate_snr_theorem

```python
def estimate_snr_theorem(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    labels: torch.Tensor, 
    sigma_intra: torch.Tensor, 
    sigma_inter: torch.Tensor, 
    tau: torch.Tensor, 
    eta: torch.Tensor
) -> torch.Tensor
```

Computes SNR estimate using the theoretical formula from the paper, which relates SNR to signal, global, and noise sensitivities, weighted by covariance matrices of the input features.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | The neural network model (must be LinearGCN) |
| `graph` | dgl.DGLGraph | The input graph |
| `labels` | torch.Tensor | Node labels |
| `sigma_intra` | torch.Tensor | Intra-class covariance matrix |
| `sigma_inter` | torch.Tensor | Inter-class covariance matrix |
| `tau` | torch.Tensor | Global shift covariance matrix |
| `eta` | torch.Tensor | Noise covariance matrix |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Tensor of shape [num_nodes] containing the estimated SNR for each node (averaged across output features) |

#### Mathematical Definition

The theorem-based SNR for a linear GCN is defined as:

$$\text{SNR}(H^{(\ell)}_{i,p}) \approx \frac{\sum_{q,r=1}^{d_{in}}(\Sigma^{(intra)}_{q,r} - \Sigma^{(inter)}_{q,r})S^{(\ell)}_{i,p,q,r} + \sum_{q,r=1}^{d_{in}}\Sigma^{(inter)}_{q,r}T^{(\ell)}_{i,p,q,r}}{\sum_{q,r=1}^{d_{in}}\Phi_{q,r}T^{(\ell)}_{i,p,q,r} + \sum_{q,r=1}^{d_{in}}\Psi_{q,r}N^{(\ell)}_{i,p,q,r}}$$

where:
- $S^{(\ell)}_{i,p,q,r}$ is the signal sensitivity
- $T^{(\ell)}_{i,p,q,r}$ is the global sensitivity
- $N^{(\ell)}_{i,p,q,r}$ is the noise sensitivity
- $\Sigma^{(intra)}$, $\Sigma^{(inter)}$, $\Phi$, and $\Psi$ are the covariance matrices

#### Example

```python
from bridge.sensitivity import estimate_snr_theorem, estimate_sensitivity_analytic
from bridge.models import LinearGCN

# Create a linear GCN model
model = LinearGCN(in_feats=5, hidden_feats=16, out_feats=3)

# Define covariance matrices
sigma_intra = 0.1 * torch.eye(5)
sigma_inter = -0.05 * torch.eye(5)
tau = torch.eye(5)
eta = torch.eye(5)

# Estimate SNR using the theorem
snr_theorem = estimate_snr_theorem(
    model=model,
    graph=g,
    labels=g.ndata['label'],
    sigma_intra=sigma_intra,
    sigma_inter=sigma_inter,
    tau=tau,
    eta=eta
)

print(f"Average theorem-based SNR: {snr_theorem.mean().item():.4f}")
```

### estimate_snr_theorem_autograd

```python
def estimate_snr_theorem_autograd(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    in_feats: int,
    labels: torch.Tensor, 
    sigma_intra: torch.Tensor, 
    sigma_inter: torch.Tensor, 
    tau: torch.Tensor, 
    eta: torch.Tensor, 
    device: str = "cuda"
) -> torch.Tensor
```

Computes SNR estimate using the theoretical formula with autograd sensitivities, making it applicable to non-linear models.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | The neural network model (any model that works with autograd) |
| `graph` | dgl.DGLGraph | The input graph |
| `in_feats` | int | Number of input features |
| `labels` | torch.Tensor | Node labels |
| `sigma_intra` | torch.Tensor | Intra-class covariance matrix |
| `sigma_inter` | torch.Tensor | Inter-class covariance matrix |
| `tau` | torch.Tensor | Global shift covariance matrix |
| `eta` | torch.Tensor | Noise covariance matrix |
| `device` | str | Device to compute on |

#### Returns

| Return Type | Description |
|-------------|-------------|
| torch.Tensor | Tensor of shape [num_nodes] containing the estimated SNR for each node (averaged across output features) |

#### Example

```python
from bridge.sensitivity import estimate_snr_theorem_autograd
from bridge.models import TwoLayerGCN

# Create a non-linear GCN model
model = TwoLayerGCN(in_feats=5, hidden_feats=16, out_feats=3)

# Estimate SNR using autograd for non-linear model
snr_autograd = estimate_snr_theorem_autograd(
    model=model,
    graph=g,
    in_feats=5,
    labels=g.ndata['label'],
    sigma_intra=sigma_intra,
    sigma_inter=sigma_inter,
    tau=tau,
    eta=eta,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Average autograd theorem-based SNR: {snr_autograd.mean().item():.4f}")
```

## Implementation Details

### Monte Carlo Estimation

The Monte Carlo SNR estimation procedure works by:

1. **Multiple Class Mean Sampling**:
   - Generates `num_montecarlo_simulations` different realizations of class-mean vectors ($\mu$)
   
2. **Multiple Feature Samples**:
   - For each class mean realization, generates `inner_samples` feature samples
   - Each sample has the same class means but different global shifts and node-specific noise
   
3. **Conditional Statistics Computation**:
   - For each node i and output feature p:
     - Computes the mean over inner samples for each class mean
     - Computes the variance over inner samples for each class mean
   
4. **SNR Calculation**:
   - Computes variance of the means (numerator)
   - Computes mean of the variances (denominator)
   - Divides to get SNR

### Theorem-Based Estimation

The theorem-based SNR estimation uses the analytical relationship between SNR and model sensitivities:

1. **Sensitivity Computation**:
   - Computes signal sensitivity ($S$), noise sensitivity ($N$), and global sensitivity ($T$)
   - For linear models, uses analytical formulas
   - For non-linear models, uses automatic differentiation

2. **Weighted Combination**:
   - Weights sensitivities by the appropriate covariance matrices
   - Constructs the numerator and denominator according to the theoretical formula

3. **SNR Calculation**:
   - Divides the weighted numerator by the weighted denominator
   - Averages across output features if needed

## Relationship to Graph Neural Networks

The SNR estimation functions provide insights into:

1. **GNN Performance Prediction**: Higher SNR typically correlates with better classification performance

2. **Bottleneck Identification**: Nodes with low SNR may act as bottlenecks for information flow

3. **Architecture Selection**: The relationship between SNR and graph structure can inform the choice of GNN architecture

4. **Optimal Graph Structure**: SNR analysis reveals what graph structures are optimal for message passing

## Usage in Experiments

The SNR estimation functions are typically used in sensitivity analysis experiments:

```python
from bridge.sensitivity import run_sensitivity_experiment

# Configure experiment
results = run_sensitivity_experiment(
    model=model,
    graph=g,
    feature_generator=feature_generator,
    in_feats=in_feats,
    num_acc_repeats=10,
    num_monte_carlo_samples=100,
    sigma_intra=sigma_intra,
    sigma_inter=sigma_inter,
    tau=tau,
    eta=eta,
    device="cuda"
)

# Extract SNR estimates
snr_mc = results['estimated_snr_mc']
snr_theorem = results['estimated_snr_theorem']
test_acc = results['mean_test_acc']

print(f"Monte Carlo SNR: {snr_mc.item():.4f}")
print(f"Theorem-based SNR: {snr_theorem.item():.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```
