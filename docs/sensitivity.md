---
layout: default
title: Sensitivity Analysis
nav_order: 5
has_children: true
permalink: /sensitivity
---

# Sensitivity Analysis
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The Sensitivity Analysis package provides tools for analyzing the signal-to-noise ratio (SNR) and sensitivity of graph neural networks. These tools offer insights into when and how graph structure affects model performance, based on the theoretical framework described in the paper.

## Key Components

### Signal, Noise, and Global Sensitivity

We introduce three key metrics that capture how an MPNN responds to different types of input perturbations:

1. **Signal Sensitivity** ($S^{(\ell)}_{i,p,q,r}$): Measures the MPNN's responsiveness to coherent class-specific changes in the input features
2. **Noise Sensitivity** ($N^{(\ell)}_{i,p,q,r}$): Measures the MPNN's responsiveness to random, unstructured variations in the input features
3. **Global Sensitivity** ($T^{(\ell)}_{i,p,q,r}$): Measures the MPNN's overall responsiveness to global shifts in the input features

These sensitivity measures are feature-independent, depending only on the model architecture and the graph structure, not on specific feature values.

### Signal-to-Noise Ratio (SNR) Estimation

The package provides two methods for estimating the SNR of graph neural networks:

1. **Monte Carlo Estimation**: Empirically estimates SNR by generating multiple node feature realizations
2. **Theorem-Based Estimation**: Uses our theoretical results to estimate SNR based on sensitivity analysis and feature covariance structures

## Experimental Analysis

### SNR vs. Homophily

A key finding from our analysis is the relationship between graph homophily and the signal-to-noise ratio of MPNNs:

- For standard GCNs, SNR is minimized at "ambiphily" (edge homophily $h = \frac{1}{k}$ where $k$ is the number of classes)
- SNR increases symmetrically as homophily either increases toward 1 or decreases toward 0
- This explains the "mid-homophily pitfall" observed in prior work, where MPNNs perform worst at ambiphily

### High-Pass vs. Low-Pass Filters

Our sensitivity analysis reveals when high-pass filters (using $I - \hat{A}$ as the graph operator) outperform standard low-pass filters:

- High-pass filters achieve higher SNR than low-pass filters for all heterophilic graphs (edge homophily $h < \frac{1}{k}$)
- Even for mildly homophilic graphs, high-pass filters can achieve higher SNR under certain conditions
- The transition point depends on the number of classes and the order of homophily being considered

### Node-Level Bottleneck Analysis

The sensitivity framework allows for node-level analysis of bottlenecks:

- By computing local homophily for each node, we can identify nodes suffering from homophilic bottlenecks
- The framework predicts which nodes will benefit most from graph rewiring

## Usage Examples

### Monte Carlo SNR Estimation

```python
from bridge.sensitivity import estimate_snr_monte_carlo, create_feature_generator

# Create a feature generator with specific covariance parameters
feature_generator = create_feature_generator(
    sigma_intra=0.1 * torch.eye(in_feats),
    sigma_inter=-0.05 * torch.eye(in_feats),
    tau=torch.eye(in_feats),
    eta=torch.eye(in_feats)
)

# Estimate SNR using Monte Carlo simulation
snr_mc = estimate_snr_monte_carlo(
    model, graph, in_feats, labels,
    num_montecarlo_simulations=100,
    feature_generator=feature_generator,
    device="cuda"
)
```

### Sensitivity Analysis

```python
from bridge.sensitivity import estimate_sensitivity_autograd

# Compute signal sensitivity
signal_sensitivity = estimate_sensitivity_autograd(
    model, graph, in_feats, labels, 
    sensitivity_type="signal",
    device="cuda"
)

# Compute noise sensitivity
noise_sensitivity = estimate_sensitivity_autograd(
    model, graph, in_feats, labels, 
    sensitivity_type="noise",
    device="cuda"
)

# Compute global sensitivity
global_sensitivity = estimate_sensitivity_autograd(
    model, graph, in_feats, labels, 
    sensitivity_type="global",
    device="cuda"
)
```

### Comprehensive Experiment

```python
from bridge.sensitivity import run_sensitivity_experiment

# Run a comprehensive sensitivity analysis
results = run_sensitivity_experiment(
    model=model,
    graph=graph,
    feature_generator=feature_generator,
    in_feats=in_feats,
    num_acc_repeats=100,
    num_monte_carlo_samples=100,
    sigma_intra=sigma_intra,
    sigma_inter=sigma_inter,
    tau=tau,
    eta=eta,
    device="cuda"
)

print(f"Monte Carlo SNR: {results['estimated_snr_mc'].item():.4f}")
print(f"Theorem-based SNR: {results['estimated_snr_theorem'].item():.4f}")
print(f"Test Accuracy: {results['mean_test_acc']:.4f}")
```

For more details on available functionality, see the [Sensitivity API]({% link sensitivity/api-reference.md %}).
