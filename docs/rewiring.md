---
layout: default
title: Graph Rewiring
nav_order: 4
has_children: true
permalink: /rewiring
---

# Graph Rewiring
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

BRIDGE leverages theoretical insights about optimal graph structures to implement a graph rewiring strategy that enhances MPNN performance. This approach involves modifying the graph's edge structure to increase higher-order homophily, which our theory shows is crucial for effective message passing.

## BRIDGE Algorithm

The BRIDGE (Block Rewiring from Inference-Derived Graph Ensembles) rewiring algorithm consists of the following steps:

1. **Cold-Start GCN Training**: Train a base GCN on the original graph to learn initial node embeddings
2. **Class Probability Estimation**: Use the trained GCN to infer class probabilities for all nodes
3. **Optimal Block Matrix Computation**: Compute an optimal block matrix based on theoretical results
4. **Graph Rewiring**: Modify the graph's connectivity based on the optimal block structure
5. **Selective GCN Training**: Train a new GCN that can choose between the original and rewired graphs

### Rewiring Pipeline

The complete rewiring pipeline is implemented in the `run_bridge_pipeline` function. The key steps are formalized in the following algorithm:

```
Algorithm: SBM Graph Rewiring
Require: Graph G = (V, E) with adjacency matrix A, training labels y_train, number of classes k
Require: α_add, α_drop ∈ [0, 1], softmax temperature t, permutation matrix P_k and mean degree ⟨d⟩ for B̂_opt

Step 1: Cold-Start and Class Probability Estimation
1. Train a cold-start GCN on the available labels y_train to obtain logits H^(ℓ)
2. Convert logits to class probability estimates Z using softmax with temperature t

Step 2: Compute the Optimal Block Matrix and Expected Adjacency
3. Compute the optimal block matrix B_opt using P_k and ⟨d⟩ according to our theory
4. Compute the expected adjacency matrix E[A_opt] = (1/n) Z B_opt Z^T

Step 3: Identify and Rewire "Surprising" Edges
5. For each node pair (i, j):
   a. Compute the "surprise" of the current edge state A_ij given the optimal structure
   b. With probability proportional to the surprise, consider this edge for rewiring
   c. If an edge exists, remove it with probability α_drop
   d. If no edge exists, add it with probability α_add

Step 4: Return the rewired graph G' = (V, E')
```

## Homophily-Masked Message Passing

To further enhance performance, we introduce a novel Homophily-Masked message passing model that adaptively chooses between the original and rewired graph structures for each node.

For each node, the model selects which graph structure provides better feature propagation based on local homophily. The message passing operation computes embeddings for both the original graph G and the rewired graph G', and at the final layer selects the output with higher local homophily:

$$H^{*(L)}_i = H^{(L)}_i(\hat{A}^{(m^*(i))}), \quad m^*(i) = \arg\max_{1 \leq k \leq K} h^{L,L}_i(\hat{A}^{(k)})$$

where $h^{L,L}_i(\hat{A}^{(k)})$ is the local homophily of node $i$ with respect to the graph operator $\hat{A}^{(k)}$.

## Graph Rewiring Parameters

The rewiring process is controlled by several key parameters:

- **Temperature (t)**: Controls the "confidence" of the class probabilities; lower values make the predictions more deterministic
- **Mean Degree (⟨d⟩)**: Controls the average number of connections per node in the rewired graph
- **Permutation Matrix (P_k)**: Determines the block structure for connections between classes
- **Edge Addition Probability (α_add)**: Controls how likely new edges are added in places suggested by the model
- **Edge Removal Probability (α_drop)**: Controls how likely existing edges are removed when in conflict with the model

These parameters can be optimized using the `objective_rewiring` function provided in the library, which uses Optuna for hyperparameter search.

## Example Usage

```python
from bridge.rewiring import run_bridge_pipeline
from bridge.utils import generate_all_symmetric_permutation_matrices

# Generate all possible symmetric permutation matrices for k classes
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]  # Choose the first permutation matrix

# Run the BRIDGE pipeline
results = run_bridge_pipeline(
    g=g,
    P_k=P_k,
    h_feats_gcn=64,
    n_layers_gcn=2,
    dropout_p_gcn=0.5,
    model_lr_gcn=1e-3,
    h_feats_selective=64,
    n_layers_selective=2,
    dropout_p_selective=0.5,
    model_lr_selective=1e-3,
    temperature=1.0,
    p_add=0.1,
    p_remove=0.1,
    d_out=10,
    num_graphs=1,
    device='cuda'
)
```

## Performance Analysis

Our experiments show that the BRIDGE rewiring approach significantly improves performance across various datasets:

- **Homophilic Graphs**: Enhances performance by optimizing the community structure
- **Heterophilic Graphs**: Creates optimal connectivity patterns that align with theoretical predictions
- **Synthetic Datasets**: Shows consistent improvements across different homophily levels

See the [Experimental Results]({% link rewiring/experimental-results.md %}) page for detailed performance evaluations.
