---
layout: default
title: Rewiring
nav_order: 4
---

# Rewiring Overview

BRIDGE performs graph rewiring using hard predictions (argmax) and fully resamples the adjacency from optimal block probabilities.

## Pipeline

1. Train a base GCN on the original graph
2. Predict hard labels and compute an optimal block matrix
3. Rewire the graph by resampling the full adjacency
4. Train a standard GCN on the rewired graph

### Algorithm

```
Algorithm: Full-Resampling SBM Graph Rewiring
Require: Graph G = (V, E), labels y, classes k, permutation matrix P_k, target mean degree ⟨d⟩

1) Train a cold-start model; get hard labels Z via argmax.
2) Compute B_opt from P_k and ⟨d⟩; compute E[A_opt] = (1/n) Z B_opt Z^T.
3) Sample each edge A_ij ~ Bernoulli(E[A_opt]_ij) and enforce symmetry.
4) Return rewired graph G'.
```

## Parameters

- Mean degree d_out
- Permutation matrix P_k

Notes:
- No temperature or partial add/remove probabilities.
- Standard models only; homophily masks removed.
