---
layout: default
title: Experimental Results
parent: Graph Rewiring
nav_order: 1
---

# Experimental Results
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Performance on Standard Benchmarks

We evaluate the performance of BRIDGE on standard benchmark datasets, comparing the base GCN accuracy with the rewired graph accuracy.

### Homophilic Datasets

| Dataset | Base GCN | BRIDGE (Rewired) | Improvement (%) |
|---------|----------|------------------|----------------|
| Cora    | 81.78 ± 0.26 | 81.82 ± 0.29 | 0.05 |
| Citeseer | 71.79 ± 0.18 | 72.19 ± 0.19 | 0.56 |

For homophilic datasets like Cora and Citeseer, BRIDGE maintains or slightly improves performance by refining the existing community structure.

### Heterophilic Datasets

| Dataset | Model | Base | BRIDGE (Rewired) | Improvement (%) |
|---------|-------|------|------------------|----------------|
| Actor | High-Pass GCN (dir) | 31.14 ± 0.99 | 33.30 ± 0.66 | 6.93 |
| Actor | High-Pass GCN (sym) | 32.07 ± 0.76 | 33.66 ± 0.47 | 4.97 |
| Squirrel | Low-Pass GCN (sym) | 46.05 ± 0.91 | 46.39 ± 1.27 | 0.73 |
| Chameleon | Low-Pass GCN (dir) | 65.75 ± 1.15 | 66.21 ± 1.14 | 0.70 |
| Wisconsin | Low-Pass GCN (sym) | 52.55 ± 4.80 | 77.65 ± 2.08 | 47.76 |
| Cornell | High-Pass GCN (sym) | 64.05 ± 3.54 | 66.49 ± 3.80 | 3.80 |

For heterophilic datasets, BRIDGE shows substantial improvements, particularly for datasets with strong heterophilic structures like Wisconsin.

## Performance on Synthetic Datasets

We also evaluate BRIDGE on synthetic datasets with controlled homophily levels.

| Homophily | Model | Base | BRIDGE (Rewired) | Improvement (%) |
|-----------|-------|------|------------------|----------------|
| h=0.10 | High-Pass GCN | 59.80 ± 0.39 | 58.20 ± 0.60 | -2.68 |
| h=0.20 | High-Pass GCN | 42.83 ± 0.32 | 42.87 ± 0.52 | 0.08 |
| h=0.30 | High-Pass GCN | 43.87 ± 0.20 | 46.13 ± 0.49 | 5.17 |
| h=0.40 | High-Pass GCN | 35.63 ± 0.66 | 40.27 ± 1.25 | 13.00 |
| h=0.50 | High-Pass GCN | 35.30 ± 0.76 | 38.83 ± 1.12 | 10.01 |
| h=0.60 | High-Pass GCN | 34.33 ± 0.73 | 39.67 ± 1.86 | 15.53 |
| h=0.70 | High-Pass GCN | 25.73 ± 0.31 | 39.23 ± 1.28 | 52.46 |
| h=0.30 | Low-Pass GCN | 39.70 ± 0.40 | 41.83 ± 0.98 | 5.37 |
| h=0.50 | Low-Pass GCN | 57.50 ± 0.55 | 58.80 ± 0.84 | 2.26 |
| h=0.60 | Low-Pass GCN | 81.63 ± 0.70 | 83.50 ± 0.47 | 2.29 |
| h=0.70 | Low-Pass GCN | 95.93 ± 0.25 | 95.87 ± 0.31 | -0.07 |

The results on synthetic datasets reveal:

1. **High-Pass GCNs** benefit significantly from rewiring, especially at higher homophily levels, demonstrating the ability of BRIDGE to restructure graphs to better suit the model architecture.

2. **Low-Pass GCNs** show moderate improvements in the mid-homophily range and maintain high performance at high homophily levels.

## Structural Analysis

To understand the impact of rewiring, we analyze various structural metrics before and after applying BRIDGE.

### Changes in Graph Structure

| Dataset | Metric | Original | Rewired |
|---------|--------|----------|---------|
| Cora | Mean Degree | 3.90 | 6.13 |
| Cora | Mean Homophily | 0.81 | 0.86 |
| Wisconsin | Mean Degree | 3.21 | 5.84 |
| Wisconsin | Mean Homophily | 0.21 | 0.68 |

BRIDGE consistently increases higher-order homophily, which our theory identifies as critical for MPNN performance.

### Edge Modifications

BRIDGE's rewiring algorithm makes strategic edge modifications:

- **Edge Additions**: New edges are added to create connections between nodes of the same class that were previously disconnected
- **Edge Removals**: Edges connecting nodes from different classes with low predicted relevance are pruned

For example, in the Wisconsin dataset, BRIDGE adds approximately 43% new edges and removes around 12% of existing edges, significantly transforming the graph structure to match the optimal pattern predicted by our theory.

## Ablation Study

To understand the contribution of different components, we conducted an ablation study:

| Dataset | Full BRIDGE | No Selective GCN | No Temperature | No Optimal P_k |
|---------|-------------|------------------|---------------|----------------|
| Squirrel | 45.68 ± 0.97 | 44.92 ± 1.09 | 44.81 ± 1.13 | 44.57 ± 1.21 |
| Wisconsin | 77.65 ± 2.08 | 70.39 ± 3.14 | 67.84 ± 3.65 | 65.49 ± 3.89 |

Key findings:

1. **Homophily-Masked Selective GCN** significantly contributes to performance, especially for heterophilic datasets
2. **Temperature parameter** in class probability estimation is important for controlling the confidence of node class assignments
3. **Optimal Permutation Matrix** selection is critical for achieving the best performance, validating our theoretical prediction about optimal graph structures