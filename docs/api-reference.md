---
layout: default
title: API Reference
nav_order: 3
---

# API Reference
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

This section documents the BRIDGE library's main modules and functions.

## Modules

- models: GCN implementations
- rewiring: Graph rewiring pipelines and operations
- training: Training utilities for standard models
- utils: Utility functions for datasets, masks, and metrics

## Key Components

- GCN: Graph Convolutional Network
- run_bridge_pipeline: End-to-end rewiring pipeline using standard models
- run_bridge_experiment: Aggregated experiments with confidence intervals
- create_rewired_graph: Low-level function to generate rewired graphs via full resampling
