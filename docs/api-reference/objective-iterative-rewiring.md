---
layout: default
title: objective_iterative_rewiring
parent: API Reference
---

# objective_iterative_rewiring
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

This Optuna objective tunes the parameters of the iterative rewiring pipeline and
SelectiveGCN model.  It samples hyperparameters such as the number of rewiring
iterations, SGC settings and learning rates, then calls
`run_iterative_bridge_experiment`.

## Basic Usage

```python
study = optuna.create_study(direction="minimize")
study.optimize(lambda t: objective_iterative_rewiring(
    t,
    g,
    best_gcn_params,
    all_matrices,
    device="cuda"
), n_trials=50)
```
