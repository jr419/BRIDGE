---
layout: default
title: run_iterative_bridge_experiment
parent: API Reference
---

# run_iterative_bridge_experiment
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

`run_iterative_bridge_experiment` wraps
`run_iterative_bridge_pipeline` to repeat the process over multiple data splits
or random seeds and compute confidence intervals.

## Example Usage

```python
stats, results = run_iterative_bridge_experiment(
    g,
    P_k=P_k,
    num_splits=10,
    n_rewire=3,
)
print(stats["test_acc_mean"])
```
