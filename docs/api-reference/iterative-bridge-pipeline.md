---
layout: default
title: run_iterative_bridge_pipeline
parent: API Reference
---

# run_iterative_bridge_pipeline
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `run_iterative_bridge_pipeline` function performs multiple rewiring rounds,
progressively modifying the graph before training a final SelectiveGCN. It uses a
fast SGC-based classifier for early iterations to speed up computation.

## Function Signature

```python
from bridge.rewiring import run_iterative_bridge_pipeline
```

Refer to the Python docstring for full parameter details. Important options
include `n_rewire` (number of rewiring iterations) and `use_sgc` to enable the
fast SGC classifier during the iterative phase.

## Example Usage

```python
results = run_iterative_bridge_pipeline(
    g=g,
    P_k=P_k,
    n_rewire=5,
    device="cuda",
)
print(results["selective"]["test_acc"])
```
