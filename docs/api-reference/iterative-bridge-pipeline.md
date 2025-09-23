---
layout: default
title: run_iterative_bridge_pipeline
parent: API Reference
---

# run_iterative_bridge_pipeline

The iterative BRIDGE pipeline progressively modifies the graph using repeated rewiring steps, then trains a standard model on the final rewired graph. Predictions are hard labels and rewiring fully resamples the adjacency. No temperature or partial add/remove probabilities are used.

## Example Usage

```python
from bridge.rewiring import run_iterative_bridge_pipeline

results = run_iterative_bridge_pipeline(
    g,
    P_k,
    model_type='GCN',
    n_rewire=5,
    d_out=10,
)

print(results["rewired"]["test_acc"])
```
