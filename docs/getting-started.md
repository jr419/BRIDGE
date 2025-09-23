---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

You can install BRIDGE directly from the repository:

```bash
git clone https://github.com/jr419/BRIDGE.git
cd BRIDGE
pip install -e .
```

This will install BRIDGE in development mode, making it available in your Python environment.

### Requirements

BRIDGE requires the following Python packages:

```
dgl>=1.1.0
numpy>=1.26.2
optuna>=4.0.0
ortools>=9.11.4210
pandas>=2.1.3
scipy>=1.11.3
scikit-learn>=1.3.2
torch>=2.0.0
tqdm>=4.66.1
pyyaml>=5.4.1
```

These dependencies will be automatically installed when using `pip install -e .`.

## Quick Start Example

Here's a simple example to get started with the BRIDGE rewiring technique:

```python
import dgl
import torch
from bridge.rewiring import run_bridge_pipeline
from bridge.utils import generate_all_symmetric_permutation_matrices

# Load a dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Generate permutation matrices
k = len(torch.unique(g.ndata['label']))
all_matrices = generate_all_symmetric_permutation_matrices(k)
P_k = all_matrices[0]  # Choose the first permutation matrix

# Run the rewiring pipeline
results = run_bridge_pipeline(
    g=g,
    P_k=P_k,
    h_feats_mpnn=64,
    n_layers_mpnn=2,
    dropout_p_mpnn=0.5,
    model_lr_mpnn=1e-3,
    d_out=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Print results
print(f"Base test accuracy: {results['cold_start']['test_acc']:.4f}")
print(f"Rewired test accuracy: {results['rewired']['test_acc']:.4f}")
```

## Notes

- Rewiring uses hard class predictions and full resampling.
- No temperature or add/remove probabilities are used.
