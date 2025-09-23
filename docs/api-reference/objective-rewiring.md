---
layout: default
title: objective_rewiring
parent: API Reference
---

# objective_rewiring

The `objective_rewiring` function defines the Optuna objective that tunes BRIDGE rewiring hyperparameters. Rewiring uses hard predictions (argmax) and full resampling; there is no temperature and no p_add/p_remove. Only standard models are used.

## Signature

```python
objective_rewiring(
    trial: optuna.Trial,
    g: dgl.DGLGraph,
    best_mpnn_params: Dict[str, Any],
    all_matrices: List[np.ndarray],
    device: Union[str, torch.device] = 'cpu',
    n_epochs: int = 1000,
    num_splits: int = 100,
    early_stopping: int = 50,
    model_type: str = 'GCN',
    do_self_loop: bool = False,
    dataset_name: str = 'unknown'
) -> float
```

## Tuned Parameters

- `matrix_idx`: Index of the permutation matrix P_k
- `d_out`: Desired mean degree for rewired graph

Base model hyperparameters are taken from `best_mpnn_params`.
