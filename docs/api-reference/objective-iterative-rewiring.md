---
layout: default
title: objective_iterative_rewiring
parent: API Reference
---

# objective_iterative_rewiring

This Optuna objective tunes the parameters of the iterative rewiring pipeline. Rewiring uses hard predictions and full resampling (no temperature, no p_add/p_remove). It can also evaluate alternative rewiring methods like SDRF and DIGL. Only standard models are used.

## Signature

```python
objective_iterative_rewiring(
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
    do_residual_connections: bool = False,
    dataset_name: str = 'unknown',
    n_rewire_iterations_range: List[int] = None,
    rewiring_method: str = "bridge",
    sdrf_tau_range: list = [0.01, 300],
    sdrf_n_iterations_range: list = [1, 300],
    sdrf_c_plus_range: list = [0, 50],
    digl_diffusion_type='ppr',
    digl_alpha_range=[0.05, 0.25],
    digl_k_options=[32, 64, 128],
    digl_t_range=[1.0, 10.0],
    digl_epsilon_range=[0.001, 0.1],
    simulated_acc: Optional[float] = None
) -> float
```

## Tuned Parameters

- `matrix_idx`, `d_out`
- Iterative parameters: `n_rewire_iterations`
- Alternative rewiring parameters (if used): SDRF (`sdrf_tau`, `sdrf_iterations`, `sdrf_c_plus`) and DIGL (`digl_diffusion_type`, `digl_alpha`, `digl_k`, `digl_t`, `digl_epsilon`)
