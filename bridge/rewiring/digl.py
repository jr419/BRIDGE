import torch
import dgl

def digl_rewired(
    g: dgl.DGLGraph,
    *,
    method: str = 'ppr',       # 'ppr' or 'heat'
    alpha: float = 0.15,
    t: float = 5.0,
    k: int | None = None,
    eps: float | None = None,
    avg_degree: int | None = None,
    self_loop_weight: float = 1.0,
) -> dgl.DGLGraph:
    """
    Rewire a DGL graph using diffusion‑based sparsification.

    Parameters
    ----------
    g: input graph (unweighted and homogeneous)
    method: 'ppr' for personalised PageRank or 'heat' for the heat kernel
    alpha: teleport probability for PPR
    t: diffusion time for the heat kernel
    k: if set, keep the k largest diffusion values per node (top‑k sparsification)
    eps: absolute threshold; keep all diffusion values >= eps
    avg_degree: target average degree when neither k nor eps is specified
    self_loop_weight: weight of the self‑loop added to each node

    Returns
    -------
    rewired DGL graph with edge weights in new_g.edata['weight']
    """
    num_nodes = g.number_of_nodes()
    device = g.device
    # Dense unweighted adjacency
    A = g.adjacency_matrix().to_dense().float().to(device)
    # Add self‑loops before normalisation (A_tilde = A + I * self_loop_weight)
    A_tilde = A + torch.eye(num_nodes, device=device) * self_loop_weight
    # Degree and symmetric normalisation
    deg = A_tilde.sum(dim=1)
    deg[deg == 0] = 1.0  # avoid division by zero
    inv_sqrt_deg = 1.0 / torch.sqrt(deg)
    T = inv_sqrt_deg.view(num_nodes, 1) * A_tilde * inv_sqrt_deg.view(1, num_nodes)
    # Exact diffusion matrix
    I = torch.eye(num_nodes, device=device, dtype=T.dtype)
    if method == 'ppr':
        # alpha * (I - (1 - alpha) * T)^{-1}
        diff = alpha * torch.linalg.inv(I - (1.0 - alpha) * T)
    elif method == 'heat':
        # exp(-t * (I - T))
        M = -t * (I - T)
        # use eigendecomposition for numerical stability
        eigvals, eigvecs = torch.linalg.eigh(M)
        diff = eigvecs @ torch.diag(torch.exp(eigvals)) @ eigvecs.T
    else:
        raise ValueError(f'unknown diffusion method {method}')
    # Remove self‑loops in diffusion matrix (they will be re‑added if needed)
    diff = diff - torch.diag_embed(torch.diagonal(diff))
    # Sparsify: top‑k per column or threshold
    if k is not None:
        k = int(min(k, num_nodes))
        sorted_idx = torch.argsort(diff, dim=0, descending=True)
        mask = torch.zeros_like(diff, dtype=torch.bool)
        col_idx = torch.arange(num_nodes, device=device).repeat_interleave(k)
        row_idx = sorted_idx[:k, :].flatten()
        mask[row_idx, col_idx] = True
        diff = diff * mask.float()
    else:
        # Flatten and select a global threshold
        flat = diff.flatten()
        if eps is None:
            if avg_degree is None:
                raise ValueError('specify either k, eps or avg_degree')
            # keep avg_degree * num_nodes largest entries
            num_keep = int(avg_degree * num_nodes)
            if num_keep <= 0:
                threshold = float('inf')
            elif num_keep >= flat.numel():
                threshold = float('-inf')
            else:
                threshold = torch.topk(flat, num_keep).values[-1].item()
        else:
            threshold = eps
        diff[diff < threshold] = 0.0
    # Column normalisation as in the reference
    col_sum = diff.sum(dim=0)
    col_sum[col_sum == 0] = 1.0
    diff = diff / col_sum
    # Build undirected edge list
    row_idx, col_idx = (diff > 0).nonzero(as_tuple=True)
    edge_weight = diff[row_idx, col_idx]
    # Add reverse edges to make the graph undirected
    src = torch.cat([row_idx, col_idx])
    dst = torch.cat([col_idx, row_idx])
    weight = torch.cat([edge_weight, edge_weight])
    new_g = dgl.graph((src, dst), num_nodes=num_nodes, device=device)
    new_g.edata['weight'] = weight
    # Copy node features
    for key, value in g.ndata.items():
        new_g.ndata[key] = value
    return new_g
