import torch
import dgl

def digl_rewired(
    g: dgl.DGLGraph,
    *,
    method: str = 'ppr',      # 'ppr' or 'heat'
    alpha: float = 0.15,
    t: float = 5.0,
    k: int | None = None,
    eps: float | None = None,
    avg_degree: int | None = None,
    self_loop_weight: float = 1.0,
) -> dgl.DGLGraph:
    """
    Rewire a DGL graph using diffusion-based sparsification.

    Parameters
    ----------
    g: input graph (unweighted and homogeneous)
    method: 'ppr' for personalised PageRank or 'heat' for the heat kernel
    alpha: teleport probability for PPR
    t: diffusion time for the heat kernel
    k: if set, keep the k largest diffusion values per node (top-k sparsification)
    eps: absolute threshold; keep all diffusion values >= eps
    avg_degree: target average degree when neither k nor eps is specified
    self_loop_weight: weight of the self-loop added to each node

    Returns
    -------
    rewired DGL graph with edge weights in new_g.edata['weight']
    """
    num_nodes = g.number_of_nodes()
    device = g.device

    # Add self-loops and get symmetric adjacency matrix
    g_with_self_loops = dgl.add_self_loop(g)
    A = g_with_self_loops.adjacency_matrix().to_dense().float().to(device)

    # Symmetric normalization
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    T = deg_inv_sqrt.view(-1, 1) * A * deg_inv_sqrt.view(1, -1)

    # Diffusion
    I = torch.eye(num_nodes, device=device, dtype=T.dtype)
    if method == 'ppr':
        diff_matrix = alpha * torch.linalg.inv(I - (1 - alpha) * T)
    elif method == 'heat':
        diff_matrix = torch.matrix_exp(-t * (I - T))
    else:
        raise ValueError(f"Unknown diffusion method: {method}")

    # Sparsification
    if k is not None:
        top_k_vals, _ = torch.topk(diff_matrix, k, dim=1)
        min_vals = top_k_vals[:, -1].unsqueeze(1)
        diff_matrix[diff_matrix < min_vals] = 0
    elif eps is not None:
        diff_matrix[diff_matrix < eps] = 0
    elif avg_degree is not None:
        num_edges_to_keep = avg_degree * num_nodes
        if num_edges_to_keep < diff_matrix.numel():
            threshold = torch.topk(diff_matrix.flatten(), num_edges_to_keep).values[-1]
            diff_matrix[diff_matrix < threshold] = 0
    else:
        raise ValueError("Either k, eps, or avg_degree must be specified for sparsification.")

    # Create new graph
    src, dst = diff_matrix.nonzero(as_tuple=True)
    weights = diff_matrix[src, dst]
    new_g = dgl.graph((src, dst), num_nodes=num_nodes, device=device)
    new_g.edata['weight'] = weights

    # Copy node features
    for key, value in g.ndata.items():
        new_g.ndata[key] = value

    return new_g