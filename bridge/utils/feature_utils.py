import torch
import dgl                           # 0.9+  (only needed for the graph wrapper)

@torch.no_grad()
def estimate_iid_variances(g,
                           feat_key: str = "feat",
                           label_key: str = "label"):
    """
    Efficiently estimate Var[mu], Var[epsilon] and Var[gamma] on a single
    attributed graph, assuming i.i.d. feature dimensions.

    Parameters
    ----------
    g : dgl.DGLGraph
        Graph with node features (`feat_key`, shape [N, d])
        and integer class labels (`label_key`, shape [N]).
    feat_key : str
        Name of the node‐feature field in g.ndata.
    label_key : str
        Name of the node‐label field in g.ndata.

    Returns
    -------
    dict
        {'var_mu': float, 'var_epsilon': float, 'var_gamma': float}
        – scalar (dimension-averaged) variance estimates.
    """
    x = g.ndata[feat_key]                 # (N, d)
    y = g.ndata[label_key].long()         # (N,)
    N, d = x.shape

    # ---------- γ : global shift -------------------------------------------
    gamma = x.mean(dim=0)                 # (d,)

    # ---------- μ_c : centred class means ----------------------------------
    k = int(y.max().item()) + 1
    counts = torch.bincount(y, minlength=k).unsqueeze(1).type_as(x)  # (k,1)
    sums   = torch.zeros(k, d, dtype=x.dtype, device=x.device)
    sums   = sums.index_add(0, y, x)       # accumulate features per class
    mu_c   = sums / counts                 # per-class mean            (k,d)
    mu_c   = mu_c - gamma                  # enforce  ⟨μ⟩ = 0 ⇒ γ = global mean

    # ---------- ε : node residuals -----------------------------------------
    mu_node = mu_c[y]                      # align μ with each node    (N,d)
    eps     = x - mu_node - gamma          # residual                  (N,d)

    # ---------- variances (dimension-wise then averaged) -------------------
    p_c          = counts / N              # class priors              (k,1)
    var_mu_dim   = (p_c * mu_c.pow(2)).sum(dim=0)       # (d,)
    var_eps_dim  = eps.pow(2).mean(dim=0)               # (d,)
    var_gamma_dim= gamma.pow(2)                          # (d,)

    return {
        'var_mu'     : var_mu_dim.mean().item(),
        'var_epsilon': var_eps_dim.mean().item(),
        'var_gamma'  : var_gamma_dim.mean().item()
    }