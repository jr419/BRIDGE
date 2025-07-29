"""
Efficient DIGL (Diffusion Improves Graph Learning) rewiring for DGL.

This module re‑implements the rewiring procedure described in
“Diffusion Improves Graph Learning” (Klicpera et al., 2019) using the
exact formulation employed in the official PyTorch Geometric
implementation of Graph Diffusion Convolution (GDC)【318729141242744†L160-L198】.

Given an input :class:`dgl.DGLGraph`, the routine computes a dense
diffusion kernel (personalised PageRank or heat kernel) on a normalised
transition matrix, sparsifies it by either a per‑node top–:math:`k`
operator or a global threshold, and constructs a new DGL graph from
the retained edges.  Edge weights are stored in the ``edata['weight']``
field; node features are copied from the original graph.

Compared to the earlier ``digl_rewired`` implementation in ``bridge/rewiring/digl.py``
the present implementation avoids expensive Python loops and uses
closed‑form matrix inverses or exponentials.  It also follows the
normalisation and sparsification strategy of GDC: normalise the
adjacency matrix into a transition matrix, compute the diffusion
matrix exactly, then sparsify the result to achieve a desired
average degree or top‑:math:`k` neighbourhood【318729141242744†L160-L198】.

The original implementation computed diffusion via power iteration
and selected edges by comparing diffusion scores to the existing
adjacency.  This is both slow and prone to producing disconnected
graphs: if diffusion scores on existing edges are all above the
threshold ``epsilon`` the removal step never triggers, while large
``add_ratio`` values can introduce many weak connections.  In
contrast, this reimplementation directly sparsifies the dense
diffusion matrix, ensuring that exactly the most important
connections (according to diffusion) are preserved and resulting
graphs remain connected under reasonable hyper‑parameters.

Example usage::

    import dgl
    from digl_dgl import digl_rewire
    g = dgl.rand_graph(100, 300)
    rewired = digl_rewire(g, method='ppr', alpha=0.1, avg_degree=32)
    # rewired is a DGLGraph whose edges correspond to the top diffusion
    # connections.  Edge weights are stored in rewired.edata['weight'].

"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import dgl

def _normalised_transition_matrix(
    g: dgl.DGLGraph,
    normalization: str = 'sym',
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a dense transition matrix from a DGL graph.

    Parameters
    ----------
    g:
        Input DGL graph.  Must be homogeneous and unweighted.  If the
        graph has multiple edges between the same pair of nodes they are
        collapsed to a single connection; DGL graphs store edges in
        adjacency lists so this does not affect functionality.
    normalization:
        Type of normalisation to apply: ``'sym'`` (symmetric),
        ``'row'`` (row‑stochastic), ``'col'`` (column‑stochastic) or
        ``None`` (no normalisation).  See the PyTorch Geometric
        documentation for ``GDC.transition_matrix`` for definitions【318729141242744†L213-L266】.
    device:
        Optional device on which to perform the computation.  If
        ``None`` the graph's device is used.

    Returns
    -------
    torch.Tensor
        A dense transition matrix of shape ``(n_nodes, n_nodes)``.
    """
    n = g.number_of_nodes()
    device = device or g.device

    # Obtain dense unweighted adjacency.
    # DGL returns a sparse tensor; convert to dense and cast to float.
    A = g.adjacency_matrix().to_dense().float().to(device)

    # Degree of each node (out degree equals in degree for undirected
    # graphs).  Avoid division by zero by clamping degrees to at least 1.
    deg = g.out_degrees().float().to(device)
    deg[deg == 0] = 1.0

    if normalization == 'sym':
        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        inv_sqrt_deg = 1.0 / torch.sqrt(deg)
        T = inv_sqrt_deg.view(n, 1) * A * inv_sqrt_deg.view(1, n)
    elif normalization == 'row':
        # Row normalisation: D^{-1} A
        T = A / deg.view(n, 1)
    elif normalization == 'col':
        # Column normalisation: A D^{-1}
        T = A / deg.view(1, n)
    elif normalization is None:
        T = A
    else:
        raise ValueError(f"unknown normalization '{normalization}'")

    return T


def _diffusion_matrix(
    T: torch.Tensor,
    method: str = 'ppr',
    alpha: float = 0.15,
    t: float = 5.0,
    coeffs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a dense diffusion matrix from a transition matrix.

    This function mirrors the exact diffusion implementation in
    :class:`torch_geometric.transforms.GDC`【318729141242744†L268-L329】.  It
    supports personalised PageRank (PPR), the heat kernel and a
    user‑specified polynomial in the transition matrix.  All diffusion
    matrices are computed exactly; for PPR and heat this requires
    inverting or exponentiating a dense matrix and is therefore best
    suited to medium‑sized graphs (up to a few thousand nodes).

    Parameters
    ----------
    T:
        Dense transition matrix (usually normalised adjacency).  Must
        have shape ``(n, n)``.
    method:
        Diffusion method: ``'ppr'`` (default), ``'heat'`` or ``'coeff'``.
    alpha:
        Teleport probability for PPR.  Ignored for other methods.
    t:
        Diffusion time for the heat kernel.  Ignored for other methods.
    coeffs:
        Sequence of coefficients :math:`(\theta_0, \dots, \theta_K)` used
        when ``method='coeff'``.  The diffusion matrix is then
        :math:`\sum_k \theta_k T^k`.  The zeroth coefficient multiplies
        the identity.

    Returns
    -------
    torch.Tensor
        Dense diffusion matrix of shape ``(n, n)``.
    """
    n = T.size(0)
    I = torch.eye(n, device=T.device, dtype=T.dtype)

    if method == 'ppr':
        # Following GDC: diff = alpha * (I + (alpha - 1) * T)^{-1}
        # This is equivalent to alpha * (I - (1 - alpha) * T)^{-1} when T
        # is row‑ or symmetric‑normalised.  See Eq. (4) in the paper.
        M = I + (alpha - 1.0) * T
        diff = alpha * torch.linalg.inv(M)
    elif method == 'heat':
        # Heat kernel: diff = exp(t * (T - I))
        M = (T - I) * t
        # When T is symmetric the matrix exponential is symmetric, and
        # eigen decomposition is numerically stable.  Otherwise fall
        # back to torch.matrix_exp (PyTorch >= 2.0) or SciPy.
        try:
            # Using eigen decomposition ensures symmetry is preserved.
            eigvals, eigvecs = torch.linalg.eigh(M)
            diff = eigvecs @ torch.diag(torch.exp(eigvals)) @ eigvecs.T
        except RuntimeError:
            # If eigen decomposition fails (e.g. non‑symmetric input)
            # resort to the general matrix exponential.  PyTorch
            # provides this in torch.matrix_exp.  If unavailable,
            # users should install SciPy and use scipy.linalg.expm.
            if hasattr(torch, 'matrix_exp'):
                diff = torch.matrix_exp(M)
            else:
                from scipy.linalg import expm  # type: ignore
                diff = torch.from_numpy(expm(M.cpu().numpy())).to(M.device)
    elif method == 'coeff':
        if coeffs is None:
            raise ValueError("'coeffs' must be provided for method='coeff'")
        K = len(coeffs)
        # Start with theta_0 * I
        diff = coeffs[0] * I
        T_power = I.clone()
        for k in range(1, K):
            T_power = T_power @ T
            diff = diff + coeffs[k] * T_power
    else:
        raise ValueError(f"unknown diffusion method '{method}'")
    return diff


def _sparsify_diffusion(
    diff: torch.Tensor,
    *,
    k: Optional[int] = None,
    eps: Optional[float] = None,
    avg_degree: Optional[int] = None,
    retain_self_loops: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparsify a dense diffusion matrix by thresholding or top‑:math:`k`.

    Parameters
    ----------
    diff:
        Dense diffusion matrix of shape ``(n, n)``.
    k:
        If given, keep the top ``k`` largest diffusion weights *per
        column* of ``diff``.  This mirrors the ``topk`` sparsification
        option in PyTorch Geometric【318729141242744†L427-L499】.
    eps:
        Absolute threshold.  If provided, keep all entries
        :math:`\ge \text{eps}`.
    avg_degree:
        Desired average degree (number of edges per node) when
        ``eps`` is not specified.  If ``eps`` is ``None`` and
        ``avg_degree`` is specified, the threshold is computed as the
        ``avg_degree * n``‑th largest entry of ``diff``【318729141242744†L427-L499】.
    retain_self_loops:
        Whether to keep self‑loops in the sparsified matrix.  When
        ``False`` (default) the diagonal of ``diff`` is ignored.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(edge_index, edge_weight)``.  ``edge_index`` is a 2×M tensor
        containing source and destination indices of retained edges,
        while ``edge_weight`` holds the corresponding diffusion values.
    """
    n = diff.size(0)
    device = diff.device

    # Zero out diagonal unless self‑loops are explicitly retained.
    if not retain_self_loops:
        diff = diff.clone()
        diff.fill_diagonal_(0.0)

    if k is not None:
        # Ensure k does not exceed the number of nodes
        k = int(min(k, n))
        # Sort each column in descending order and pick top k row indices
        # indices shape: (n, k) if we sort along dim=0 (rows).  We
        # transpose so that we can easily repeat column indices.
        sorted_idx = torch.argsort(diff, dim=0, descending=True)
        top_idx = sorted_idx[:k, :]  # top k rows for each column
        # Flatten to 1D lists
        row_idx = top_idx.flatten()
        col_idx = torch.arange(n, device=device).repeat_interleave(k)
        edge_weight = diff[row_idx, col_idx]
        edge_index = torch.stack([row_idx, col_idx], dim=0)
    else:
        # Determine threshold eps.  If eps is None, compute it from
        # avg_degree
        flat = diff.flatten()
        if eps is None:
            if avg_degree is None:
                raise ValueError(
                    "either 'eps' or 'avg_degree' must be provided when k is None"
                )
            # Number of edges to keep = avg_degree * n
            num_keep = int(avg_degree * n)
            # If num_keep exceeds total entries, set eps to -inf so all
            # edges are kept.
            if num_keep <= 0:
                eps = float('inf')
            elif num_keep >= flat.numel():
                eps = float('-inf')
            else:
                sorted_flat = torch.sort(flat, descending=True).values
                eps = sorted_flat[num_keep - 1].item()
        # Build mask of entries exceeding threshold
        mask = diff >= eps
        # Extract coordinates
        row_idx, col_idx = mask.nonzero(as_tuple=True)
        edge_weight = diff[row_idx, col_idx]
        edge_index = torch.stack([row_idx, col_idx], dim=0)
    return edge_index, edge_weight


def digl_rewired(
    g: dgl.DGLGraph,
    *,
    method: str = 'ppr',
    alpha: float = 0.15,
    t: float = 5.0,
    coeffs: Optional[torch.Tensor] = None,
    normalization: str = 'sym',
    k: Optional[int] = None,
    eps: Optional[float] = None,
    avg_degree: Optional[int] = 64,
    retain_self_loops: bool = False,
    diffusion_type = 'ppr',
) -> dgl.DGLGraph:
    """Rewire a graph using diffusion‐based sparsification.

    This function implements an exact version of the DIGL rewiring
    procedure following the PyTorch Geometric ``GDC`` transform【318729141242744†L160-L198】.  It
    normalises the adjacency matrix, computes a diffusion kernel, sparsifies
    it by either keeping the top ``k`` neighbours per node or using a
    global threshold/average degree, and constructs a new graph with
    diffusion weights on the edges.  Node features are copied from the
    original graph.

    Parameters
    ----------
    g:
        Input DGL graph.  The graph should be unweighted; if it
        contains multiple edges they are treated as a single connection.
    method:
        Diffusion method: ``'ppr'`` (personalised PageRank), ``'heat'``
        (heat kernel) or ``'coeff'`` (polynomial in the transition
        matrix).  Defaults to ``'ppr'``.
    alpha:
        Teleport probability for PPR.  Only used when ``method='ppr'``.
    t:
        Diffusion time for the heat kernel.  Only used when
        ``method='heat'``.
    coeffs:
        Coefficients for the ``'coeff'`` diffusion method.  Should be a
        one‑dimensional tensor of length ``K`` with ``theta_k`` values.
    normalization:
        Normalisation applied to the adjacency prior to diffusion.
        Choices are ``'sym'``, ``'row'``, ``'col'`` or ``None``.  The
        default ``'sym'`` uses symmetric normalisation (``D^{-1/2}AD^{-1/2}``).
    k:
        If specified, retain the top ``k`` largest diffusion weights per
        target node (column).  In this case ``eps`` and ``avg_degree``
        are ignored.  For unweighted graphs a typical value is in the
        tens (e.g. 32 or 64).
    eps:
        Absolute threshold for diffusion values.  All edges with
        weights below ``eps`` are discarded.  If both ``eps`` and ``k``
        are ``None``, ``avg_degree`` is used to compute a threshold.
    avg_degree:
        Desired average degree of the rewired graph when neither
        ``k`` nor ``eps`` are provided.  The threshold is chosen so that
        approximately ``avg_degree * num_nodes`` edges are kept.
    retain_self_loops:
        Whether to keep self loops in the rewired graph.  Defaults to
        ``False``.
    diffusion_type:
        Deprecated; use ``method`` instead.  This parameter is retained
        for compatibility with the old ``digl_rewired`` function in
        ``bridge/rewiring/digl.py``.  It is ignored in the present
        implementation.

    Returns
    -------
    dgl.DGLGraph
        The rewired graph.  The edge weights (diffusion scores) are
        stored in ``edata['weight']``.  Node features from the input
        graph are copied to the output.
    """
    # Normalise adjacency to a transition matrix
    T = _normalised_transition_matrix(g, normalization=normalization)
    # Compute diffusion matrix exactly
    diff = _diffusion_matrix(T, method=method, alpha=alpha, t=t, coeffs=coeffs)
    # Symmetrise the diffusion matrix to ensure undirected graphs
    diff = (diff + diff.T) * 0.5
    # Sparsify diffusion matrix to obtain edges and weights
    edge_index, edge_weight = _sparsify_diffusion(
        diff,
        k=k,
        eps=eps,
        avg_degree=avg_degree,
        retain_self_loops=retain_self_loops,
    )

    # Construct new DGL graph.  ``edge_index`` is 2×M; its rows are
    # source and destination indices.  DGL expects a pair of index
    # tensors for edges.
    src, dst = edge_index
    new_g = dgl.graph((src, dst), num_nodes=g.number_of_nodes(), device=g.device)

    # Attach diffusion weights to edges
    new_g.edata['weight'] = edge_weight

    # Copy node features from the original graph
    for key, feat in g.ndata.items():
        new_g.ndata[key] = feat

    return new_g