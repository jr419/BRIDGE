"""
Graph Convolutional Network models for sensitivity analysis.

This module provides GNN model implementations specifically designed for
sensitivity analysis in the context of graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Any, Dict, Tuple, Union, Optional

from bridge.utils.graph_utils import dense_adjacency


class LinearGCN(nn.Module):
    """
    A simple linear GCN layer: h = Â @ x @ W.

    The normalized adjacency matrix Â is computed as D^{-1/2}AD^{-1/2}, where
    A is the adjacency matrix with self-loops and D is the degree matrix.
    This model does not include biases or nonlinearities.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden feature dimension (unused in this implementation)
        out_feats: Output feature dimension
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # Just a linear transformation; no biases
        self.weight1 = nn.Parameter(torch.randn(in_feats, out_feats))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LinearGCN.
        
        Args:
            graph: A DGLGraph (the adjacency will be normalized)
            x: Node features, either:
              - shape (num_nodes, in_feats) for a single feature sample, OR
              - shape (num_nodes, in_feats, num_samples) for multiple samples.

        Returns:
            If x is 2D: shape (num_nodes, out_feats).
            If x is 3D: shape (num_nodes, num_samples, out_feats).
        """
        # -- 1) Obtain adjacency as a dense matrix on the same device as x
        A = dense_adjacency(graph, device=x.device, dtype=torch.double)

        # -- 2) Normalize the adjacency matrix
        N = A.shape[0]
        d_arr = A.sum(1).double()
        # Handle isolated nodes by adding self-loops
        A[d_arr==0,:][:,d_arr==0] = torch.eye(N).to(x.device).double()[d_arr==0,:][:,d_arr==0]
        d_arr[d_arr==0] = 1
        d_inv_arr = (1/d_arr**(1/2))
        A_norm = d_inv_arr[:,None]*A*d_inv_arr[None,:]

        if x.dim() == 2:
            # x: [N, in_feats]
            # => output: [N, out_feats]
            return A_norm @ x.double() @ self.weight1

        elif x.dim() == 3:
            # x: [N, F, S]
            N, F, S = x.shape

            # Flatten out the last dimension for a single matmul with A_norm
            x_2d = x.reshape(N, F*S)   # [N, F*S]
            Ax = A_norm @ x_2d         # => [N, F*S]

            # Reshape back to 3D => [N, F, S]
            Ax_3d = Ax.view(N, F, S)

            # Multiply by weight1 [F, O], to get shape [N, S, O].
            # Using einsum 'nfs,fo->nos':
            h_3d = torch.einsum('nfs,fo->nos', Ax_3d, self.weight1)

            # shape => [N, S, O]
            return h_3d

        else:
            raise ValueError("Unsupported input dimensionality for x.")
        

class FNN(nn.Module):
    """
    A simple linear layer: h = Â @ x @ W.

    The normalized adjacency matrix Â is computed as D^{-1/2}AD^{-1/2}, where
    A is the adjacency matrix with self-loops and D is the degree matrix.
    This model does not include biases or nonlinearities.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden feature dimension (unused in this implementation)
        out_feats: Output feature dimension
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # Just a linear transformation; no biases
        self.weight1 = nn.Parameter(torch.randn(in_feats, out_feats))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LinearGCN.
        
        Args:
            graph: A DGLGraph (the adjacency will be normalized)
            x: Node features, either:
              - shape (num_nodes, in_feats) for a single feature sample, OR
              - shape (num_nodes, in_feats, num_samples) for multiple samples.

        Returns:
            If x is 2D: shape (num_nodes, out_feats).
            If x is 3D: shape (num_nodes, num_samples, out_feats).
        """

        if x.dim() == 2:
            # x: [N, in_feats]
            # => output: [N, out_feats]
            return x.double() @ self.weight1

        elif x.dim() == 3:
            # x: [N, F, S]
            N, F, S = x.shape

            # Flatten out the last dimension for a single matmul with A_norm
            x_2d = x.reshape(N, F*S)   # [N, F*S]

            # Reshape back to 3D => [N, F, S]
            x_3d = x_2d.view(N, F, S)

            # Multiply by weight1 [F, O], to get shape [N, S, O].
            # Using einsum 'nfs,fo->nos':
            h_3d = torch.einsum('nfs,fo->nos', x_3d, self.weight1)

            # shape => [N, S, O]
            return h_3d

        else:
            raise ValueError("Unsupported input dimensionality for x.")


class TwoLayerGCN(nn.Module):
    """
    A two-layer GCN with ReLU activation between layers.
    
    This model applies two graph convolution layers with a ReLU nonlinearity
    after the first layer, and includes trainable bias terms.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden feature dimension
        out_feats: Output feature dimension
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # The +1 accounts for the bias term, which is appended as a feature
        self.weight1 = nn.Parameter(torch.randn(in_feats+1, hidden_feats))
        self.weight2 = nn.Parameter(torch.randn(hidden_feats+1, out_feats))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TwoLayerGCN.
        
        Args:
            graph: A DGLGraph (the adjacency will be normalized)
            x: Node features, either:
              - shape (num_nodes, in_feats) for a single feature sample, OR
              - shape (num_nodes, in_feats, num_samples) for multiple samples.

        Returns:
            If x is 2D: shape (num_nodes, out_feats).
            If x is 3D: shape (num_nodes, num_samples, out_feats).
        """
        # -- 1) Obtain adjacency as a dense matrix on the same device as x
        A = dense_adjacency(graph, device=x.device, dtype=torch.double)

        # -- 2) Normalize the adjacency matrix
        N = A.shape[0]
        d_arr = A.sum(1).double()
        # Handle isolated nodes by adding self-loops
        A[d_arr==0,:][:,d_arr==0] = torch.eye(N).to(x.device).double()[d_arr==0,:][:,d_arr==0]
        d_arr[d_arr==0] = 1
        d_inv_arr = (1/d_arr**(1/2))
        A_norm = d_inv_arr[:,None]*A*d_inv_arr[None,:]

        if x.dim() == 2:
            # x: [N, in_feats]
            # => output: [N, out_feats]
            # Append bias feature
            h = torch.cat((x, torch.ones((x.shape[0], 1), device=x.device)), dim=1)
            h = F.relu(A_norm @ h.double() @ self.weight1)
            h = torch.cat((h, torch.ones((h.shape[0], 1), device=h.device)), dim=1)
            return A_norm @ h @ self.weight2

        elif x.dim() == 3:
            # x: [N, F, S]
            h = x
            
            # Process through both layers
            for i, W in enumerate([self.weight1, self.weight2]):
                # Append bias feature along feature dimension
                h = torch.cat((h, torch.ones((h.shape[0], 1, h.shape[2]), device=h.device)), dim=1)
                
                N, F_num, S = h.shape
                
                # Flatten last dimension for adjacency matrix multiplication
                h_2d = h.reshape(N, F_num*S)   # [N, F_num*S]
                Ah = A_norm @ h_2d           # => [N, F_num*S]
                
                # Reshape back to 3D => [N, F_num, S]
                Ah_3d = Ah.view(N, F_num, S)
                
                # Multiply by weight [F_num, O], to get shape [N, S, O].
                h = torch.einsum('nfs,fo->nos', Ah_3d, W)
                
                # Apply ReLU after the first layer only
                if i == 0:
                    h = F.relu(h)
                    
            return h

        else:
            raise ValueError("Unsupported input dimensionality for x.")


class TwoLayerFNN(nn.Module):
    """
    A two-layer feedforward neural network with ReLU activation between layers.
    
    This model applies two linear transformations with a ReLU nonlinearity
    after the first layer, and includes trainable bias terms.
    
    Args:
        in_feats: Input feature dimension
        hidden_feats: Hidden feature dimension
        out_feats: Output feature dimension
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # The +1 accounts for the bias term, which is appended as a feature
        self.weight1 = nn.Parameter(torch.randn(in_feats+1, hidden_feats))
        self.weight2 = nn.Parameter(torch.randn(hidden_feats+1, out_feats))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TwoLayerFNN.
        
        Args:
            graph: A DGLGraph (not used but kept for consistency)
            x: Node features, either:
              - shape (num_nodes, in_feats) for a single feature sample, OR
              - shape (num_nodes, in_feats, num_samples) for multiple samples.

        Returns:
            If x is 2D: shape (num_nodes, out_feats).
            If x is 3D: shape (num_nodes, num_samples, out_feats).
        """

        if x.dim() == 2:
            # x: [N, in_feats]
            # => output: [N, out_feats]
            # Append bias feature
            h = torch.cat((x, torch.ones((x.shape[0], 1), device=x.device)), dim=1)
            h = F.relu(h.double() @ self.weight1)
            h = torch.cat((h, torch.ones((h.shape[0], 1), device=h.device)), dim=1)
            return h @ self.weight2

        elif x.dim() == 3:
            # x: [N, F, S]
            h = x
            
            # Process through both layers
            for i, W in enumerate([self.weight1, self.weight2]):
                # Append bias feature along feature dimension
                h = torch.cat((h, torch.ones((h.shape[0], 1, h.shape[2]), device=h.device)), dim=1)
                
                N, F_num, S = h.shape
                
                # No adjacency matrix multiplication for FNN - just apply weight
                # Multiply by weight [F_num, O], to get shape [N, S, O].
                h = torch.einsum('nfs,fo->nos', h, W)
                
                # Apply ReLU after the first layer only
                if i == 0:
                    h = F.relu(h)
                    
            return h

        else:
            raise ValueError("Unsupported input dimensionality for x.")


def _normalized_adjacency(graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
    """Return D^{-1/2} A D^{-1/2} using exactly the graph edges provided."""
    A = dense_adjacency(graph, device=x.device, dtype=x.dtype)
    degrees = A.sum(1)
    d_inv_sqrt = degrees.clamp_min(1).pow(-0.5)
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


def _propagate(A_norm: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    if h.dim() == 2:
        return A_norm @ h
    if h.dim() == 3:
        n, feat_dim, num_samples = h.shape
        propagated = A_norm @ h.reshape(n, feat_dim * num_samples)
        return propagated.view(n, feat_dim, num_samples)
    raise ValueError("Unsupported input dimensionality for h.")


def _apply_linear(layer: nn.Linear, h: torch.Tensor) -> torch.Tensor:
    if h.dim() == 2:
        return layer(h)
    if h.dim() == 3:
        out = torch.einsum("nfs,of->nos", h, layer.weight)
        if layer.bias is not None:
            out = out + layer.bias[None, :, None]
        return out
    raise ValueError("Unsupported input dimensionality for h.")


class FeedForwardNN(nn.Module):
    """Graph-free baseline with the same 2D/3D calling convention as GCN backbones."""

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        n_layers: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_feats, hidden_feats, bias=bias)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_feats, hidden_feats, bias=bias)
            for _ in range(n_layers)
        )
        self.output_proj = nn.Linear(hidden_feats, out_feats, bias=bias)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(_apply_linear(self.input_proj, x.double()))
        for layer in self.hidden_layers:
            h = F.relu(_apply_linear(layer, h))
        return _apply_linear(self.output_proj, h)


class VanillaGCN(nn.Module):
    """Isotropic GCN backbone with input projection, propagation blocks, and classifier."""

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        n_layers: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_feats, hidden_feats, bias=bias)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_feats, hidden_feats, bias=bias)
            for _ in range(n_layers)
        )
        self.output_proj = nn.Linear(hidden_feats, out_feats, bias=bias)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        A_norm = _normalized_adjacency(graph, x)
        h = F.relu(_apply_linear(self.input_proj, x.double()))
        for layer in self.hidden_layers:
            h = _propagate(A_norm, h)
            h = F.relu(_apply_linear(layer, h))
        return _apply_linear(self.output_proj, h)


class InitialResidualGCN(nn.Module):
    """GCN backbone that injects the initial hidden representation at each layer."""

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        n_layers: int = 2,
        alpha: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        self.alpha = alpha
        self.input_proj = nn.Linear(in_feats, hidden_feats, bias=bias)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_feats, hidden_feats, bias=bias)
            for _ in range(n_layers)
        )
        self.output_proj = nn.Linear(hidden_feats, out_feats, bias=bias)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        A_norm = _normalized_adjacency(graph, x)
        h0 = F.relu(_apply_linear(self.input_proj, x.double()))
        h = h0
        for layer in self.hidden_layers:
            propagated = _propagate(A_norm, h)
            transformed = _apply_linear(layer, propagated)
            h = F.relu((1.0 - self.alpha) * transformed + self.alpha * h0)
        return _apply_linear(self.output_proj, h)


class GCNII(nn.Module):
    """GCNII-style backbone with initial residuals and identity mapping."""

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        n_layers: int = 2,
        alpha: float = 0.1,
        lambda_: float = 0.5,
        bias: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        self.alpha = alpha
        self.lambda_ = lambda_
        self.input_proj = nn.Linear(in_feats, hidden_feats, bias=bias)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_feats, hidden_feats, bias=bias)
            for _ in range(n_layers)
        )
        self.output_proj = nn.Linear(hidden_feats, out_feats, bias=bias)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        A_norm = _normalized_adjacency(graph, x)
        h0 = F.relu(_apply_linear(self.input_proj, x.double()))
        h = h0
        for layer_idx, layer in enumerate(self.hidden_layers, start=1):
            propagated = _propagate(A_norm, h)
            p_l = (1.0 - self.alpha) * propagated + self.alpha * h0
            beta_l = torch.log(
                torch.tensor(self.lambda_ / layer_idx + 1.0, dtype=p_l.dtype, device=p_l.device)
            )
            transformed = _apply_linear(layer, p_l)
            h = F.relu((1.0 - beta_l) * p_l + beta_l * transformed)
        return _apply_linear(self.output_proj, h)


def _normalize_binary_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    degrees = adjacency.sum(1)
    d_inv_sqrt = torch.zeros_like(degrees)
    nonzero = degrees > 0
    d_inv_sqrt[nonzero] = degrees[nonzero].pow(-0.5)
    return d_inv_sqrt[:, None] * adjacency * d_inv_sqrt[None, :]


class H2GCN(nn.Module):
    """H2GCN backbone with 1-hop/2-hop propagation and JK-style concat."""

    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        n_layers: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        self.input_proj = nn.Linear(in_feats, hidden_feats, bias=bias)
        final_feats = (2 ** (n_layers + 1) - 1) * hidden_feats
        self.output_proj = nn.Linear(final_feats, out_feats, bias=bias)
        self._adjacency_cache = None

    def _normalized_hop_adjacencies(
        self,
        graph: dgl.DGLGraph,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (
            id(graph),
            graph.num_nodes(),
            graph.num_edges(),
            x.device.type,
            x.device.index,
            x.dtype,
        )
        if self._adjacency_cache is not None and self._adjacency_cache[0] == cache_key:
            return self._adjacency_cache[1], self._adjacency_cache[2]

        adjacency = (dense_adjacency(graph, device=x.device, dtype=x.dtype) > 0).to(dtype=x.dtype)
        n = adjacency.shape[0]
        idx = torch.arange(n, device=x.device)

        two_hop = ((adjacency @ adjacency) > 0).to(dtype=x.dtype)
        two_hop[idx, idx] = 0
        two_hop[adjacency.bool()] = 0

        one_hop_norm = _normalize_binary_adjacency(adjacency)
        two_hop_norm = _normalize_binary_adjacency(two_hop)
        self._adjacency_cache = (cache_key, one_hop_norm, two_hop_norm)
        return one_hop_norm, two_hop_norm

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        one_hop, two_hop = self._normalized_hop_adjacencies(graph, x)

        h = F.relu(_apply_linear(self.input_proj, x.double()))
        representations = [h]

        for _ in range(self.n_layers):
            h = torch.cat((_propagate(one_hop, h), _propagate(two_hop, h)), dim=1)
            representations.append(h)

        h_final = torch.cat(representations, dim=1)
        return _apply_linear(self.output_proj, h_final)


BACKBONE_ALIASES = {
    "vanila_gcn": "vanilla_gcn",
    "vanilla": "vanilla_gcn",
    "initial_residual": "initial_residual_gcn",
    "gcn_ii": "gcnii",
    "h2_gcn": "h2gcn",
    "h2-gcn": "h2gcn",
}


def normalize_backbone_type(backbone_type: str) -> str:
    normalized = backbone_type.lower()
    return BACKBONE_ALIASES.get(normalized, normalized)


def create_sensitivity_model(
    backbone_type: str,
    in_feats: int,
    hidden_feats: int,
    out_feats: int,
    n_layers: int = 2,
    alpha: float = 0.1,
    lambda_: float = 0.5,
    bias: bool = True,
) -> nn.Module:
    """Create a sensitivity GCN backbone by registry name."""
    normalized = normalize_backbone_type(backbone_type)
    if normalized == "vanilla_gcn":
        return VanillaGCN(in_feats, hidden_feats, out_feats, n_layers=n_layers, bias=bias)
    if normalized == "initial_residual_gcn":
        return InitialResidualGCN(
            in_feats,
            hidden_feats,
            out_feats,
            n_layers=n_layers,
            alpha=alpha,
            bias=bias,
        )
    if normalized == "gcnii":
        return GCNII(
            in_feats,
            hidden_feats,
            out_feats,
            n_layers=n_layers,
            alpha=alpha,
            lambda_=lambda_,
            bias=bias,
        )
    if normalized == "h2gcn":
        return H2GCN(in_feats, hidden_feats, out_feats, n_layers=n_layers, bias=bias)
    raise ValueError(f"Unknown sensitivity backbone type: {backbone_type}")


def create_fnn_baseline_model(
    in_feats: int,
    hidden_feats: int,
    out_feats: int,
    n_layers: int = 2,
    bias: bool = True,
) -> nn.Module:
    """Create the graph-free baseline used alongside a sensitivity backbone."""
    return FeedForwardNN(in_feats, hidden_feats, out_feats, n_layers=n_layers, bias=bias)
