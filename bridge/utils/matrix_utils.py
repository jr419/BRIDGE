"""
Matrix operations for graph analysis and rewiring.

This module provides functions for working with matrices in the context of
graph analysis, particularly for stochastic block models and permutation matrices.
"""

import torch
import numpy as np
import math
from itertools import permutations
from ortools.linear_solver import pywraplp
from typing import Tuple, List, Dict, Union, Optional, Any
import scipy.stats as stats

def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute the mean and confidence interval using the t-distribution.
    Suitable for small sample sizes when the population standard deviation is unknown.

    Args:
        data: List of numerical values.
        confidence: Desired confidence level (e.g., 0.95 for 95% CI).

    Returns:
        Tuple[float, float, float]:
            - Mean value
            - Lower bound of the confidence interval
            - Upper bound of the confidence interval
        Returns (mean, NaN, NaN) if sample size is less than 2.
    """
    arr = np.array(data)
    n = len(arr)

    if n < 2:
        # Cannot compute std deviation or CI with less than 2 points
        mean_ = np.mean(arr) if n == 1 else np.nan # Handle n=0 and n=1
        print("Warning: Sample size < 2, cannot compute standard deviation or confidence interval.")
        return mean_, np.nan, np.nan

    mean_ = np.mean(arr)
    std_ = np.std(arr, ddof=1)  # Sample standard deviation (uses n-1 in denominator)

    # Degrees of freedom
    df = n - 1

    # Calculate the critical t-value using the percent point function (ppf),
    # which is the inverse of the cumulative distribution function (CDF).
    # For a two-sided interval with confidence C, we want the value t such that
    # the area between -t and +t is C. This leaves (1-C)/2 in each tail.
    # So, we look up the t-value for the cumulative probability C + (1-C)/2 = (1+C)/2.
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Calculate the margin of error (half-width)
    # Standard Error (SE) = std_ / sqrt(n)
    margin_of_error = t_crit * (std_ / math.sqrt(n))

    # Calculate the confidence interval
    lower_bound = mean_ - margin_of_error
    upper_bound = mean_ + margin_of_error

    return mean_, lower_bound, upper_bound


def infer_B(g: torch.Tensor, Z: torch.Tensor, sym: bool = True) -> torch.Tensor:
    """
    Infer the Stochastic Block Model (SBM) block matrix parameters from a graph.
    
    Args:
        g: Input graph
        Z: One-hot encoding of the block assignment vector
        sym: Whether to enforce symmetry in the block matrix
        
    Returns:
        torch.Tensor: Inferred block matrix B
    """
    A = g.adjacency_matrix().to_dense()
    n = g.number_of_nodes()
    group_sizes = Z.sum(dim=0)
    B = n * (Z.T @ A @ Z) / (group_sizes.unsqueeze(-1) * group_sizes)
    k = Z.shape[1]
    if sym:
        B[np.triu_indices(k, 1)] = B.T[np.triu_indices(k, 1)]
    return B


def matrix_to_permutation(mat: Union[List[List[int]], np.ndarray]) -> List[int]:
    """
    Convert a permutation matrix to its one-line notation.
    
    If mat[i][j] = 1, then permutation(i) = j.
    
    Args:
        mat: N x N permutation matrix
        
    Returns:
        List[int]: Permutation in one-line notation (0-based)
    """
    N = len(mat)
    p = [None] * N
    for i in range(N):
        for j in range(N):
            if mat[i][j] == 1:
                p[i] = j
                break
    return p


def count_2cycles(permutation: List[int]) -> int:
    """
    Count the number of 2-cycles in the cycle decomposition of a permutation.
    
    Args:
        permutation: Permutation in one-line notation (0-based)
        
    Returns:
        int: Number of 2-cycles in the permutation
    """
    N = len(permutation)
    visited = [False] * N
    transposition_count = 0
    
    for start in range(N):
        if not visited[start]:
            # Follow the cycle starting at 'start'
            length = 0
            current = start
            while not visited[current]:
                visited[current] = True
                current = permutation[current]
                length += 1
            
            # If the cycle length is 2, it contributes exactly 1 to the count
            if length == 2:
                transposition_count += 1
    
    return transposition_count


def sort_involutions(matrices: List[np.ndarray]) -> List[np.ndarray]:
    """
    Sort a list of permutation matrices by their distance from the identity.
    
    First sorts by the number of 2-cycles (distance from identity),
    then by lexicographic order of their one-line notation.
    
    Args:
        matrices: List of permutation matrices
        
    Returns:
        List[np.ndarray]: Sorted list of permutation matrices
    """
    def sort_key(mat):
        p = matrix_to_permutation(mat)
        # Primary key: number of 2-cycles
        d = count_2cycles(p)
        # Secondary key: the permutation itself (for tie-break)
        return (d, p)
    
    return sorted(matrices, key=sort_key)


def closest_symmetric_permutation_matrix(B: np.ndarray) -> np.ndarray:
    """
    Find the closest symmetric permutation matrix to a given square matrix.
    
    Uses linear programming to solve the assignment problem with symmetry constraints.
    
    Args:
        B: Square matrix to approximate
        
    Returns:
        np.ndarray: The closest symmetric permutation matrix to B
    """
    n = B.shape[0]
    assert B.shape[1] == n, "B must be a square matrix"
    
    # Compute cost matrix C where c_ij = 1 - 2*b_ij
    C = 1 - 2 * B

    # Initialize the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("Could not create solver.")
        return None

    # Decision variables: x_ij = 1 if nodes i and j are matched
    x = {}
    for i in range(n):
        for j in range(i, n):
            x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')

    # Objective function: minimize sum of costs c_ij * x_ij
    objective = solver.Objective()
    for i in range(n):
        for j in range(i, n):
            objective.SetCoefficient(x[i, j], C[i, j])
    objective.SetMinimization()

    # Constraints: Each node must be matched exactly once
    for i in range(n):
        constraint = solver.Constraint(1, 1)
        # Sum over x_ij where i <= j
        for j in range(i, n):
            constraint.SetCoefficient(x[i, j], 1)
        # Sum over x_ji where j < i
        for j in range(0, i):
            constraint.SetCoefficient(x[j, i], 1)

    # Solve the ILP
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print("The solver did not find an optimal solution.")
        return None

    # Extract the matching from the solution
    matching = []
    for i in range(n):
        for j in range(i, n):
            if x[i, j].solution_value() > 0.5:
                matching.append((i, j))

    # Construct the symmetric permutation matrix P
    P = np.zeros((n, n))
    for i, j in matching:
        if i == j:
            P[i, i] = 1
        else:
            P[i, j] = 1
            P[j, i] = 1

    return P


def generate_all_symmetric_permutation_matrices(k: int) -> List[np.ndarray]:
    """
    Generate all possible k×k symmetric permutation matrices.
    
    Args:
        k: Size of the matrices
        
    Returns:
        List[np.ndarray]: List of all symmetric permutation matrices of size k×k
    """
    def is_symmetric(P):
        return np.array_equal(P, P.T)
    
    # Generate base permutation matrices
    all_perms = list(permutations(range(k)))
    all_matrices = []
    
    # Convert each permutation to a matrix
    perm_matrices = []
    for perm in all_perms:
        P = np.zeros((k, k))
        for i, j in enumerate(perm):
            P[i, j] = 1
        perm_matrices.append(P)
    
    # Generate all possible symmetric combinations
    for P in perm_matrices:
        # Check if it's symmetric and not already in the list
        if is_symmetric(P) and not any(np.array_equal(P, existing) for existing in all_matrices):
            all_matrices.append(P)

    return sort_involutions(all_matrices)


def optimal_B(
    g: torch.Tensor,
    y_label: torch.Tensor,
    y_adj: torch.Tensor,
    P_k: np.ndarray,
    lam: float = 0.5,
    k: Optional[int] = None,
    d_out: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the optimal block matrix for a given graph and permutation matrix.
    
    Args:
        g: Input graph
        y_label: Node labels
        y_adj: Adjacency matrix
        P_k: Symmetric permutation matrix to use
        lam: Regularization parameter
        k: Number of unique labels (if None, inferred from y_label)
        d_out: Desired output mean degree (if None, inferred from the graph)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Optimal block matrix
            - Original block matrix
    """
    n = g.number_of_nodes()
    if k is None:
        k = len(np.unique(y_label))
    if d_out is None:
        d_out = g.in_degrees().float().mean().item()
    
    Z = torch.zeros((n, k))
    Z[torch.arange(n), y_adj] = 1

    B = infer_B(g.cpu(), Z).numpy()
    B = np.nan_to_num(B.astype(float))

    pi = Z.numpy().sum(0) / n
    Pi_inv = np.diag(1/pi)

    B_opt = (d_out/k) * Pi_inv @ P_k @ Pi_inv
    return B_opt, B
