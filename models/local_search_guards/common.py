"""Shared helpers for the guarded local search: FP64 energies, tolerances, graph coloring, result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import sparse


class GuardInvariantError(RuntimeError):
    """Raised when a declared local-search invariant is violated."""


@dataclass
class RefinementResult:
    """Standard result returned by every local-search implementation."""

    spins: Any
    local_search_seconds: float
    summary: dict[str, Any] = field(default_factory=dict)
    round_rows: list[dict[str, Any]] = field(default_factory=list)



def canonical_csr(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    result.sum_duplicates()
    result.sort_indices()
    return result



def normalization_factor(matrix: sparse.spmatrix) -> float:
    matrix = canonical_csr(matrix)
    degrees = np.asarray(abs(matrix).sum(axis=1)).ravel()
    if degrees.size == 0:
        raise ValueError("Cannot normalize an empty matrix")
    scale = float(degrees.max())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid normalization factor: {scale}")
    return scale




def energy_tolerance(
    energies: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    xp: Any,
) -> Any:
    energies = xp.asarray(energies)
    scale = xp.maximum(
        xp.asarray(1.0, dtype=energies.dtype), xp.abs(energies)
    )
    return (
        xp.asarray(absolute_tolerance, dtype=energies.dtype)
        + xp.asarray(relative_tolerance, dtype=energies.dtype) * scale
    )


def gpu_batch_energies(
    matrix: Any,
    spins: Any,
    *,
    chunk_size: int = 16,
    candidate_indices: Any | None = None,
    base_energies: Any | None = None,
) -> Any:
    """Evaluate ``-0.5 s^T J s`` in FP64 for candidate-major spins."""
    import cupy as cp

    spins = cp.asarray(spins)
    if spins.ndim != 2:
        raise ValueError(f"Expected candidate-major spins, got {spins.shape}")
    candidate_count, node_count = spins.shape
    matrix64 = matrix.astype(cp.float64)
    if base_energies is None:
        energies = cp.empty(candidate_count, dtype=cp.float64)
    else:
        energies = cp.asarray(base_energies, dtype=cp.float64).copy()
    if candidate_indices is None:
        candidate_indices = cp.arange(candidate_count, dtype=cp.int32)
    candidate_indices = cp.asarray(candidate_indices, dtype=cp.int32)

    for start in range(0, int(candidate_indices.size), int(chunk_size)):
        block_indices = candidate_indices[start : start + int(chunk_size)]
        block = spins[block_indices].T.astype(cp.float64, copy=False)
        product = matrix64.dot(block)
        energies[block_indices] = -0.5 * cp.sum(block * product, axis=0)
    return energies


def gpu_single_flip_deltas(matrix: Any, spins: Any) -> Any:
    """Return FP64 single-flip deltas with shape (candidates, nodes)."""
    import cupy as cp

    spins64 = cp.asarray(spins, dtype=cp.float64)
    fields = matrix.astype(cp.float64).dot(spins64.T)
    diagonal = matrix.diagonal().astype(cp.float64)[:, cp.newaxis]
    spin_columns = spins64.T
    return (
        2.0 * spin_columns * (fields - diagonal * spin_columns)
    ).T


def timed_gpu_call(operation: Callable[[], Any]) -> tuple[Any, float]:
    """Run one GPU operation and return synchronized elapsed milliseconds."""
    import cupy as cp

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return result, float(cp.cuda.get_elapsed_time(start, end))



def peak_cupy_memory() -> dict[str, int]:
    import cupy as cp

    pool = cp.get_default_memory_pool()
    pinned = cp.get_default_pinned_memory_pool()
    return {
        "cupy_used_bytes": int(pool.used_bytes()),
        "cupy_pool_bytes": int(pool.total_bytes()),
        "cupy_pinned_free_blocks": int(pinned.n_free_blocks()),
    }


def greedy_largest_first_coloring(
    matrix: sparse.spmatrix,
) -> np.ndarray:
    """Create a deterministic proper coloring of the structural graph."""
    matrix = canonical_csr(matrix)
    node_count = matrix.shape[0]
    degrees = np.diff(matrix.indptr)
    order = np.lexsort((np.arange(node_count), -degrees))
    colors = np.full(node_count, -1, dtype=np.int32)
    seen = np.full(max(1, node_count), -1, dtype=np.int64)

    for stamp, node in enumerate(order):
        neighbors = matrix.indices[matrix.indptr[node] : matrix.indptr[node + 1]]
        assigned = colors[neighbors]
        assigned = assigned[assigned >= 0]
        if assigned.size:
            seen[assigned] = stamp
        color = 0
        while seen[color] == stamp:
            color += 1
        colors[node] = color

    validate_coloring(matrix, colors)
    return colors


def validate_coloring(
    matrix: sparse.spmatrix, colors: np.ndarray
) -> dict[str, Any]:
    matrix = canonical_csr(matrix)
    colors = np.asarray(colors, dtype=np.int32)
    if colors.shape != (matrix.shape[0],):
        raise GuardInvariantError(
            f"Invalid coloring shape {colors.shape} for {matrix.shape[0]} nodes"
        )
    rows = np.repeat(
        np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr)
    )
    internal = (rows != matrix.indices) & (
        colors[rows] == colors[matrix.indices]
    )
    directed_conflicts = int(np.count_nonzero(internal))
    if directed_conflicts:
        raise GuardInvariantError(
            f"Coloring contains {directed_conflicts // 2} internal edges"
        )
    counts = np.bincount(colors)
    return {
        "color_count": int(colors.max()) + 1 if colors.size else 0,
        "minimum_color_size": int(counts.min()) if counts.size else 0,
        "maximum_color_size": int(counts.max()) if counts.size else 0,
        "directed_internal_edges": directed_conflicts,
    }



