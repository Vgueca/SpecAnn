"""Guarded local-search policies, a drop-in local search chosen with ``ls_policy``.

SpectralAnnealing, SpecAdj and SpecSignedLap pass their candidates to these refiners with fixed settings.
"""

from __future__ import annotations

import cupy as cp

from .registry import get_refiner

# ls_policy value -> refiner name (paper name in the comment)
POLICIES = {
    "fixed_guard": "Safe-Fixed",               # Fixed Guard
    "adaptive_guard": "Safe-Adaptive",         # Adaptive Guard
    "independent_set": "Conflict-Free",        # Independent-Set
    "sequential_oracle": "Sequential-Oracle",  # exact 1-flip descent (small graphs only)
}
# policies that also need the normalized couplings on the host (graph coloring)
NEEDS_HOST_MATRIX = {"independent_set"}

# Fixed local-search settings used in the paper; max_iterations comes from the solver's
# max_local_iterations.
PAPER_LOCAL_SEARCH = {
    "subset_fraction": 0.9,
    "final_sweep_fraction": 1.0,
    "final_sweep_iters": 25,
    "final_sweep_rounds": 25,
    "energy_chunk_size": 16,
    "absolute_tolerance": 1e-10,
    "relative_tolerance": 1e-12,
    "identity_absolute_tolerance": 1e-9,
    "identity_relative_tolerance": 1e-10,
    "safe_patience": 3,
    "minimum_subset_fraction": 0.01,
    "batch_shrink_factor": 0.5,
    "batch_growth_factor": 1.25,
    "rejection_threshold": 0.25,
    "oracle_max_steps_multiplier": 100,
}


def run_policy(policy, J_csr, spins_list, *, max_iterations, dtype, seed, matrix_cpu=None):
    """Refine the candidates with a guarded policy; return (n x k int8 spins, iterations, summary)."""
    config = dict(PAPER_LOCAL_SEARCH, max_iterations=int(max_iterations), local_dtype=dtype,
                  local_search_seed=int(seed or 0), matrix_cpu=matrix_cpu)
    spins = cp.stack([s.astype(cp.int8, copy=False) for s in spins_list], axis=0)   # candidate-major
    result = get_refiner(POLICIES[policy])(J_csr, spins, config)
    summary = dict(result.summary)
    iterations = next((int(summary[key]) for key in ("main_iterations", "main_rounds", "kernel_updates")
                       if summary.get(key) is not None), 0)
    return cp.asarray(result.spins, dtype=cp.int8).T, iterations, summary


LS_POLICIES = ("default",) + tuple(POLICIES)


def refine_candidates(policy, J_csr, spins_list, *, max_iterations, dtype, seed, J_host=None):
    """run_policy for SpecAdj and SpecSignedLap; return (n x k int8 spins, iterations).

    J_host (on the CPU) is only used by Independent-Set, to color the graph.
    """
    import numpy as np
    import scipy.sparse as sp
    matrix_cpu = (sp.csr_matrix(J_host).astype(np.float32)
                  if policy in NEEDS_HOST_MATRIX and J_host is not None else None)
    refined, iterations, _ = run_policy(policy, J_csr, spins_list, max_iterations=max_iterations,
                                        dtype=dtype, seed=seed, matrix_cpu=matrix_cpu)
    return refined, iterations
