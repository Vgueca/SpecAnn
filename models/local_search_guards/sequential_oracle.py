"""Sequential-Oracle policy: exact one-flip steepest descent, for small graphs.

Each step flips the single best spin of every candidate until no flip lowers the energy.
"""

from __future__ import annotations

import time
from typing import Any

import cupy as cp

from models.spectral_annealing_safe import _flip_candidate_nodes_kernel

from .common import (
    GuardInvariantError,
    RefinementResult,
    energy_tolerance,
    gpu_batch_energies,
    gpu_single_flip_deltas,
    peak_cupy_memory,
    timed_gpu_call,
)


def refine(matrix, spins, config) -> RefinementResult:
    cp.random.seed(int(config["local_search_seed"]))
    spins_current = cp.asarray(spins, dtype=cp.int8).copy()
    candidate_count, node_count = map(int, spins_current.shape)
    spins_proposal = cp.empty_like(spins_current)
    matrix64 = matrix.astype(cp.float64)
    current_energies = gpu_batch_energies(
        matrix64,
        spins_current,
        chunk_size=int(config["energy_chunk_size"]),
    )
    absolute_tolerance = float(config["absolute_tolerance"])
    relative_tolerance = float(config["relative_tolerance"])
    identity_absolute = float(config["identity_absolute_tolerance"])
    identity_relative = float(config["identity_relative_tolerance"])
    maximum_steps = int(config["oracle_max_steps_multiplier"]) * node_count
    flip_threads = 256

    round_rows: list[dict[str, Any]] = []
    total_flips = 0
    total_proposal_ms = 0.0
    total_energy_ms = 0.0
    maximum_identity_error = 0.0
    peak_memory = peak_cupy_memory()
    converged = False

    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    for step in range(1, maximum_steps + 1):
        def find_best_flips():
            deltas = gpu_single_flip_deltas(matrix64, spins_current)
            nodes = cp.argmin(deltas, axis=1).astype(cp.int32)
            candidates = cp.arange(candidate_count, dtype=cp.int32)
            best = deltas[candidates, nodes]
            return deltas, nodes, best

        (
            all_deltas,
            best_nodes,
            best_deltas,
        ), proposal_ms = timed_gpu_call(find_best_flips)
        total_proposal_ms += proposal_ms
        tolerances = energy_tolerance(
            current_energies,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            xp=cp,
        )
        improving = best_deltas < -tolerances
        improving_candidates = cp.flatnonzero(improving).astype(cp.int32)
        active_count = int(improving_candidates.size)
        if active_count == 0:
            converged = True
            break

        proposal_nodes = best_nodes[improving_candidates]
        cp.copyto(spins_proposal, spins_current)
        blocks = (active_count + flip_threads - 1) // flip_threads

        def execute_flips():
            _flip_candidate_nodes_kernel(
                (blocks,),
                (flip_threads,),
                (
                    active_count,
                    node_count,
                    improving_candidates,
                    proposal_nodes,
                    spins_current.reshape(-1),
                    spins_proposal.reshape(-1),
                ),
            )

        _, execute_ms = timed_gpu_call(execute_flips)
        total_proposal_ms += execute_ms
        proposal_energies, energy_ms = timed_gpu_call(
            lambda: gpu_batch_energies(
                matrix64,
                spins_proposal,
                chunk_size=int(config["energy_chunk_size"]),
            )
        )
        total_energy_ms += energy_ms
        actual_delta = proposal_energies - current_energies
        expected_delta = best_deltas
        identity_scale = cp.maximum(
            1.0,
            cp.maximum(cp.abs(actual_delta), cp.abs(expected_delta)),
        )
        identity_tolerance = (
            identity_absolute + identity_relative * identity_scale
        )
        identity_error = cp.abs(actual_delta - expected_delta)
        bad_identity = improving & (identity_error > identity_tolerance)
        bad_decrease = improving & (actual_delta >= -tolerances)
        if cp.any(bad_identity):
            candidate = int(cp.flatnonzero(bad_identity)[0].item())
            raise GuardInvariantError(
                "Sequential-Oracle single-flip energy identity failed at "
                f"step={step}, candidate={candidate}, "
                f"error={float(identity_error[candidate].item()):.6g}"
            )
        if cp.any(bad_decrease):
            candidate = int(cp.flatnonzero(bad_decrease)[0].item())
            raise GuardInvariantError(
                "Sequential-Oracle accepted a non-decreasing flip at "
                f"step={step}, candidate={candidate}, "
                f"delta={float(actual_delta[candidate].item()):.6g}"
            )

        cp.copyto(spins_current, spins_proposal)
        current_energies = proposal_energies
        total_flips += active_count
        maximum_identity_error = max(
            maximum_identity_error,
            float(cp.max(identity_error[improving]).item()),
        )
        observed = actual_delta[improving]
        round_rows.append(
            {
                "phase": "oracle",
                "round": step,
                "kernel_index": step,
                "active_candidates_before": candidate_count,
                "candidate_updates_with_flips": active_count,
                "selected_nodes": 1,
                "subset_fraction": 1.0 / node_count,
                "proposed_flips": active_count,
                "executed_flips": active_count,
                "accepted_updates": active_count,
                "rejected_updates": 0,
                "energy_increasing_candidate_updates": 0,
                "accepted_monotonicity_violations": 0,
                "minimum_energy_delta": float(cp.min(observed).item()),
                "mean_energy_delta": float(cp.mean(observed).item()),
                "maximum_energy_delta": float(cp.max(observed).item()),
                "maximum_single_flip_identity_error": float(
                    cp.max(identity_error[improving]).item()
                ),
                "proposal_ms": proposal_ms + execute_ms,
                "energy_evaluation_ms": energy_ms,
            }
        )
        observed_memory = peak_cupy_memory()
        peak_memory = {
            key: max(peak_memory.get(key, 0), value)
            for key, value in observed_memory.items()
        }

    cp.cuda.Stream.null.synchronize()
    local_search_seconds = time.perf_counter() - started
    final_deltas = gpu_single_flip_deltas(matrix64, spins_current)
    final_tolerances = energy_tolerance(
        current_energies,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        xp=cp,
    )
    final_improving = final_deltas < -final_tolerances[:, cp.newaxis]
    remaining = int(cp.sum(final_improving).item())
    if not converged or remaining:
        raise GuardInvariantError(
            "Sequential-Oracle failed to reach a 1-flip local optimum: "
            f"converged={converged}, remaining_improving_spins={remaining}, "
            f"maximum_steps={maximum_steps}"
        )

    summary = {
        "implementation": "sequential_oracle",
        "candidate_count": candidate_count,
        "node_count": node_count,
        "kernel_updates": len(round_rows),
        "candidate_updates": total_flips,
        "candidate_updates_with_flips": total_flips,
        "proposed_flips": total_flips,
        "executed_flips": total_flips,
        "accepted_updates": total_flips,
        "rejected_updates": 0,
        "energy_increasing_candidate_updates": 0,
        "accepted_monotonicity_violations": 0,
        "maximum_single_flip_identity_error": maximum_identity_error,
        "remaining_improving_spins": remaining,
        "local_optimal_candidate_count": candidate_count,
        "proposal_seconds": total_proposal_ms / 1000.0,
        "energy_evaluation_seconds": total_energy_ms / 1000.0,
        "local_search_seconds": local_search_seconds,
        **peak_memory,
    }
    return RefinementResult(
        spins=spins_current,
        local_search_seconds=local_search_seconds,
        summary=summary,
        round_rows=round_rows,
    )
