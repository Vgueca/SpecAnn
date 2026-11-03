"""Independent-Set policy: flip only nodes of one graph-color class at a time.

Nodes of one class share no edge, so the joint energy change is the sum of single-flip changes.
Each update is checked in FP64; any broken check raises GuardInvariantError.
"""

from __future__ import annotations

import time
from typing import Any

import cupy as cp
import numpy as np

from .common import (
    GuardInvariantError,
    RefinementResult,
    energy_tolerance,
    gpu_batch_energies,
    greedy_largest_first_coloring,
    peak_cupy_memory,
    timed_gpu_call,
    validate_coloring,
)


_conflict_free_proposal = cp.RawKernel(
    r'''
extern "C" __global__
void conflict_free_proposal(
    const int n_selected,
    const int n,
    const int k,
    const int* selected_nodes,
    const int* indptr,
    const int* indices,
    const double* data,
    const signed char* spins_in,
    signed char* spins_out,
    const double* candidate_tolerances,
    const unsigned char* active_candidates,
    double* individual_deltas,
    int* flip_counts)
{
    int lane = threadIdx.x;
    int selected_index = blockIdx.x * blockDim.y + threadIdx.y;
    int candidate = blockIdx.y;
    if (selected_index >= n_selected || candidate >= k) return;

    long long delta_index =
        (long long)candidate * (long long)n_selected + selected_index;
    individual_deltas[delta_index] = 0.0;
    if (active_candidates[candidate] == 0) return;

    int node = selected_nodes[selected_index];
    double local_sum = 0.0;
    for (int edge = indptr[node] + lane;
         edge < indptr[node + 1];
         edge += 32) {
        int neighbor = indices[edge];
        local_sum +=
            data[edge] * (double)spins_in[(long long)candidate * n + neighbor];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) {
        long long spin_index = (long long)candidate * n + node;
        signed char spin = spins_in[spin_index];
        double delta = 2.0 * (double)spin * local_sum;
        individual_deltas[delta_index] = delta;
        if (delta < -candidate_tolerances[candidate]) {
            spins_out[spin_index] = -spin;
            atomicAdd(&flip_counts[candidate], 1);
        }
    }
}
''',
    "conflict_free_proposal",
)


def refine(matrix, spins, config) -> RefinementResult:
    """Refine candidate-major spins by checked updates over the color classes, in random order."""
    matrix_cpu = config["matrix_cpu"]
    cp.random.seed(int(config["local_search_seed"]))
    rng = np.random.default_rng(int(config["local_search_seed"]))
    spins_current = cp.asarray(spins, dtype=cp.int8).copy()
    candidate_count, node_count = map(int, spins_current.shape)

    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    coloring = greedy_largest_first_coloring(matrix_cpu)
    coloring_validation = validate_coloring(matrix_cpu, coloring)
    color_classes_host = [
        np.flatnonzero(coloring == color).astype(np.int32)
        for color in range(coloring_validation["color_count"])
    ]
    if not color_classes_host:
        raise GuardInvariantError("Conflict-free coloring has no classes")

    matrix64 = matrix.astype(cp.float64)
    indptr = matrix64.indptr.astype(cp.int32)
    indices = matrix64.indices.astype(cp.int32)
    data = matrix64.data.astype(cp.float64)
    spins_proposal = cp.empty_like(spins_current)
    active = cp.ones(candidate_count, dtype=cp.uint8)
    flip_counts = cp.zeros(candidate_count, dtype=cp.int32)
    current_energies = gpu_batch_energies(
        matrix64,
        spins_current,
        chunk_size=int(config["energy_chunk_size"]),
    )

    main_limit = int(config["max_iterations"])
    final_limit = (
        int(config["final_sweep_iters"])
        * int(config["final_sweep_rounds"])
    )
    maximum_updates = main_limit + final_limit
    absolute_tolerance = float(config["absolute_tolerance"])
    relative_tolerance = float(config["relative_tolerance"])
    identity_absolute = float(config["identity_absolute_tolerance"])
    identity_relative = float(config["identity_relative_tolerance"])

    warp_size = 32
    warps_per_block = 4
    threads = (warp_size, warps_per_block)
    round_rows: list[dict[str, Any]] = []
    total_flips = 0
    total_candidate_updates = 0
    total_candidate_updates_with_flips = 0
    total_proposal_ms = 0.0
    total_energy_ms = 0.0
    maximum_identity_error = 0.0
    completed_sweeps = 0
    certified_candidates = cp.zeros(candidate_count, dtype=cp.bool_)
    sweep_flip_counts = cp.zeros(candidate_count, dtype=cp.int64)
    color_order = rng.permutation(len(color_classes_host))
    color_position = 0
    peak_memory = peak_cupy_memory()

    for kernel_index in range(1, maximum_updates + 1):
        if not cp.any(active):
            break

        color = int(color_order[color_position])
        selected_host = color_classes_host[color]
        selected_nodes = cp.asarray(selected_host, dtype=cp.int32)
        selected_count = int(selected_nodes.size)
        individual_deltas = cp.empty(
            (candidate_count, selected_count), dtype=cp.float64
        )
        flip_counts.fill(0)
        tolerances = energy_tolerance(
            current_energies,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            xp=cp,
        )
        cp.copyto(spins_proposal, spins_current)
        grid_x = (
            selected_count + warps_per_block - 1
        ) // warps_per_block

        def propose() -> None:
            _conflict_free_proposal(
                (grid_x, candidate_count),
                threads,
                (
                    selected_count,
                    node_count,
                    candidate_count,
                    selected_nodes,
                    indptr,
                    indices,
                    data,
                    spins_current.reshape(-1),
                    spins_proposal.reshape(-1),
                    tolerances,
                    active,
                    individual_deltas.reshape(-1),
                    flip_counts,
                ),
            )

        _, proposal_ms = timed_gpu_call(propose)
        total_proposal_ms += proposal_ms
        has_flips = (flip_counts > 0) & active.astype(cp.bool_)
        executed_mask = (
            individual_deltas < -tolerances[:, cp.newaxis]
        ) & active.astype(cp.bool_)[:, cp.newaxis]
        expected_delta = cp.sum(
            cp.where(executed_mask, individual_deltas, 0.0), axis=1
        )

        proposal_energies, energy_ms = timed_gpu_call(
            lambda: gpu_batch_energies(
                matrix64,
                spins_proposal,
                chunk_size=int(config["energy_chunk_size"]),
            )
        )
        total_energy_ms += energy_ms
        actual_delta = proposal_energies - current_energies

        identity_scale = cp.maximum(
            1.0,
            cp.maximum(cp.abs(actual_delta), cp.abs(expected_delta)),
        )
        identity_tolerance = (
            identity_absolute + identity_relative * identity_scale
        )
        identity_error = cp.abs(actual_delta - expected_delta)
        bad_identity = has_flips & (identity_error > identity_tolerance)
        bad_decrease = has_flips & (actual_delta >= -tolerances)
        counted_flips = cp.sum(executed_mask, axis=1).astype(cp.int32)
        bad_count = counted_flips != flip_counts

        if cp.any(bad_count):
            candidate = int(cp.flatnonzero(bad_count)[0].item())
            raise GuardInvariantError(
                "Conflict-Free flip-mask mismatch at "
                f"kernel={kernel_index}, candidate={candidate}"
            )
        if cp.any(bad_identity):
            candidate = int(cp.flatnonzero(bad_identity)[0].item())
            raise GuardInvariantError(
                "Conflict-Free joint delta does not match the sum of "
                "individual deltas at "
                f"kernel={kernel_index}, candidate={candidate}, "
                f"error={float(identity_error[candidate].item()):.6g}, "
                f"tolerance={float(identity_tolerance[candidate].item()):.6g}"
            )
        if cp.any(bad_decrease):
            candidate = int(cp.flatnonzero(bad_decrease)[0].item())
            raise GuardInvariantError(
                "Conflict-Free committed a non-decreasing update at "
                f"kernel={kernel_index}, candidate={candidate}, "
                f"delta={float(actual_delta[candidate].item()):.6g}"
            )

        cp.copyto(spins_current, spins_proposal)
        current_energies = proposal_energies
        sweep_flip_counts += flip_counts.astype(cp.int64)
        flips = int(cp.sum(flip_counts).item())
        candidates_with_flips = int(cp.sum(has_flips).item())
        total_flips += flips
        total_candidate_updates += int(cp.sum(active).item())
        total_candidate_updates_with_flips += candidates_with_flips
        maximum_identity_error = max(
            maximum_identity_error,
            float(cp.max(identity_error).item()),
        )

        phase = "main" if kernel_index <= main_limit else "final"
        observed_deltas = actual_delta[has_flips]
        round_rows.append(
            {
                "phase": phase,
                "round": kernel_index if phase == "main" else kernel_index - main_limit,
                "kernel_index": kernel_index,
                "sweep": completed_sweeps + 1,
                "color": color,
                "color_count": len(color_classes_host),
                "selected_nodes": selected_count,
                "subset_fraction": selected_count / node_count,
                "active_candidates_before": int(cp.sum(active).item()),
                "candidate_updates_with_flips": candidates_with_flips,
                "proposed_flips": flips,
                "executed_flips": flips,
                "accepted_updates": candidates_with_flips,
                "rejected_updates": 0,
                "energy_increasing_candidate_updates": 0,
                "accepted_monotonicity_violations": 0,
                "independent_set_internal_edges": 0,
                "minimum_individual_delta": (
                    float(cp.min(individual_deltas[executed_mask]).item())
                    if flips
                    else 0.0
                ),
                "minimum_energy_delta": (
                    float(cp.min(observed_deltas).item())
                    if observed_deltas.size
                    else 0.0
                ),
                "mean_energy_delta": (
                    float(cp.mean(observed_deltas).item())
                    if observed_deltas.size
                    else 0.0
                ),
                "maximum_energy_delta": (
                    float(cp.max(observed_deltas).item())
                    if observed_deltas.size
                    else 0.0
                ),
                "maximum_joint_delta_identity_error": float(
                    cp.max(identity_error).item()
                ),
                "proposal_ms": proposal_ms,
                "energy_evaluation_ms": energy_ms,
            }
        )

        observed_memory = peak_cupy_memory()
        peak_memory = {
            key: max(peak_memory.get(key, 0), value)
            for key, value in observed_memory.items()
        }

        color_position += 1
        if color_position == len(color_classes_host):
            completed_sweeps += 1
            no_flips_in_sweep = (sweep_flip_counts == 0) & active.astype(
                cp.bool_
            )
            certified_candidates |= no_flips_in_sweep
            active = (
                active.astype(cp.bool_) & ~no_flips_in_sweep
            ).astype(cp.uint8)
            sweep_flip_counts.fill(0)
            color_order = rng.permutation(len(color_classes_host))
            color_position = 0

    cp.cuda.Stream.null.synchronize()
    local_search_seconds = time.perf_counter() - started
    summary = {
        "implementation": "conflict_free_coloring",
        "candidate_count": candidate_count,
        "node_count": node_count,
        "color_count": int(coloring_validation["color_count"]),
        "minimum_color_size": int(
            coloring_validation["minimum_color_size"]
        ),
        "maximum_color_size": int(
            coloring_validation["maximum_color_size"]
        ),
        "independent_set_internal_edges": 0,
        "completed_color_sweeps": completed_sweeps,
        "kernel_updates": len(round_rows),
        "candidate_updates": total_candidate_updates,
        "candidate_updates_with_flips": (
            total_candidate_updates_with_flips
        ),
        "proposed_flips": total_flips,
        "executed_flips": total_flips,
        "accepted_updates": int(
            sum(row["accepted_updates"] for row in round_rows)
        ),
        "rejected_updates": 0,
        "energy_increasing_candidate_updates": 0,
        "accepted_monotonicity_violations": 0,
        "maximum_joint_delta_identity_error": maximum_identity_error,
        "proposal_seconds": total_proposal_ms / 1000.0,
        "energy_evaluation_seconds": total_energy_ms / 1000.0,
        "color_sweep_certified_candidates": int(
            cp.sum(certified_candidates).item()
        ),
        "local_search_seconds": local_search_seconds,
        **peak_memory,
    }
    return RefinementResult(
        spins=spins_current,
        local_search_seconds=local_search_seconds,
        summary=summary,
        round_rows=round_rows,
    )
