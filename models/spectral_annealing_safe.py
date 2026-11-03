"""Guarded local search: each batch of spin flips is proposed on a copy of the spins and kept, per
candidate, only if the exact Ising energy strictly goes down. Used by the Fixed and Adaptive Guard policies."""

import time

import cupy as cp
import numpy as np

from models.safe_update_policy import (
    adapt_batch_fraction,
    adapt_candidate_fractions,
    strict_energy_acceptance,
)


SAFE_UPDATE_VERSION = "two_buffer_exact_energy_v3"


_propose_spins_kernel_float32 = cp.RawKernel(
    r'''
extern "C" __global__
void propose_spins_kernel_float32(
    const int n_selected,
    const int n,
    const int k,
    const int* selected_nodes,
    const int* candidate_selected_counts,
    const int* indptr,
    const int* indices,
    const float* data,
    const signed char* spins_in,
    signed char* spins_out,
    const unsigned char* active_mask,
    int* proposal_counts)
{
    int lane = threadIdx.x;
    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int candidate = blockIdx.y;

    if (warp_id >= n_selected || candidate >= k) return;
    if (active_mask[candidate] == 0) return;
    if (warp_id >= candidate_selected_counts[candidate]) return;

    int node = selected_nodes[warp_id];
    int row_start = indptr[node];
    int row_end = indptr[node + 1];
    float local_sum = 0.0f;

    for (int edge = row_start + lane; edge < row_end; edge += 32) {
        int neighbor = indices[edge];
        local_sum += data[edge] * (float)spins_in[candidate * n + neighbor];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        int spin_index = candidate * n + node;
        signed char spin = spins_in[spin_index];
        float delta_energy = 2.0f * (float)spin * local_sum;
        if (delta_energy < 0.0f) {
            spins_out[spin_index] = -spin;
            atomicAdd(&proposal_counts[candidate], 1);
        }
    }
}
''',
    "propose_spins_kernel_float32",
)


_propose_spins_kernel_float64 = cp.RawKernel(
    r'''
extern "C" __global__
void propose_spins_kernel_float64(
    const int n_selected,
    const int n,
    const int k,
    const int* selected_nodes,
    const int* candidate_selected_counts,
    const int* indptr,
    const int* indices,
    const double* data,
    const signed char* spins_in,
    signed char* spins_out,
    const unsigned char* active_mask,
    int* proposal_counts)
{
    int lane = threadIdx.x;
    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int candidate = blockIdx.y;

    if (warp_id >= n_selected || candidate >= k) return;
    if (active_mask[candidate] == 0) return;
    if (warp_id >= candidate_selected_counts[candidate]) return;

    int node = selected_nodes[warp_id];
    int row_start = indptr[node];
    int row_end = indptr[node + 1];
    double local_sum = 0.0;

    for (int edge = row_start + lane; edge < row_end; edge += 32) {
        int neighbor = indices[edge];
        local_sum += data[edge] * (double)spins_in[candidate * n + neighbor];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        int spin_index = candidate * n + node;
        signed char spin = spins_in[spin_index];
        double delta_energy = 2.0 * (double)spin * local_sum;
        if (delta_energy < 0.0) {
            spins_out[spin_index] = -spin;
            atomicAdd(&proposal_counts[candidate], 1);
        }
    }
}
''',
    "propose_spins_kernel_float64",
)


_commit_proposals_kernel = cp.RawKernel(
    r'''
extern "C" __global__
void commit_proposals_kernel(
    const long long total_spins,
    const int n,
    const signed char* proposals,
    const unsigned char* accepted_candidates,
    signed char* current_spins)
{
    long long index =
        (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (index >= total_spins) return;

    int candidate = (int)(index / (long long)n);
    if (accepted_candidates[candidate] != 0) {
        current_spins[index] = proposals[index];
    }
}
''',
    "commit_proposals_kernel",
)


_flip_candidate_nodes_kernel = cp.RawKernel(
    r'''
extern "C" __global__
void flip_candidate_nodes_kernel(
    const int update_count,
    const int n,
    const int* candidate_indices,
    const int* node_indices,
    const signed char* spins_in,
    signed char* spins_out)
{
    int update_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (update_index >= update_count) return;

    int candidate = candidate_indices[update_index];
    int node = node_indices[update_index];
    long long spin_index =
        (long long)candidate * (long long)n + (long long)node;
    spins_out[spin_index] = -spins_in[spin_index];
}
''',
    "flip_candidate_nodes_kernel",
)


_select_independent_nodes_kernel = cp.RawKernel(
    r'''
extern "C" __global__
void select_independent_nodes_kernel(
    const int n,
    const int* indptr,
    const int* indices,
    const float* priorities,
    unsigned char* selected_mask)
{
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n) return;

    float node_priority = priorities[node];
    unsigned char selected = 1;
    for (int edge = indptr[node]; edge < indptr[node + 1]; ++edge) {
        int neighbor = indices[edge];
        if (neighbor == node) continue;
        float neighbor_priority = priorities[neighbor];
        if (
            neighbor_priority > node_priority
            || (
                neighbor_priority == node_priority
                && neighbor > node
            )
        ) {
            selected = 0;
            break;
        }
    }
    selected_mask[node] = selected;
}
''',
    "select_independent_nodes_kernel",
)


def _time_gpu_operation(operation):
    """Run an operation and return (result, GPU time in milliseconds)."""
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    result = operation()
    end_event.record()
    end_event.synchronize()
    return result, float(cp.cuda.get_elapsed_time(start_event, end_event))


def _compute_batch_energies(
    J_energy,
    spins_flat,
    n,
    k,
    *,
    energy_dtype,
    chunk_size,
    candidate_indices=None,
    base_energies=None,
):
    """Compute exact candidate energies in chunks, to bound GPU memory."""
    spins_matrix = spins_flat.reshape(k, n)
    if base_energies is None:
        energies = cp.empty(k, dtype=energy_dtype)
    else:
        energies = base_energies.copy()

    if candidate_indices is None:
        candidate_indices = cp.arange(k, dtype=cp.int32)

    candidate_count = int(candidate_indices.size)
    for start in range(0, candidate_count, chunk_size):
        indices_block = candidate_indices[start:start + chunk_size]
        candidate_block = spins_matrix[indices_block].T.astype(
            energy_dtype, copy=False
        )
        matrix_product = J_energy.dot(candidate_block)
        energies[indices_block] = -0.5 * cp.sum(
            candidate_block * matrix_product, axis=0
        )

    return energies


def _empty_phase_totals(phase):
    return {
        "phase": phase,
        "rounds": 0,
        "candidate_updates": 0,
        "candidates_with_flips": 0,
        "accepted_updates": 0,
        "rejected_updates": 0,
        "no_flip_updates": 0,
        "proposed_flips": 0,
        "accepted_flips": 0,
        "rejected_flips": 0,
        "proposed_energy_increases": 0,
        "monotonicity_violations": 0,
        "largest_proposed_energy_increase": 0.0,
        "largest_rejected_flip_count": 0,
        "proposal_ms": 0.0,
        "energy_evaluation_ms": 0.0,
        "commit_ms": 0.0,
    }


def _run_guarded_phase(
    *,
    phase,
    J_csr,
    J_energy,
    spins_current,
    spins_proposal,
    current_energies,
    n,
    k,
    max_iterations,
    subset_fraction,
    patience,
    local_dtype,
    energy_dtype,
    energy_chunk_size,
    absolute_tolerance,
    relative_tolerance,
    collect_round_metrics,
    adaptive_batches,
    minimum_subset_fraction,
    batch_shrink_factor,
    batch_growth_factor,
    rejection_threshold,
    adaptation_scope,
    proposal_mode,
    proposal_trials,
    proposal_semantics,
):
    """Run one guarded local-search phase (main or final)."""
    if max_iterations <= 0:
        return current_energies, 0, _empty_phase_totals(phase), []

    if local_dtype == cp.float32:
        proposal_kernel = _propose_spins_kernel_float32
    elif local_dtype == cp.float64:
        proposal_kernel = _propose_spins_kernel_float64
    else:
        raise ValueError(f"Unsupported local-search dtype: {local_dtype}")

    indptr = J_csr.indptr.astype(cp.int32)
    indices = J_csr.indices.astype(cp.int32)
    data = J_csr.data.astype(local_dtype)

    active = cp.ones(k, dtype=cp.uint8)
    no_accept_streak = cp.zeros(k, dtype=cp.int32)
    proposal_counts = cp.zeros(k, dtype=cp.int32)
    totals = _empty_phase_totals(phase)
    round_rows = []

    warp_size = 32
    warps_per_block = 4
    threads_per_block = (warp_size, warps_per_block)
    total_spins = int(n) * int(k)
    commit_threads = 256
    commit_blocks = (total_spins + commit_threads - 1) // commit_threads
    current_subset_fraction = float(subset_fraction)
    per_candidate_adaptation = (
        adaptive_batches and adaptation_scope == "per_candidate"
    )
    candidate_subset_fractions = (
        cp.full(k, subset_fraction, dtype=cp.float64)
        if per_candidate_adaptation
        else None
    )
    independent_mask = (
        cp.empty(n, dtype=cp.uint8)
        if proposal_mode == "random_independent_set"
        else None
    )
    independent_threads = 256
    independent_blocks = (
        (n + independent_threads - 1) // independent_threads
    )
    best_proposal = (
        cp.empty_like(spins_proposal)
        if proposal_trials > 1
        else None
    )
    best_proposal_counts = (
        cp.zeros_like(proposal_counts)
        if proposal_trials > 1
        else None
    )

    for round_index in range(max_iterations):
        active_before = int(cp.sum(active).item())
        if active_before == 0:
            break

        active_before_bool = active.astype(cp.bool_)
        selection_ms = 0.0
        if proposal_mode == "random_independent_set":
            def select_independent_nodes():
                priorities = cp.random.random(
                    n, dtype=cp.float32
                )
                _select_independent_nodes_kernel(
                    (independent_blocks,),
                    (independent_threads,),
                    (
                        n,
                        indptr,
                        indices,
                        priorities,
                        independent_mask,
                    ),
                )
                return cp.flatnonzero(independent_mask).astype(cp.int32)

            selected_nodes, selection_ms = _time_gpu_operation(
                select_independent_nodes
            )
            num_selected = int(selected_nodes.size)
            subset_fraction_used = num_selected / n
            minimum_fraction_used = subset_fraction_used
            mean_fraction_used = subset_fraction_used
            mean_selected_nodes = float(num_selected)
            candidate_selected_counts = cp.where(
                active_before_bool,
                num_selected,
                0,
            ).astype(cp.int32)
        elif per_candidate_adaptation:
            candidate_selected_counts = cp.where(
                active_before_bool,
                cp.maximum(
                    1,
                    (candidate_subset_fractions * n).astype(cp.int32),
                ),
                0,
            ).astype(cp.int32)
            num_selected = int(
                cp.max(candidate_selected_counts).item()
            )
        else:
            subset_fraction_used = current_subset_fraction
            minimum_fraction_used = subset_fraction_used
            mean_fraction_used = subset_fraction_used
            num_selected = max(1, int(subset_fraction_used * n))
            candidate_selected_counts = cp.where(
                active_before_bool,
                num_selected,
                0,
            ).astype(cp.int32)
            mean_selected_nodes = float(num_selected)

        num_warps_x = (
            num_selected + warps_per_block - 1
        ) // warps_per_block
        grid = (num_warps_x, k)
        active_indices = cp.flatnonzero(active)
        total_proposal_ms = selection_ms
        total_energy_evaluation_ms = 0.0
        if proposal_trials > 1:
            cp.copyto(best_proposal, spins_current)
            best_proposal_energies = current_energies.copy()
            best_proposal_counts.fill(0)
            proposal_found = cp.zeros(k, dtype=cp.bool_)

        for trial_index in range(proposal_trials):
            if proposal_mode == "batch":
                selected_nodes = cp.random.permutation(n).astype(
                    cp.int32
                )[:num_selected]
            proposal_counts.fill(0)

            def propose():
                cp.copyto(spins_proposal, spins_current)
                proposal_input = (
                    spins_proposal
                    if proposal_semantics == "inplace"
                    else spins_current
                )
                proposal_kernel(
                    grid,
                    threads_per_block,
                    (
                        num_selected,
                        n,
                        k,
                        selected_nodes,
                        candidate_selected_counts,
                        indptr,
                        indices,
                        data,
                        proposal_input,
                        spins_proposal,
                        active,
                        proposal_counts,
                    ),
                )

            _, trial_proposal_ms = _time_gpu_operation(propose)
            total_proposal_ms += trial_proposal_ms

            def evaluate_proposals():
                return _compute_batch_energies(
                    J_energy,
                    spins_proposal,
                    n,
                    k,
                    energy_dtype=energy_dtype,
                    chunk_size=energy_chunk_size,
                    candidate_indices=active_indices,
                    base_energies=current_energies,
                )

            trial_energies, trial_energy_ms = _time_gpu_operation(
                evaluate_proposals
            )
            total_energy_evaluation_ms += trial_energy_ms

            if proposal_trials > 1:
                trial_has_flips = (
                    active_before_bool & (proposal_counts > 0)
                )
                trial_is_best = trial_has_flips & (
                    ~proposal_found
                    | (trial_energies < best_proposal_energies)
                )
                trial_is_best_u8 = trial_is_best.astype(cp.uint8)
                _commit_proposals_kernel(
                    (commit_blocks,),
                    (commit_threads,),
                    (
                        total_spins,
                        n,
                        spins_proposal,
                        trial_is_best_u8,
                        best_proposal,
                    ),
                )
                best_proposal_energies = cp.where(
                    trial_is_best,
                    trial_energies,
                    best_proposal_energies,
                )
                best_proposal_counts = cp.where(
                    trial_is_best,
                    proposal_counts,
                    best_proposal_counts,
                )
                proposal_found |= trial_has_flips
            else:
                proposal_energies = trial_energies

        if proposal_trials > 1:
            cp.copyto(proposal_counts, best_proposal_counts)
            proposal_energies = cp.where(
                proposal_found,
                best_proposal_energies,
                current_energies,
            )
            proposal_source = best_proposal
        else:
            proposal_source = spins_proposal
        proposal_ms = total_proposal_ms
        energy_evaluation_ms = total_energy_evaluation_ms

        active_bool = active.astype(cp.bool_)
        has_flips = active_bool & (proposal_counts > 0)
        accepted, energy_delta, tolerance = strict_energy_acceptance(
            current_energies,
            proposal_energies,
            has_flips,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            xp=cp,
        )
        rejected = has_flips & ~accepted
        no_flips = active_bool & ~has_flips

        accepted_u8 = accepted.astype(cp.uint8)

        def commit():
            _commit_proposals_kernel(
                (commit_blocks,),
                (commit_threads,),
                (
                    total_spins,
                    n,
                    proposal_source,
                    accepted_u8,
                    spins_current,
                ),
            )

        _, commit_ms = _time_gpu_operation(commit)
        current_energies = cp.where(
            accepted, proposal_energies, current_energies
        )

        proposal_counts_host = cp.asnumpy(proposal_counts)
        accepted_host = cp.asnumpy(accepted)
        rejected_host = cp.asnumpy(rejected)
        no_flips_host = cp.asnumpy(no_flips)
        delta_host = cp.asnumpy(energy_delta)
        tolerance_host = cp.asnumpy(tolerance)
        has_flips_host = cp.asnumpy(has_flips)

        if proposal_mode == "random_independent_set":
            stalled = active_bool & ~accepted
            next_minimum_fraction = subset_fraction_used
            next_mean_fraction = subset_fraction_used
            current_subset_fraction = subset_fraction_used
        elif per_candidate_adaptation:
            fractions_before_host = cp.asnumpy(
                candidate_subset_fractions
            )
            active_before_host = cp.asnumpy(active_before_bool)
            active_fractions_host = fractions_before_host[
                active_before_host
            ]
            minimum_fraction_used = float(
                active_fractions_host.min()
            )
            mean_fraction_used = float(
                active_fractions_host.mean()
            )
            subset_fraction_used = float(
                active_fractions_host.max()
            )
            selected_counts_host = np.maximum(
                1, (active_fractions_host * n).astype(np.int32)
            )
            mean_selected_nodes = float(selected_counts_host.mean())

        accepted_updates = int(accepted_host.sum())
        rejected_updates = int(rejected_host.sum())
        no_flip_updates = int(no_flips_host.sum())
        proposed_flips = int(proposal_counts_host[has_flips_host].sum())
        accepted_flips = int(proposal_counts_host[accepted_host].sum())
        rejected_flips = int(proposal_counts_host[rejected_host].sum())

        proposed_deltas = delta_host[has_flips_host]
        proposed_tolerances = tolerance_host[has_flips_host]
        if proposed_deltas.size:
            proposed_increases = int(
                (proposed_deltas > proposed_tolerances).sum()
            )
            largest_increase = float(max(0.0, proposed_deltas.max()))
            mean_delta = float(proposed_deltas.mean())
            min_delta = float(proposed_deltas.min())
            max_delta = float(proposed_deltas.max())
        else:
            proposed_increases = 0
            largest_increase = 0.0
            mean_delta = 0.0
            min_delta = 0.0
            max_delta = 0.0

        if accepted_updates:
            accepted_deltas = delta_host[accepted_host]
            accepted_tolerances = tolerance_host[accepted_host]
            monotonicity_violations = int(
                (accepted_deltas >= -accepted_tolerances).sum()
            )
            max_accepted_delta = float(accepted_deltas.max())
        else:
            monotonicity_violations = 0
            max_accepted_delta = 0.0

        largest_rejected_flip_count = (
            int(proposal_counts_host[rejected_host].max())
            if rejected_updates
            else 0
        )

        if per_candidate_adaptation:
            next_fractions_host, _ = adapt_candidate_fractions(
                fractions_before_host,
                accepted_host,
                rejected_host,
                initial_fraction=subset_fraction,
                minimum_fraction=minimum_subset_fraction,
                shrink_factor=batch_shrink_factor,
                growth_factor=batch_growth_factor,
                xp=np,
            )
            (
                candidate_subset_fractions,
                candidate_batch_was_reduced,
            ) = adapt_candidate_fractions(
                candidate_subset_fractions,
                accepted,
                rejected,
                initial_fraction=subset_fraction,
                minimum_fraction=minimum_subset_fraction,
                shrink_factor=batch_shrink_factor,
                growth_factor=batch_growth_factor,
                xp=cp,
            )
            stalled = no_flips | (
                rejected & ~candidate_batch_was_reduced
            )
            next_active_fractions_host = next_fractions_host[
                active_before_host
            ]
            next_minimum_fraction = float(
                next_active_fractions_host.min()
            )
            next_mean_fraction = float(
                next_active_fractions_host.mean()
            )
            current_subset_fraction = float(
                next_active_fractions_host.max()
            )
        elif adaptive_batches:
            at_minimum_fraction = (
                subset_fraction_used
                <= minimum_subset_fraction + 1e-15
            )
            (
                current_subset_fraction,
                batch_was_reduced,
            ) = adapt_batch_fraction(
                subset_fraction_used,
                initial_fraction=subset_fraction,
                minimum_fraction=minimum_subset_fraction,
                accepted_updates=accepted_updates,
                rejected_updates=rejected_updates,
                rejection_threshold=rejection_threshold,
                shrink_factor=batch_shrink_factor,
                growth_factor=batch_growth_factor,
            )
            stalled = no_flips | (
                rejected
                & (at_minimum_fraction or not batch_was_reduced)
            )
            next_minimum_fraction = current_subset_fraction
            next_mean_fraction = current_subset_fraction
        else:
            stalled = active_bool & ~accepted
            next_minimum_fraction = current_subset_fraction
            next_mean_fraction = current_subset_fraction

        no_accept_streak = cp.where(
            active_bool,
            cp.where(
                accepted,
                cp.zeros_like(no_accept_streak),
                cp.where(
                    stalled,
                    no_accept_streak + 1,
                    cp.zeros_like(no_accept_streak),
                ),
            ),
            no_accept_streak,
        )
        active = (
            active_bool & (no_accept_streak < patience)
        ).astype(cp.uint8)

        totals["rounds"] += 1
        totals["candidate_updates"] += active_before
        totals["candidates_with_flips"] += (
            accepted_updates + rejected_updates
        )
        totals["accepted_updates"] += accepted_updates
        totals["rejected_updates"] += rejected_updates
        totals["no_flip_updates"] += no_flip_updates
        totals["proposed_flips"] += proposed_flips
        totals["accepted_flips"] += accepted_flips
        totals["rejected_flips"] += rejected_flips
        totals["proposed_energy_increases"] += proposed_increases
        totals["monotonicity_violations"] += monotonicity_violations
        totals["largest_proposed_energy_increase"] = max(
            totals["largest_proposed_energy_increase"],
            largest_increase,
        )
        totals["largest_rejected_flip_count"] = max(
            totals["largest_rejected_flip_count"],
            largest_rejected_flip_count,
        )
        totals["proposal_ms"] += proposal_ms
        totals["energy_evaluation_ms"] += energy_evaluation_ms
        totals["commit_ms"] += commit_ms

        if collect_round_metrics:
            round_rows.append(
                {
                    "phase": phase,
                    "round": round_index + 1,
                    "active_candidates_before": active_before,
                    "subset_fraction_used": subset_fraction_used,
                    "minimum_subset_fraction_used": minimum_fraction_used,
                    "mean_subset_fraction_used": mean_fraction_used,
                    "next_subset_fraction": current_subset_fraction,
                    "next_minimum_subset_fraction": (
                        next_minimum_fraction
                    ),
                    "next_mean_subset_fraction": next_mean_fraction,
                    "selected_nodes": num_selected,
                    "mean_selected_nodes": mean_selected_nodes,
                    "candidates_with_flips": (
                        accepted_updates + rejected_updates
                    ),
                    "accepted_updates": accepted_updates,
                    "rejected_updates": rejected_updates,
                    "no_flip_updates": no_flip_updates,
                    "proposed_flips": proposed_flips,
                    "accepted_flips": accepted_flips,
                    "rejected_flips": rejected_flips,
                    "proposed_energy_increases": proposed_increases,
                    "monotonicity_violations": monotonicity_violations,
                    "minimum_proposed_energy_delta": min_delta,
                    "mean_proposed_energy_delta": mean_delta,
                    "maximum_proposed_energy_delta": max_delta,
                    "maximum_accepted_energy_delta": max_accepted_delta,
                    "proposal_ms": proposal_ms,
                    "energy_evaluation_ms": energy_evaluation_ms,
                    "commit_ms": commit_ms,
                }
            )

    return current_energies, totals["rounds"], totals, round_rows


def _run_single_flip_polish(
    *,
    J_csr,
    J_energy,
    spins_current,
    spins_proposal,
    current_energies,
    n,
    k,
    max_iterations,
    candidate_limit,
    local_dtype,
    energy_dtype,
    energy_chunk_size,
    absolute_tolerance,
    relative_tolerance,
    collect_round_metrics,
):
    """Polish the lowest-energy candidates with exact one-flip steepest descent."""
    phase = "polish"
    totals = _empty_phase_totals(phase)
    round_rows = []
    if max_iterations <= 0 or candidate_limit <= 0:
        return current_energies, 0, totals, round_rows

    selected_count = min(int(candidate_limit), int(k))
    active_candidates = cp.argsort(current_energies)[
        :selected_count
    ].astype(cp.int32)
    diagonal = J_csr.diagonal().astype(local_dtype)

    total_spins = int(n) * int(k)
    commit_threads = 256
    commit_blocks = (total_spins + commit_threads - 1) // commit_threads
    flip_threads = 256

    for round_index in range(max_iterations):
        active_count = int(active_candidates.size)
        if active_count == 0:
            break

        def propose_best_single_flips():
            cp.copyto(spins_proposal, spins_current)
            spins_matrix = spins_current.reshape(k, n)
            candidate_block = spins_matrix[active_candidates].T.astype(
                local_dtype, copy=False
            )
            local_fields = J_csr.dot(candidate_block)
            off_diagonal_fields = (
                local_fields - diagonal[:, cp.newaxis] * candidate_block
            )
            deltas = (
                2.0 * candidate_block * off_diagonal_fields
            )
            best_nodes = cp.argmin(deltas, axis=0).astype(cp.int32)
            columns = cp.arange(active_count, dtype=cp.int32)
            best_deltas = deltas[best_nodes, columns]
            improving = best_deltas < 0.0
            proposal_candidates = active_candidates[improving]
            proposal_nodes = best_nodes[improving]
            proposal_count = int(proposal_candidates.size)
            if proposal_count:
                flip_blocks = (
                    proposal_count + flip_threads - 1
                ) // flip_threads
                _flip_candidate_nodes_kernel(
                    (flip_blocks,),
                    (flip_threads,),
                    (
                        proposal_count,
                        n,
                        proposal_candidates,
                        proposal_nodes,
                        spins_current,
                        spins_proposal,
                    ),
                )
            return proposal_candidates, best_deltas

        (
            proposal_candidates,
            best_local_deltas,
        ), proposal_ms = _time_gpu_operation(propose_best_single_flips)
        proposal_count = int(proposal_candidates.size)

        def evaluate_proposals():
            return _compute_batch_energies(
                J_energy,
                spins_proposal,
                n,
                k,
                energy_dtype=energy_dtype,
                chunk_size=energy_chunk_size,
                candidate_indices=proposal_candidates,
                base_energies=current_energies,
            )

        proposal_energies, energy_evaluation_ms = _time_gpu_operation(
            evaluate_proposals
        )
        has_flips = cp.zeros(k, dtype=cp.bool_)
        has_flips[proposal_candidates] = True
        accepted, energy_delta, tolerance = strict_energy_acceptance(
            current_energies,
            proposal_energies,
            has_flips,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            xp=cp,
        )
        rejected = has_flips & ~accepted
        accepted_u8 = accepted.astype(cp.uint8)

        def commit():
            _commit_proposals_kernel(
                (commit_blocks,),
                (commit_threads,),
                (
                    total_spins,
                    n,
                    spins_proposal,
                    accepted_u8,
                    spins_current,
                ),
            )

        _, commit_ms = _time_gpu_operation(commit)
        current_energies = cp.where(
            accepted, proposal_energies, current_energies
        )

        accepted_host = cp.asnumpy(accepted)
        rejected_host = cp.asnumpy(rejected)
        has_flips_host = cp.asnumpy(has_flips)
        delta_host = cp.asnumpy(energy_delta)
        tolerance_host = cp.asnumpy(tolerance)
        best_local_deltas_host = cp.asnumpy(best_local_deltas)

        accepted_updates = int(accepted_host.sum())
        rejected_updates = int(rejected_host.sum())
        no_flip_updates = active_count - proposal_count
        proposed_deltas = delta_host[has_flips_host]
        proposed_tolerances = tolerance_host[has_flips_host]
        if proposed_deltas.size:
            proposed_increases = int(
                (proposed_deltas > proposed_tolerances).sum()
            )
            largest_increase = float(
                max(0.0, proposed_deltas.max())
            )
            mean_delta = float(proposed_deltas.mean())
            min_delta = float(proposed_deltas.min())
            max_delta = float(proposed_deltas.max())
        else:
            proposed_increases = 0
            largest_increase = 0.0
            mean_delta = 0.0
            min_delta = 0.0
            max_delta = 0.0

        if accepted_updates:
            accepted_deltas = delta_host[accepted_host]
            accepted_tolerances = tolerance_host[accepted_host]
            monotonicity_violations = int(
                (accepted_deltas >= -accepted_tolerances).sum()
            )
            max_accepted_delta = float(accepted_deltas.max())
        else:
            monotonicity_violations = 0
            max_accepted_delta = 0.0

        totals["rounds"] += 1
        totals["candidate_updates"] += active_count
        totals["candidates_with_flips"] += proposal_count
        totals["accepted_updates"] += accepted_updates
        totals["rejected_updates"] += rejected_updates
        totals["no_flip_updates"] += no_flip_updates
        totals["proposed_flips"] += proposal_count
        totals["accepted_flips"] += accepted_updates
        totals["rejected_flips"] += rejected_updates
        totals["proposed_energy_increases"] += proposed_increases
        totals["monotonicity_violations"] += monotonicity_violations
        totals["largest_proposed_energy_increase"] = max(
            totals["largest_proposed_energy_increase"],
            largest_increase,
        )
        totals["largest_rejected_flip_count"] = max(
            totals["largest_rejected_flip_count"],
            1 if rejected_updates else 0,
        )
        totals["proposal_ms"] += proposal_ms
        totals["energy_evaluation_ms"] += energy_evaluation_ms
        totals["commit_ms"] += commit_ms

        if collect_round_metrics:
            round_rows.append(
                {
                    "phase": phase,
                    "round": round_index + 1,
                    "active_candidates_before": active_count,
                    "subset_fraction_used": 1.0 / n,
                    "minimum_subset_fraction_used": 1.0 / n,
                    "mean_subset_fraction_used": 1.0 / n,
                    "next_subset_fraction": 1.0 / n,
                    "next_minimum_subset_fraction": 1.0 / n,
                    "next_mean_subset_fraction": 1.0 / n,
                    "selected_nodes": 1,
                    "mean_selected_nodes": 1.0,
                    "candidates_with_flips": proposal_count,
                    "accepted_updates": accepted_updates,
                    "rejected_updates": rejected_updates,
                    "no_flip_updates": no_flip_updates,
                    "proposed_flips": proposal_count,
                    "accepted_flips": accepted_updates,
                    "rejected_flips": rejected_updates,
                    "proposed_energy_increases": proposed_increases,
                    "monotonicity_violations": monotonicity_violations,
                    "minimum_proposed_energy_delta": min_delta,
                    "mean_proposed_energy_delta": mean_delta,
                    "maximum_proposed_energy_delta": max_delta,
                    "maximum_accepted_energy_delta": max_accepted_delta,
                    "minimum_local_single_flip_delta": float(
                        best_local_deltas_host.min()
                    ),
                    "proposal_ms": proposal_ms,
                    "energy_evaluation_ms": energy_evaluation_ms,
                    "commit_ms": commit_ms,
                }
            )

        active_candidates = cp.flatnonzero(accepted).astype(cp.int32)

    return current_energies, totals["rounds"], totals, round_rows


def safe_flip_refinement(
    J_csr,
    spins_list,
    *,
    max_iterations=1000,
    final_sweep_fraction=1.0,
    final_sweep_iters=25,
    num_final_rounds=25,
    subset_fraction=0.9,
    dtype=cp.float32,
    energy_precision=64,
    energy_chunk_size=16,
    absolute_tolerance=1e-10,
    relative_tolerance=1e-12,
    patience=3,
    collect_round_metrics=True,
    adaptive_batches=True,
    minimum_subset_fraction=0.01,
    batch_shrink_factor=0.5,
    batch_growth_factor=1.25,
    rejection_threshold=0.25,
    adaptation_scope="global",
    proposal_mode="batch",
    proposal_trials=1,
    proposal_semantics="immutable",
    single_flip_polish_iterations=0,
    polish_candidate_count=16,
):
    """Guarded batch local search; return (n x k spins, main iterations, summary, round rows).

    Flips are proposed on a copy of the spins and kept only if the exact energy strictly goes down.
    """
    if not spins_list:
        raise ValueError("safe_flip_refinement requires at least one candidate")
    if energy_precision not in (32, 64):
        raise ValueError("energy_precision must be 32 or 64")
    if energy_chunk_size < 1:
        raise ValueError("energy_chunk_size must be positive")
    if patience < 1:
        raise ValueError("patience must be positive")
    if not 0.0 < subset_fraction <= 1.0:
        raise ValueError("subset_fraction must be in (0, 1]")
    if not 0.0 < final_sweep_fraction <= 1.0:
        raise ValueError("final_sweep_fraction must be in (0, 1]")
    if not 0.0 < minimum_subset_fraction <= subset_fraction:
        raise ValueError(
            "minimum_subset_fraction must be in (0, subset_fraction]"
        )
    if not 0.0 < batch_shrink_factor < 1.0:
        raise ValueError("batch_shrink_factor must be in (0, 1)")
    if batch_growth_factor <= 1.0:
        raise ValueError("batch_growth_factor must be greater than 1")
    if not 0.0 <= rejection_threshold <= 1.0:
        raise ValueError("rejection_threshold must be in [0, 1]")
    if adaptation_scope not in {"global", "per_candidate"}:
        raise ValueError(
            "adaptation_scope must be 'global' or 'per_candidate'"
        )
    if proposal_mode not in {"batch", "random_independent_set"}:
        raise ValueError(
            "proposal_mode must be 'batch' or "
            "'random_independent_set'"
        )
    if proposal_trials < 1:
        raise ValueError("proposal_trials must be positive")
    if proposal_mode != "batch" and proposal_trials != 1:
        raise ValueError(
            "proposal_trials greater than one requires batch mode"
        )
    if proposal_semantics not in {"immutable", "inplace"}:
        raise ValueError(
            "proposal_semantics must be 'immutable' or 'inplace'"
        )
    if single_flip_polish_iterations < 0:
        raise ValueError(
            "single_flip_polish_iterations must be non-negative"
        )
    if polish_candidate_count < 1:
        raise ValueError("polish_candidate_count must be positive")

    start_time = time.perf_counter()
    spins = cp.stack(
        [spin.astype(cp.int8).copy() for spin in spins_list], axis=1
    )
    n, k = spins.shape
    spins_current = spins.T.reshape(-1).copy()
    spins_proposal = cp.empty_like(spins_current)

    energy_dtype = cp.float64 if energy_precision == 64 else cp.float32
    J_energy = J_csr.astype(energy_dtype)

    def evaluate_initial():
        return _compute_batch_energies(
            J_energy,
            spins_current,
            n,
            k,
            energy_dtype=energy_dtype,
            chunk_size=energy_chunk_size,
        )

    current_energies, initial_energy_evaluation_ms = _time_gpu_operation(
        evaluate_initial
    )
    initial_energies = current_energies.copy()

    (
        current_energies,
        main_iterations,
        main_totals,
        main_round_rows,
    ) = _run_guarded_phase(
        phase="main",
        J_csr=J_csr,
        J_energy=J_energy,
        spins_current=spins_current,
        spins_proposal=spins_proposal,
        current_energies=current_energies,
        n=n,
        k=k,
        max_iterations=max_iterations,
        subset_fraction=subset_fraction,
        patience=patience,
        local_dtype=dtype,
        energy_dtype=energy_dtype,
        energy_chunk_size=energy_chunk_size,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        collect_round_metrics=collect_round_metrics,
        adaptive_batches=adaptive_batches,
        minimum_subset_fraction=minimum_subset_fraction,
        batch_shrink_factor=batch_shrink_factor,
        batch_growth_factor=batch_growth_factor,
        rejection_threshold=rejection_threshold,
        adaptation_scope=adaptation_scope,
        proposal_mode=proposal_mode,
        proposal_trials=proposal_trials,
        proposal_semantics=proposal_semantics,
    )

    final_max_iterations = int(final_sweep_iters) * int(num_final_rounds)
    (
        current_energies,
        final_iterations,
        final_totals,
        final_round_rows,
    ) = _run_guarded_phase(
        phase="final",
        J_csr=J_csr,
        J_energy=J_energy,
        spins_current=spins_current,
        spins_proposal=spins_proposal,
        current_energies=current_energies,
        n=n,
        k=k,
        max_iterations=final_max_iterations,
        subset_fraction=final_sweep_fraction,
        patience=patience,
        local_dtype=dtype,
        energy_dtype=energy_dtype,
        energy_chunk_size=energy_chunk_size,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        collect_round_metrics=collect_round_metrics,
        adaptive_batches=adaptive_batches,
        minimum_subset_fraction=minimum_subset_fraction,
        batch_shrink_factor=batch_shrink_factor,
        batch_growth_factor=batch_growth_factor,
        rejection_threshold=rejection_threshold,
        adaptation_scope=adaptation_scope,
        proposal_mode=proposal_mode,
        proposal_trials=proposal_trials,
        proposal_semantics=proposal_semantics,
    )

    (
        current_energies,
        polish_iterations,
        polish_totals,
        polish_round_rows,
    ) = _run_single_flip_polish(
        J_csr=J_csr,
        J_energy=J_energy,
        spins_current=spins_current,
        spins_proposal=spins_proposal,
        current_energies=current_energies,
        n=n,
        k=k,
        max_iterations=single_flip_polish_iterations,
        candidate_limit=polish_candidate_count,
        local_dtype=dtype,
        energy_dtype=energy_dtype,
        energy_chunk_size=energy_chunk_size,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        collect_round_metrics=collect_round_metrics,
    )

    cp.cuda.Stream.null.synchronize()
    elapsed_seconds = time.perf_counter() - start_time
    initial_host = cp.asnumpy(initial_energies)
    final_host = cp.asnumpy(current_energies)

    summary = {
        "safe_update_version": SAFE_UPDATE_VERSION,
        "candidate_count": int(k),
        "node_count": int(n),
        "main_iterations": int(main_iterations),
        "final_iterations": int(final_iterations),
        "polish_iterations": int(polish_iterations),
        "polish_candidate_count": int(polish_candidate_count),
        "initial_energy_evaluation_ms": initial_energy_evaluation_ms,
        "local_search_seconds": elapsed_seconds,
        "energy_precision": int(energy_precision),
        "energy_chunk_size": int(energy_chunk_size),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "main_patience": int(patience),
        "adaptive_batches": bool(adaptive_batches),
        "adaptation_scope": adaptation_scope,
        "proposal_mode": proposal_mode,
        "proposal_trials": int(proposal_trials),
        "proposal_semantics": proposal_semantics,
        "minimum_subset_fraction": float(minimum_subset_fraction),
        "batch_shrink_factor": float(batch_shrink_factor),
        "batch_growth_factor": float(batch_growth_factor),
        "rejection_threshold": float(rejection_threshold),
        "subset_fraction": float(subset_fraction),
        "final_sweep_fraction": float(final_sweep_fraction),
        "initial_min_energy_normalized": float(initial_host.min()),
        "final_min_energy_normalized": float(final_host.min()),
        "candidate_energy_increases": int(
            (final_host > initial_host).sum()
        ),
    }

    for phase_totals in (main_totals, final_totals, polish_totals):
        phase = phase_totals["phase"]
        for key, value in phase_totals.items():
            if key != "phase":
                summary[f"{phase}_{key}"] = value

    summary["total_accepted_updates"] = (
        main_totals["accepted_updates"]
        + final_totals["accepted_updates"]
        + polish_totals["accepted_updates"]
    )
    summary["total_rejected_updates"] = (
        main_totals["rejected_updates"]
        + final_totals["rejected_updates"]
        + polish_totals["rejected_updates"]
    )
    summary["total_proposed_flips"] = (
        main_totals["proposed_flips"]
        + final_totals["proposed_flips"]
        + polish_totals["proposed_flips"]
    )
    summary["total_accepted_flips"] = (
        main_totals["accepted_flips"]
        + final_totals["accepted_flips"]
        + polish_totals["accepted_flips"]
    )
    summary["total_rejected_flips"] = (
        main_totals["rejected_flips"]
        + final_totals["rejected_flips"]
        + polish_totals["rejected_flips"]
    )
    summary["total_monotonicity_violations"] = (
        main_totals["monotonicity_violations"]
        + final_totals["monotonicity_violations"]
        + polish_totals["monotonicity_violations"]
    )
    guarded_updates = (
        summary["total_accepted_updates"] + summary["total_rejected_updates"]
    )
    summary["update_rejection_rate"] = (
        summary["total_rejected_updates"] / guarded_updates
        if guarded_updates
        else 0.0
    )

    if summary["candidate_energy_increases"] != 0:
        raise RuntimeError(
            "Guard invariant failed: at least one final candidate has higher "
            "energy than its initial candidate"
        )
    if summary["total_monotonicity_violations"] != 0:
        raise RuntimeError(
            "Guard invariant failed: a committed update was not a strict "
            "energy decrease"
        )

    final_spins = spins_current.reshape(k, n).T.astype(cp.int8)
    return (
        final_spins,
        main_iterations,
        summary,
        main_round_rows + final_round_rows + polish_round_rows,
    )


