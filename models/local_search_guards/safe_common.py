"""Fixed and Adaptive Guard: thin wrapper around ``safe_flip_refinement`` with shared summary names."""

from __future__ import annotations

import cupy as cp

from models.spectral_annealing_safe import safe_flip_refinement

from .common import RefinementResult, peak_cupy_memory


def refine_safe(matrix, spins, config, *, adaptive: bool) -> RefinementResult:
    cp.random.seed(int(config["local_search_seed"]))
    spins = cp.asarray(spins, dtype=cp.int8)
    spin_list = [spins[index].copy() for index in range(spins.shape[0])]
    final_spins_nk, _, summary, round_rows = safe_flip_refinement(
        matrix,
        spin_list,
        max_iterations=int(config["max_iterations"]),
        final_sweep_fraction=float(config["final_sweep_fraction"]),
        final_sweep_iters=int(config["final_sweep_iters"]),
        num_final_rounds=int(config["final_sweep_rounds"]),
        subset_fraction=float(config["subset_fraction"]),
        dtype=config["local_dtype"],
        energy_precision=64,
        energy_chunk_size=int(config["energy_chunk_size"]),
        absolute_tolerance=float(config["absolute_tolerance"]),
        relative_tolerance=float(config["relative_tolerance"]),
        patience=int(config["safe_patience"]),
        collect_round_metrics=True,
        adaptive_batches=adaptive,
        minimum_subset_fraction=float(config["minimum_subset_fraction"]),
        batch_shrink_factor=float(config["batch_shrink_factor"]),
        batch_growth_factor=float(config["batch_growth_factor"]),
        rejection_threshold=float(config["rejection_threshold"]),
        adaptation_scope="per_candidate",
        proposal_mode="batch",
        proposal_trials=1,
        proposal_semantics="immutable",
        single_flip_polish_iterations=0,
    )
    final_spins = final_spins_nk.T.copy()
    summary = dict(summary)
    summary.update(peak_cupy_memory())
    summary["implementation"] = (
        "safe_adaptive" if adaptive else "safe_fixed"
    )
    summary["kernel_updates"] = (
        int(summary.get("main_rounds", 0))
        + int(summary.get("final_rounds", 0))
    )
    summary["candidate_updates"] = int(
        summary.get("main_candidate_updates", 0)
        + summary.get("final_candidate_updates", 0)
    )
    summary["candidate_updates_with_flips"] = int(
        summary.get("main_candidates_with_flips", 0)
        + summary.get("final_candidates_with_flips", 0)
    )
    summary["proposed_flips"] = int(
        summary.get("total_proposed_flips", 0)
    )
    summary["executed_flips"] = int(
        summary.get("total_accepted_flips", 0)
    )
    summary["accepted_updates"] = int(
        summary.get("total_accepted_updates", 0)
    )
    summary["rejected_updates"] = int(
        summary.get("total_rejected_updates", 0)
    )
    summary["energy_increasing_candidate_updates"] = int(
        summary.get("main_proposed_energy_increases", 0)
        + summary.get("final_proposed_energy_increases", 0)
    )
    summary["accepted_monotonicity_violations"] = int(
        summary.get("total_monotonicity_violations", 0)
    )

    normalized_rows = []
    for kernel_index, row in enumerate(round_rows, start=1):
        row = dict(row)
        row["kernel_index"] = kernel_index
        row["energy_increasing_candidate_updates"] = row.get(
            "proposed_energy_increases", 0
        )
        row["accepted_monotonicity_violations"] = row.get(
            "monotonicity_violations", 0
        )
        row["executed_flips"] = row.get("accepted_flips", 0)
        normalized_rows.append(row)

    return RefinementResult(
        spins=final_spins,
        local_search_seconds=float(summary["local_search_seconds"]),
        summary=summary,
        round_rows=normalized_rows,
    )
