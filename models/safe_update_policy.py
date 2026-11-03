"""Acceptance and batch-size rules for guarded parallel spin updates (work with NumPy or CuPy)."""


def strict_energy_acceptance(
    current_energies,
    proposal_energies,
    has_proposed_flips,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    xp,
):
    """Accept a proposal only if it has flips, finite energies, and an energy drop above the tolerance.

    ``xp`` is NumPy (CPU tests) or CuPy (GPU). Returns (accepted, delta, tolerance).
    """
    current = xp.asarray(current_energies)
    proposal = xp.asarray(proposal_energies)
    has_flips = xp.asarray(has_proposed_flips, dtype=xp.bool_)

    scale = xp.maximum(xp.asarray(1.0, dtype=current.dtype), xp.abs(current))
    tolerance = (
        xp.asarray(absolute_tolerance, dtype=current.dtype)
        + xp.asarray(relative_tolerance, dtype=current.dtype) * scale
    )
    delta = proposal - current
    finite = xp.isfinite(current) & xp.isfinite(proposal)
    accepted = has_flips & finite & (delta < -tolerance)
    return accepted, delta, tolerance


def adapt_batch_fraction(
    current_fraction,
    *,
    initial_fraction,
    minimum_fraction,
    accepted_updates,
    rejected_updates,
    rejection_threshold,
    shrink_factor,
    growth_factor,
):
    """Return the next batch fraction (shared by all candidates) and whether it really shrank."""
    guarded_updates = accepted_updates + rejected_updates
    rejection_rate = (
        rejected_updates / guarded_updates if guarded_updates else 0.0
    )

    if (
        rejected_updates
        and rejection_rate >= rejection_threshold
        and current_fraction > minimum_fraction
    ):
        next_fraction = max(
            minimum_fraction, current_fraction * shrink_factor
        )
        return next_fraction, next_fraction < current_fraction

    if accepted_updates and rejected_updates == 0:
        return (
            min(initial_fraction, current_fraction * growth_factor),
            False,
        )

    return current_fraction, False


def adapt_candidate_fractions(
    current_fractions,
    accepted_updates,
    rejected_updates,
    *,
    initial_fraction,
    minimum_fraction,
    shrink_factor,
    growth_factor,
    xp,
):
    """Adapt the batch fraction separately for each candidate.

    A rejected candidate retries with a smaller batch; an accepted one grows back toward the initial fraction.
    """
    current = xp.asarray(current_fractions)
    accepted = xp.asarray(accepted_updates, dtype=xp.bool_)
    rejected = xp.asarray(rejected_updates, dtype=xp.bool_)

    minimum = xp.asarray(minimum_fraction, dtype=current.dtype)
    initial = xp.asarray(initial_fraction, dtype=current.dtype)
    smaller = xp.maximum(minimum, current * shrink_factor)
    larger = xp.minimum(initial, current * growth_factor)

    updated = xp.where(rejected, smaller, current)
    updated = xp.where(accepted, larger, updated)
    actually_shrunk = rejected & (updated < current)
    return updated, actually_shrunk
