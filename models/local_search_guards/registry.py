"""Maps each local-search variant name to its module."""

from __future__ import annotations

from importlib import import_module


REFINERS = {
    "Safe-Fixed": "models.local_search_guards.safe_fixed",
    "Safe-Adaptive": "models.local_search_guards.safe_adaptive",
    "Conflict-Free": "models.local_search_guards.conflict_free",
    "Sequential-Oracle": "models.local_search_guards.sequential_oracle",
}


def get_refiner(name: str):
    try:
        module_name = REFINERS[name]
    except KeyError as exc:
        choices = ", ".join(REFINERS)
        raise KeyError(f"Unknown guard variant {name!r}; choose {choices}") from exc
    return import_module(module_name).refine
