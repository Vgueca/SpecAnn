"""The solvers of this repository, by the `solver_name` a run gives in configs/."""
import importlib

SOLVERS = {
    'spectral_annealing': ('models.spectral_annealing', 'SpectralAnnealing'),   # SpecAnn
    'spec_adj': ('models.spec_adj', 'Alpha0GPU'),                               # SpecAdj (alpha = 0)
    'spec_signed_lap': ('models.spec_signed_lap', 'Alpha1GPU'),                 # SpecSignedLap (alpha = 1)
}


def get_solver_class(solver_name: str):
    """Import and return the class of a solver name."""
    if solver_name not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_name!r}; choose one of {sorted(SOLVERS)}")
    module, cls = SOLVERS[solver_name]
    return getattr(importlib.import_module(module), cls)
