"""Run a solver on the instances of a dataset, for several seeds.

Each solver writes its own results (results/<dataset>/<run>/<instance>.csv and the per-alpha CSVs); this
module loads the instances, calls the solver for every seed, writes the spin vectors on request, and counts
the runs that fail.
"""
import inspect
import re
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict

from .data_loader import DataLoader
from .failures import is_out_of_memory
from .selector import get_solver_class
from .solution_io import save_solution, solution_path

# Config keys used by the pipeline itself; every other key is passed to the solver.
PIPELINE_KEYS = frozenset({'solver', 'dataset', 'instance', 'num_seeds', 'random_seed', 'run_name',
                           'save_spins', 'warmup'})

# Name of the throwaway instance solved by the warm-up.
WARMUP_INSTANCE = '__warmup__.txt'


def _filter_params(solver_class, params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the params the solver's constructor does not name, with a warning (they are often typos)."""
    signature = inspect.signature(solver_class.__init__)
    named = {name for name, p in signature.parameters.items() if p.kind is not inspect.Parameter.VAR_KEYWORD}
    unknown = sorted(k for k in params if k not in named)
    if unknown:
        warnings.warn(f"{solver_class.__name__} does not take {unknown}; these config keys are ignored.",
                      UserWarning)
    return {k: v for k, v in params.items() if k in named}


def _warmup_instance(n: int = 64, seed: int = 0):
    """A small random +-1 Ising graph, used only to warm up a solver."""
    import numpy as np
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    A = sp.triu(sp.random(n, n, density=0.2, random_state=rng, data_rvs=lambda k: rng.choice([-1.0, 1.0], k)), 1)
    return (A + A.T).tocsr()


def _warm_up(solver_class, solver_params, dataset, run_name, seed, out_dir):
    """Solve a small synthetic instance before the timed runs, so that one-time GPU setup costs are not
    charged to the first instance. Its files are removed."""
    stem = Path(WARMUP_INSTANCE).stem
    before = set(out_dir.rglob(f'*{stem}*')) if out_dir.exists() else set()
    try:
        solver_class(instance_name=WARMUP_INSTANCE, dataset=dataset, run_name=run_name, random_seed=seed,
                     **solver_params).solve(_warmup_instance())
    except Exception as e:
        print(f"  WARNING: warm-up failed ({type(e).__name__}: {e}); the first instance's time includes "
              f"one-time initialization costs")
    finally:
        for path in (sorted(out_dir.rglob(f'*{stem}*')) if out_dir.exists() else []):
            if path.is_file() and path not in before:
                path.unlink()


def _size(name: str) -> int:
    """The first number in an instance name ("G7.txt" -> 7, "ba_35_1_2.txt" -> 35), 0 if none."""
    match = re.search(r"(\d+)", name.rsplit('.', 1)[0])
    return int(match.group(1)) if match else 0


def run_solver(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the solver on every instance and seed of the config; return a summary of the runs."""
    loader = DataLoader(config['dataset'])
    instances = list(config['instance']) if config.get('instance') else loader.list_instances()
    if isinstance(config.get('instance'), str):
        instances = [config['instance']]
    instances.sort(key=_size)

    solver_class = get_solver_class(config['solver'])
    params = _filter_params(solver_class, {k: v for k, v in config.items() if k not in PIPELINE_KEYS})
    seeds = [int(config.get('random_seed', 42)) + i for i in range(int(config.get('num_seeds', 1)))]
    run_name = config.get('run_name', config['solver'])
    out_dir = Path("results") / config['dataset'] / run_name

    if config.get('warmup'):
        print("Warm-up solve on a small synthetic instance (not recorded)")
        _warm_up(solver_class, params, config['dataset'], run_name, seeds[0], out_dir)

    total, failed = 0, []
    for instance_name in instances:
        print(f"Processing instance: {instance_name}")
        try:
            J, index = loader.load_instance(instance_name, return_index=True)
        except Exception as e:
            traceback.print_exc()
            failed += [(instance_name, seed, f"load: {type(e).__name__}: {e}") for seed in seeds]
            total += len(seeds)
            continue

        for i, seed in enumerate(seeds):
            print(f"  Seed {i + 1}/{len(seeds)} (seed={seed})")
            total += 1
            try:
                solver = solver_class(instance_name=instance_name, dataset=config['dataset'], run_name=run_name,
                                      random_seed=seed, **params)
                energy, spins, _ = solver.solve(J)
                if config.get('save_spins'):
                    save_solution(solution_path(out_dir, instance_name, seed), spins, J, index,
                                  dataset=config['dataset'], instance_name=instance_name, run_name=run_name,
                                  seed=seed)
            except Exception as e:
                traceback.print_exc()
                what = "out of memory" if is_out_of_memory(e) else type(e).__name__
                failed.append((instance_name, seed, f"{what}: {e}"))
                e = None            # the traceback holds the solver's GPU arrays: release them
                print(f"  FAILED: {instance_name} seed={seed}: {failed[-1][2]}")

    return {'run_name': run_name, 'dataset': config['dataset'], 'total_runs': total,
            'successful_runs': total - len(failed), 'failed_runs': len(failed), 'failures': failed}
