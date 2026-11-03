"""Command-line arguments and the YAML config."""
import argparse
import re
from pathlib import Path
from typing import Any, Dict

import yaml


class _ConfigLoader(yaml.SafeLoader):
    """YAML SafeLoader that also reads numbers such as 1e-6 as floats.

    Plain PyYAML (YAML 1.1) needs a dot (1.0e-6) and reads 1e-6 as a string.
    Quoted values stay strings.
    """


_ConfigLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$'),
    list('-+0123456789'),
)


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="SpecAnn: run a spectral solver on the instances of a dataset")
    parser.add_argument('--solver', required=True,
                        help='Run name from the config file (e.g. SpecAnn, SpecAdj, SpecSignedLap)')
    parser.add_argument('--dataset', required=True, help='Dataset folder in datasets/')
    parser.add_argument('--instance', nargs='+', default=None,
                        help='Instance file name(s) (default: every instance of the dataset)')
    parser.add_argument('--config', default='default', help='Config file name in configs/ (default: default)')
    parser.add_argument('--num-seeds', dest='num_seeds', type=int, default=None,
                        help='Seeds per instance: random_seed, random_seed+1, ... (overrides the config file)')
    parser.add_argument('--seed', dest='seed', type=int, default=None,
                        help='First seed (with --num-seeds; alone: this seed only)')
    parser.add_argument('--save-spins', dest='save_spins', action='store_true', default=None,
                        help='Write the best spin vector of each run to results/<dataset>/<run>/solutions/')
    parser.add_argument('--no-save-spins', dest='save_spins', action='store_false',
                        help='Do not write spin vectors (overrides the config file)')
    return parser.parse_args()


def load_config(config_name: str) -> Dict[str, Any]:
    """Load configs/<config_name>.yml."""
    config_path = Path(f"configs/{config_name}.yml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.load(f, Loader=_ConfigLoader)


def merge_args_with_config(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """The run config: the top-level values of the config file, the params of the run named by --solver,
    then the command-line flags (which win)."""
    if args.solver not in config or not isinstance(config[args.solver], dict):
        runs = sorted(k for k, v in config.items() if isinstance(v, dict))
        raise ValueError(f"Unknown run name {args.solver!r}; the config file has: {', '.join(runs)}")
    run = config[args.solver]
    merged = {k: v for k, v in config.items() if not isinstance(v, dict)}
    merged.update(run.get('params') or {})
    merged['solver'] = run['solver_name']
    merged['run_name'] = args.solver
    merged['dataset'] = args.dataset
    if args.instance:
        merged['instance'] = list(args.instance)

    if getattr(args, 'save_spins', None) is not None:
        merged['save_spins'] = args.save_spins
    if getattr(args, 'num_seeds', None) is not None:
        if args.num_seeds < 1:
            raise ValueError(f"--num-seeds must be >= 1, got {args.num_seeds}")
        merged['num_seeds'] = args.num_seeds
    if getattr(args, 'seed', None) is not None:
        if args.seed < 0:
            raise ValueError(f"--seed must be >= 0, got {args.seed}")
        merged['random_seed'] = args.seed
        if getattr(args, 'num_seeds', None) is None:
            merged['num_seeds'] = 1                 # --seed alone: this seed only
    return merged
