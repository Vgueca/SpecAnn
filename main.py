#!/usr/bin/env python3
"""Run SpecAnn, SpecAdj or SpecSignedLap on a dataset.

    python main.py --solver SpecAnn --dataset Gset --instance G1.txt
"""
import sys

from utils.core import run_solver
from utils.utils import load_config, merge_args_with_config, parse_arguments


def main() -> int:
    args = parse_arguments()
    config = merge_args_with_config(args, load_config(args.config))
    print(f"Run: {config['run_name']} ({config['solver']}), dataset: {config['dataset']}")
    print("=" * 50)
    results = run_solver(config)
    print("=" * 50)
    print(f"Finished: {results['successful_runs']}/{results['total_runs']} runs OK"
          + (f", {results['failed_runs']} failed:" if results['failed_runs'] else "."))
    for instance, seed, error in results['failures']:
        print(f"  {instance} seed={seed}: {error}")
    return 2 if results['failed_runs'] else 0


if __name__ == "__main__":
    sys.exit(main())
