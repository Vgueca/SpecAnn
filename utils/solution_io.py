"""Write and read best spin configurations as plain-text ``.sol`` files.

    results/<dataset>/<run>/solutions/<instance>__seed<seed>.sol

A ``#`` header, then two columns ``node_id spin`` in the node numbering of the instance file (isolated
nodes, dropped by the loader, are missing: any spin fits them). Energy: E = -0.5 * s^T J s.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Header keys read back as numbers.
_INT_KEYS = {'seed', 'n_original', 'n_solve', 'id_base', 'bias_node', 'gauge_flipped', 'weights_negated'}
_FLOAT_KEYS = {'J_sum', 'energy', 'cut_value'}


def ising_energy(J, s: np.ndarray) -> float:
    """E = -0.5 * s^T J s, in float64."""
    v = np.asarray(s, dtype=np.float64).ravel()
    return float(-0.5 * v @ (J @ v))


def cut_value(energy: float, J_sum: float, weights_negated: bool) -> Optional[float]:
    """The cut weight of a MaxCut instance (loaded with J = -W): (W_tot - E) / 2; None for other instances."""
    if not weights_negated:
        return None
    return (-0.5 * J_sum - energy) / 2.0


def solution_path(results_dir, instance_name: str, seed) -> Path:
    """results_dir/solutions/<instance without .txt>__seed<seed>.sol"""
    base = Path(instance_name).name
    if base.endswith('.txt'):
        base = base[:-4]
    return Path(results_dir) / "solutions" / f"{base}__seed{seed}.sol"


def save_solution(path, spins, J, index, *, dataset: str, instance_name: str, run_name: str, seed) -> Path:
    """Write the spin vector returned by a solver (rows of J) in the node numbering of the instance file."""
    path = Path(path)
    s = np.asarray(spins.get() if hasattr(spins, 'get') else spins).astype(np.int8).ravel()
    energy = ising_energy(J, s)
    node_ids = np.asarray(index.node_ids, dtype=np.int64)

    # Flip all spins so that the bias node is +1, then drop it: the vector then solves the problem as
    # written in the file (Solution.to_solve_space() undoes this).
    bias = int(index.bias_node_index)
    gauge_flipped = False
    if bias >= 0:
        if s[bias] == -1:
            s, gauge_flipped = (-s).astype(np.int8), True
        keep = np.arange(s.size) != bias
        s, node_ids = s[keep], node_ids[keep]

    header = [
        ('dataset', dataset), ('instance', instance_name), ('run_name', run_name), ('seed', seed),
        ('n_original', index.n_original), ('n_solve', index.n_solve), ('id_base', index.id_base),
        ('bias_node', bias), ('gauge_flipped', int(gauge_flipped)),
        ('weights_negated', int(bool(index.weights_negated))), ('J_sum', index.J_sum),
        ('energy', energy), ('cut_value', cut_value(energy, index.J_sum, index.weights_negated)),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".sol.tmp.{os.getpid()}")
    with open(tmp, 'w') as f:
        for key, value in header:
            f.write(f"# {key} {'' if value is None else value}\n")
        f.write("# columns node_id spin\n")
        np.savetxt(f, np.column_stack([node_ids + index.id_base, s.astype(np.int64)]), fmt='%d %+d')
    os.replace(tmp, path)
    return path


@dataclass
class Solution:
    """A spin configuration in the node numbering of the instance file."""
    node_ids: np.ndarray            # int64, node ids as written in the file
    spins: np.ndarray               # int8, +1/-1, aligned with node_ids
    header: Dict[str, Any] = field(default_factory=dict)

    @property
    def energy(self) -> Optional[float]:
        return self.header.get('energy')

    def to_solve_space(self) -> np.ndarray:
        """The vector over the rows of the loaded matrix: the bias node put back, the gauge flip undone."""
        s = self.spins.astype(np.int8, copy=True)
        bias = int(self.header.get('bias_node', -1))
        if bias >= 0:
            s = np.insert(s, bias, np.int8(1))
        if int(self.header.get('gauge_flipped', 0)):
            s = (-s).astype(np.int8)
        return s


def load_solution(path) -> Solution:
    """Read a ``.sol`` file."""
    header: Dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            parts = line[1:].strip().split(None, 1)
            if not parts or parts[0] == 'columns':
                continue
            key, raw = parts[0], (parts[1].strip() if len(parts) > 1 else '')
            if raw == '':
                header[key] = None
            elif key in _INT_KEYS:
                header[key] = int(float(raw))
            elif key in _FLOAT_KEYS:
                header[key] = float(raw)
            else:
                header[key] = raw
    data = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if data.size == 0:
        return Solution(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8), header)
    return Solution(data[:, 0].astype(np.int64), data[:, 1].astype(np.int8), header)
