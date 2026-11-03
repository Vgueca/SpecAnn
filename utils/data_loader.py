"""Load Ising instances from a dataset folder as sparse coupling matrices.

Two text formats (nodes numbered from 1):
- Gset (a folder whose name contains "Gset"): a line "n m", then one line "i j w" per edge; the weights are
  negated, so that the lowest energy is the largest cut.
- Any other folder: a line "n m", then n lines with the linear field h_i of each node, then one line
  "i j J" per coupling. Nonzero fields are carried by one extra spin (the bias node).
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class DataLoadError(Exception):
    """Raised when an instance file cannot be read."""


@dataclass
class InstanceIndex:
    """Map from the rows of a loaded matrix back to the nodes of the instance file.

    Loading drops isolated nodes and may add a bias node that carries the linear fields. node_ids[i] is
    the 0-based file node of row i; the id written in the file is node_ids[i] + id_base.
    """
    n_original: int                 # node count declared by the file (without the bias node)
    n_solve: int                    # rows of the matrix handed to the solver
    node_ids: np.ndarray            # int64, length n_solve, original 0-based ids
    bias_node_index: int = -1       # row of the bias node, -1 if none
    id_base: int = 1                # the file numbers nodes from 1
    weights_negated: bool = False   # True for Gset (MaxCut weights negated)
    J_sum: float = 0.0              # sum of the loaded matrix, to recover cut values


class DataLoader:
    """Load the instances of one dataset folder (datasets/<name>)."""

    def __init__(self, dataset_name: str, datasets_path: str = "datasets"):
        self.dataset_name = dataset_name
        self.dataset_path = Path(datasets_path) / dataset_name
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        self.dataset_format = 'gset' if 'gset' in dataset_name.lower() else 'graph_linear_bias'

    def list_instances(self) -> List[str]:
        """Return the sorted instance file names (*.txt) of the dataset."""
        return sorted(f.name for f in self.dataset_path.glob("*.txt"))

    def load_instance(self, instance_name: str, return_index: bool = False):
        """Load an instance and return its coupling matrix J (sparse CSR), or (J, InstanceIndex)."""
        instance_path = self.dataset_path / instance_name
        if not instance_path.exists():
            raise FileNotFoundError(f"Instance file not found: {instance_path}")
        if self.dataset_format == 'gset':
            J, index = self._load_gset_instance(instance_path)
        else:
            J, index = self._load_linear_bias_instance(instance_path)
        return (J, index) if return_index else J

    def _finalize(self, J: csr_matrix, n_original: int, bias_original_id: int,
                  id_base: int, weights_negated: bool) -> Tuple[csr_matrix, InstanceIndex]:
        """Drop isolated nodes and record how the surviving rows map to file node ids."""
        node_ids = np.arange(J.shape[0], dtype=np.int64)
        J, node_ids = self._remove_isolated_nodes(J, node_ids)

        if bias_original_id >= 0:
            hits = np.where(node_ids == bias_original_id)[0]
            bias_node_index = int(hits[0]) if hits.size else -1
        else:
            bias_node_index = -1

        index = InstanceIndex(
            n_original=n_original,
            n_solve=int(J.shape[0]),
            node_ids=node_ids,
            bias_node_index=bias_node_index,
            id_base=id_base,
            weights_negated=weights_negated,
            J_sum=float(J.sum()),
        )
        return J, index
    
    def _robust_parse_line(self, line: str, expected_parts: int = 2) -> Optional[Tuple]:
        """Parse a line "i j [w]" into (i, j, w); w = 1.0 if expected_parts is 2.

        Return None for blank lines, comments and lines that do not parse.
        """
        try:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                return None
                
            parts = line.split()
            if len(parts) < expected_parts:
                return None
            
            if expected_parts == 2:
                i = int(float(parts[0]))
                j = int(float(parts[1]))
                return (i, j, 1.0)
            else:
                i = int(float(parts[0]))
                j = int(float(parts[1]))
                w = float(parts[2])
                return (i, j, w)
        except (ValueError, IndexError, TypeError):
            return None
    
    def _load_gset_instance(self, file_path: Path) -> Tuple[csr_matrix, InstanceIndex]:
        """Load a Gset file: a line "n m", then lines "i j w" (nodes from 1)."""
        try:
            with open(file_path, 'r') as f:
                n, m = map(int, f.readline().strip().split())
                
                rows, cols, data = [], [], []
                for line in f:
                    parsed = self._robust_parse_line(line, expected_parts=3)
                    if parsed is not None:
                        i, j, w = parsed
                        i, j = i - 1, j - 1
                        if 0 <= i < n and 0 <= j < n and i != j:
                            rows.extend([i, j])
                            cols.extend([j, i])
                            data.extend([-w, -w])
                
                J = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
                # Weights are negated above so that minimizing energy maximizes the cut.
                return self._finalize(J, n_original=n, bias_original_id=-1, id_base=1,
                                      weights_negated=True)
        except Exception as e:
            raise DataLoadError(f"Error reading Gset file {file_path}: {e}")
    
    def _load_linear_bias_instance(self, file_path: Path) -> Tuple[csr_matrix, InstanceIndex]:
        """Load a text file: "n m", n lines with h, then lines "i j w" (nodes from 1).

        Nonzero h become edges to an added bias node.
        """
        try:
            with open(file_path, 'r') as f:
                n, m = map(int, f.readline().strip().split())
                
                node_weights = [float(f.readline().strip()) for _ in range(n)]
                
                edges, weights = [], []
                for line in f:
                    parsed = self._robust_parse_line(line, expected_parts=3)
                    if parsed is not None:
                        i, j, w = parsed
                        i, j = i - 1, j - 1
                        if 0 <= i < n and 0 <= j < n and i != j:
                            edges.append((i, j))
                            weights.append(w)
                
                add_bias_node = any(h != 0 for h in node_weights)
                new_node_id = n
                n_vertices = n + 1 if add_bias_node else n
                
                if add_bias_node:
                    for i, h in enumerate(node_weights):
                        if h != 0:
                            edges.append((new_node_id, i))
                            weights.append(h)
                
                rows, cols, data = [], [], []
                for (i, j), w in zip(edges, weights):
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([w, w])
                
                J = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
                return self._finalize(J, n_original=n,
                                      bias_original_id=new_node_id if add_bias_node else -1,
                                      id_base=1, weights_negated=False)
        except Exception as e:
            raise DataLoadError(f"Error reading linear bias file {file_path}: {e}")
    
    def _remove_isolated_nodes(self, J: csr_matrix,
                               node_ids: Optional[np.ndarray] = None):
        """Remove isolated nodes (all-zero rows and columns) from J.

        With node_ids, return (J_reduced, node_ids of the kept nodes); else only J_reduced.
        """
        track = node_ids is not None

        if J.nnz == 0:
            return (J, node_ids) if track else J
        
        # J is symmetric, so row sums are enough.
        row_sums = np.array(np.abs(J).sum(axis=1)).flatten()
        connected_nodes = np.where(row_sums != 0)[0]
        
        n_isolated = J.shape[0] - len(connected_nodes)
        if n_isolated > 0:
            warnings.warn(f"Removed {n_isolated} isolated node(s) (no couplings); the matrix has "
                          f"{len(connected_nodes)} of {J.shape[0]} nodes.", UserWarning)
            J_reduced = J[np.ix_(connected_nodes, connected_nodes)]
            if track:
                return J_reduced, node_ids[connected_nodes]
            return J_reduced

        return (J, node_ids) if track else J
