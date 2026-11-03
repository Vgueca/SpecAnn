"""
SpecAdj, the alpha = 0 end of SpecAnn's path, on the GPU: the signs of the leading eigenvector
of the coupling matrix J, then the GPU local search.
"""

import numpy as np
import time
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
from utils.eigsh import eigsh


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Alpha0GPU:
    """
    SpecAdj solver (alpha = 0).

    Parameters:
        instance_name: instance file name; names the results CSV.
        dataset: dataset name; results go to results/<dataset>/<run_name>/.
        random_seed: seed of CuPy's random generator (None: not seeded).
        mode: stored in the results; not used.
        beta: not used.
        alpha: not used (always 0).
        use_local_search: refine the rounded vector with the GPU local search.
        max_local_iterations: maximum number of local search passes.
        eigsh_tol: tolerance of eigsh.
        eigsh_maxiter: maximum number of eigsh iterations.
        run_name: name of the run and of its results folder (default "alpha0_gpu").
        ls_policy: "default" (parallel in-place local search, as in SpecAnn) or a guarded
            policy of models/local_search_guards.
        precision: 32 or 64. At 32, J and eigsh use float32, energies are summed in float64,
            and the reported energy is recomputed in float64 from the input J.
        **kwargs: other keys; printed and ignored.
    """

    def __init__(self, instance_name: str, dataset: str, random_seed: int = None,
                 mode: int = 0, beta: float = None, alpha: float = 0,
                 use_local_search: bool = True, max_local_iterations: int = 1000,
                 eigsh_tol: float = 1e-6, eigsh_maxiter: int = 1000, run_name: str = None,
                 ls_policy: str = "default", precision: int = 32, **kwargs):
        self.instance_name = instance_name
        from models.local_search_guards.policies import LS_POLICIES
        if ls_policy not in LS_POLICIES:
            raise ValueError(f"ls_policy must be one of {LS_POLICIES}, got {ls_policy!r}")
        self.ls_policy = ls_policy
        self.dataset = dataset
        self.random_seed = random_seed
        self.mode = mode
        self.beta = beta
        self.alpha = 0
        self.use_local_search = use_local_search
        self.max_local_iterations = max_local_iterations
        self.eigsh_tol = float(eigsh_tol)  
        self.eigsh_maxiter = int(eigsh_maxiter)
        if precision not in (32, 64):
            raise ValueError(f"precision must be 32 or 64, got {precision}")
        self.precision = int(precision)
        self.dtype = cp.float32 if self.precision == 32 else cp.float64

        self.solver_name = run_name if run_name else "alpha0_gpu"
        
        if random_seed is not None:
            cp.random.seed(random_seed)
            
        print(f"Initialized {self.solver_name} solver (GPU sparse). Results will be saved under '{self.solver_name}'.")
        

        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  mode: {self.mode}")
        print(f"  beta: {self.beta}")
        print(f"  alpha: {self.alpha}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  max_local_iterations: {self.max_local_iterations}")
        print(f"  eigsh_tol: {self.eigsh_tol}")
        print(f"  eigsh_maxiter: {self.eigsh_maxiter}")
        print(f"  precision: {self.precision} bits ({self.dtype})")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J: np.ndarray):
        """Solve the instance with couplings J. Returns (energy, spins, time in seconds)."""
        start_time = time.time()

        # copy=False: no host copy when J already has this dtype
        J_csr = cpx_sparse.csr_matrix(J.astype(self.dtype, copy=False), dtype=self.dtype)
        J_abs = J_csr.copy()
        J_abs.data = cp.abs(J_abs.data)

        deg_abs_original = J_abs @ cp.ones(J_csr.shape[1], dtype=self.dtype)

        max_deg_abs = cp.max(deg_abs_original)

        self._normalization_factor = float(max_deg_abs)
        
        if self.precision == 32:
            # Scale in float32, sharing indices and indptr (J_csr / c would return float64)
            J_csr = cpx_sparse.csr_matrix((J_csr.data / max_deg_abs, J_csr.indices, J_csr.indptr),
                                          shape=J_csr.shape)
        else:
            J_csr = J_csr / max_deg_abs
        # alpha = 0: D = I, so the matrix is Q = I - J (scaled J)

        evals, evecs = self._solve_eigenvalue_problem(J_csr, k=1)
        # Rounding
        spins = cp.where(evecs[:, 0] >= 0, 1.0, -1.0)
        
        
        energy = self.compute_energy(spins, J_csr)
        local_iterations = 0
        if self.use_local_search:
            print(f"Using local search with {self.max_local_iterations} iterations")
            if self.ls_policy == "default":
                spins_batch, local_iterations = flip_refinement(
                    J_csr, [spins], max_iterations=self.max_local_iterations
                )
            else:
                from models.local_search_guards.policies import refine_candidates
                spins_batch, local_iterations = refine_candidates(
                    self.ls_policy, J_csr, [spins], max_iterations=self.max_local_iterations,
                    dtype=cp.float32, seed=self.random_seed, J_host=J)

            refined_spins = spins_batch[:, 0]
            refined_energy = self.compute_energy(refined_spins, J_csr)
            if refined_energy < energy:
                spins = refined_spins
                energy = refined_energy
                print(f"Local search improved energy to {energy}")
            else:
                print(f"Local search did not improve energy to {energy}")

        
        total_time = time.time() - start_time
        if self.precision == 32:
            # Report the energy in float64 from the input J (as SpecAnn does). After the timer:
            # this is reporting, not search.
            s64 = cp.asnumpy(spins).astype(np.float64)
            energy = float(-0.5 * s64 @ (J @ s64))

        self._store_general_results(energy, self.alpha, 1.0, total_time, local_iterations)
        
        return energy, spins, total_time

    def _solve_eigenvalue_problem(self, J_csr, k: int = 1):
        """The k smallest eigenpairs of I - J, i.e. the k leading eigenvectors of J."""
        n = J_csr.shape[0]
        S_sparse = cpx_sparse.identity(n, dtype=J_csr.dtype, format="csr") - J_csr

        evals, evecs_y, iter_count = _eigsh_finite(S_sparse, k, self.eigsh_tol, self.eigsh_maxiter)

        # eigsh stops silently at maxiter: then, or without the iteration count (stock CuPy), check the residual.
        self._eigsh_iterations = iter_count          # None with stock CuPy
        self._eigsh_converged = True
        if iter_count is None or iter_count >= self.eigsh_maxiter:
            y = evecs_y[:, 0]
            residual = float(cp.linalg.norm(S_sparse @ y - evals[0] * y) / cp.linalg.norm(y))
            self._eigsh_converged = residual <= self.eigsh_tol

        return evals[:k], evecs_y[:, :k]


    def compute_energy(self, spins: "cp.ndarray", J_csr) -> float:
        """Ising energy -0.5 s^T J s of a spin vector, in the units of the input J."""
        if self.precision == 32:
            # float32 couplings, float64 sums (SpecAnn's kernel)
            from models.spectral_annealing import fp32_energies
            return float(fp32_energies(J_csr, spins, getattr(self, '_normalization_factor', 1.0))[0])
        if hasattr(J_csr, 'dot'):
            energy = -0.5 * cp.dot(spins, J_csr.dot(spins))
        else:
            energy = -0.5 * cp.einsum("i,ij,j->", spins, J_csr, spins)

        if hasattr(self, '_normalization_factor'):
            energy *= self._normalization_factor
        return float(energy)


    def _store_general_results(self, energy: float, alpha: float, beta: float, time_taken: float, local_iterations: int):
        """Write (or replace) this run's row in results/<dataset>/<run_name>/<instance>.csv."""
        results_dir = Path(f"results/{self.dataset}/{self.solver_name}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        instance_base = self.instance_name.replace('.txt', '')
        csv_file = results_dir / f"{instance_base}.csv"
        
        result_data = {
            'instance_name': self.instance_name,
            'dataset': self.dataset,
            'seed': self.random_seed,
            'solver_name': self.solver_name,
            'energy': energy,
            'time': time_taken,
            'local_search': self.use_local_search,
            'local_search_iterations': local_iterations,
            'alpha': alpha,
            'beta': beta,
            'mode': self.mode,
            'precision': self.precision,
            'eigsh_maxiter': self.eigsh_maxiter,
            'eigsh_iterations': getattr(self, '_eigsh_iterations', None),
            'eigsh_converged': getattr(self, '_eigsh_converged', None),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if csv_file.exists():
            df = pd.read_csv(csv_file, float_precision='round_trip')
        else:
            df = pd.DataFrame()
        
        if not df.empty:
            duplicate_mask = (
                (df['instance_name'] == result_data['instance_name']) &
                (df['dataset'] == result_data['dataset']) &
                (df['seed'] == result_data['seed']) &
                (df['solver_name'] == result_data['solver_name'])
            )
            
            if duplicate_mask.any():
                df.loc[duplicate_mask, list(result_data.keys())] = list(result_data.values())
            else:
                df = pd.concat([df, pd.DataFrame([result_data])], ignore_index=True)
        else:
            df = pd.DataFrame([result_data])
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Results saved to: {csv_file}")
        




# ============================================================
# CUDA kernel: one warp per selected node and candidate; flips the
# spin if that lowers the energy. Updates are in place, in parallel.
# ============================================================
_update_spins_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_spins_kernel(
    const int n_selected,
    const int n,
    const int k,
    const int* selected_nodes,
    const int* indptr,
    const int* indices,
    const float* data,
    signed char* spins,
    int* improvement_counts)
{
    int lane = threadIdx.x;
    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;  // node index in selected_nodes
    int s = blockIdx.y;                                   // solution id

    if (warp_id >= n_selected || s >= k) return;

    int node = selected_nodes[warp_id];
    int row_start = indptr[node];
    int row_end   = indptr[node+1];

    float local_sum = 0.0f;

    // Partial sum across warp
    for (int idx = row_start + lane; idx < row_end; idx += 32) {
        int j = indices[idx];
        long long idx_n = (long long)s * n + j;  // 64-bit: k*n can exceed 2^31
        local_sum += data[idx] * (float)spins[idx_n];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Lane 0 decides whether to flip
    if (lane == 0) {
        long long idx_spin = (long long)s * n + node;
        signed char si = spins[idx_spin];
        float delta_E = 2.0f * (float)si * local_sum;
        if (delta_E < 0.0f) {
            spins[idx_spin] = -si;
            atomicAdd(&improvement_counts[s], 1);
        }
    }
}
''', "update_spins_kernel")



# ============================================================
# GPU final sweep
# ============================================================
def gpu_final_sweep(J_csr,
                    spins_flat: cp.ndarray,
                    n: int,
                    k: int,
                    node_fraction: float = 0.5,
                    max_sub_iters: int = 5,
                    num_rounds: int = 3):
    """
    Final sweep of the local search, on k candidates stored one after the other in spins_flat.

    Each of num_rounds rounds takes a new random subset of node_fraction * n nodes and runs up
    to max_sub_iters passes on it, stopping early when no spin flips. Returns spins_flat.
    """
    indptr  = J_csr.indptr.astype(cp.int32)
    indices = J_csr.indices.astype(cp.int32)
    data    = J_csr.data.astype(cp.float32)

    warp_size = 32
    warps_per_block = 4
    threads_per_block = (warp_size, warps_per_block)

    for round_idx in range(num_rounds):
        num_to_select = max(1, int(n * node_fraction))
        node_indices = cp.arange(n, dtype=cp.int32)
        node_indices = cp.random.permutation(node_indices)
        selected_nodes = node_indices[:num_to_select]

        num_warps_x = (num_to_select + warps_per_block - 1) // warps_per_block
        grid = (num_warps_x, k)
        improvement_counts = cp.zeros(k, dtype=cp.int32)

        for sub_iter in range(max_sub_iters):
            improvement_counts.fill(0)
            
            _update_spins_kernel(
                grid, threads_per_block,
                (num_to_select, n, k,
                 selected_nodes, indptr, indices, data,
                 spins_flat, improvement_counts)
            )
            
            if not cp.any(improvement_counts > 0):
                break

    return spins_flat



def _eigsh_finite(S, k, tol, maxiter):
    """
    The k smallest eigenpairs of S with eigsh, retried once from a new random start if the
    result is not finite (a Lanczos breakdown, more likely in float32). A second failure raises
    NumericalFailure, so a NaN vector is never rounded silently.
    """
    from utils.failures import NumericalFailure
    for attempt in (1, 2):
        evals, evecs_y, iter_count = eigsh(S, k=k, which='SA', tol=tol, maxiter=maxiter)
        if bool(cp.isfinite(evals).all()) and bool(cp.isfinite(evecs_y).all()):
            return evals, evecs_y, iter_count
        print(f"Warning: eigsh returned a non-finite eigenpair (try {attempt} of 2)")
    raise NumericalFailure("eigsh returned a non-finite eigenpair twice")


# ============================================================
# Local search
# ============================================================

def flip_refinement(J_csr, spins_list, max_iterations=1000,
                    final_sweep_fraction=1.0, final_sweep_iters=25,
                    num_rounds=25, subset_fraction=0.9):
    """
    GPU local search of a batch of candidates. Returns (n x k int8 spins, number of passes).

    Args:
        J_csr: couplings (CSR matrix on the GPU)
        spins_list: the k candidates (+-1 vectors); they are not modified
        max_iterations: maximum number of passes; each pass updates a new random subset of
            nodes, and the search stops once every candidate has had a pass without a flip
        final_sweep_fraction: share of the nodes in each round of the final sweep
        final_sweep_iters: maximum passes per round of the final sweep
        num_rounds: number of rounds of the final sweep
        subset_fraction: share of the nodes updated in each pass
    """

    Spins = cp.stack([sp.astype(cp.int8).copy() for sp in spins_list], axis=1)
    n, k = Spins.shape
    spins_flat = Spins.T.reshape(-1).copy()

    indptr  = J_csr.indptr.astype(cp.int32)
    indices = J_csr.indices.astype(cp.int32)
    data    = J_csr.data.astype(cp.float32)

    improvement_counts = cp.zeros(k, dtype=cp.int32)
    active_mask = cp.ones(k, dtype=cp.bool_)
    max_iters_used = 0

    warp_size = 32
    warps_per_block = 4
    threads_per_block = (warp_size, warps_per_block)

    for it in range(max_iterations):
        max_iters_used += 1
        improvement_counts.fill(0)

        num_selected = max(1, int(subset_fraction * n))
        selected_nodes = cp.random.permutation(n).astype(cp.int32)[:num_selected]

        num_warps_x = (num_selected + warps_per_block - 1) // warps_per_block
        grid = (num_warps_x, k)

        _update_spins_kernel(
            grid, threads_per_block,
            (num_selected, n, k,
             selected_nodes, indptr, indices, data,
             spins_flat, improvement_counts)
        )

        no_improvement = (improvement_counts == 0)
        active_mask &= ~no_improvement

        if not cp.any(active_mask):
            break

    spins_flat = gpu_final_sweep(
        J_csr, spins_flat, n, k,
        node_fraction=final_sweep_fraction,
        max_sub_iters=final_sweep_iters,
        num_rounds=num_rounds
    )

    Spins_final = spins_flat.reshape(k, n).T.astype(cp.int8)

    return Spins_final, max_iters_used



