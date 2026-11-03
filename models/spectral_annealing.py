"""
SpecAnn (Spectral Annealing, Algorithm 1 of the paper) on the GPU: for each alpha of a schedule,
the warm-started leading eigenvector of a normalized coupling matrix is rounded to spins; the
distinct candidates are then refined with a batched GPU local search.
"""
import contextlib
import sys
import numpy as np
import time
import warnings
import pandas as pd
from pathlib import Path
from datetime import datetime

import cupy as cp
import scipy.sparse as sp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.linalg as cpx_splinalg

from utils.eigsh import eigsh
from utils.failures import NumericalFailure, is_out_of_memory
from utils.predictors import diagonal_predictor, secant_predictor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



class NAlphaOperator(cpx_splinalg.LinearOperator):
    """The operator N_alpha v = D^{-1/2} J D^{-1/2} v, with D = diag(D_vec)."""
    def __init__(self, J_csr, D_vec, dtype):
        self.J_csr = J_csr
        self.D_vec = D_vec
        self.D_inv_sqrt = cp.reciprocal(cp.sqrt(cp.maximum(D_vec, cp.array(1e-12, dtype=dtype))))
        n = D_vec.size
        super().__init__(dtype=dtype, shape=(n, n))

    def _matvec(self, v):
        z = self.D_inv_sqrt * v
        w = self.J_csr.dot(z)
        return self.D_inv_sqrt * w


class SRhoOperator(cpx_splinalg.LinearOperator):
    """The shifted operator S_rho v = rho v - N_alpha v: its smallest eigenvalue gives the largest of N_alpha."""
    def __init__(self, N_op, rho):
        self.N_op = N_op
        self.rho = rho
        n = N_op.shape[0]
        super().__init__(dtype=N_op.dtype, shape=(n, n))

    def _matvec(self, v):
        return self.rho * v - self.N_op(v)


def compute_shift_rho(lambda1_prev, gershgorin_bound, eps=1e-3):
    """Shift rho = (1 + eps) * max(lambda1_prev, gershgorin_bound)."""
    # A non-finite lambda1_prev must never reach rho: Python's max(nan, g) is nan.
    if lambda1_prev is None or not np.isfinite(lambda1_prev) or lambda1_prev <= 0:
        base = gershgorin_bound
    else:
        base = max(lambda1_prev, gershgorin_bound)

    rho = (1.0 + eps) * base
    if rho <= 0:
        rho = (1.0 + eps) * max(gershgorin_bound, 1.0)
    return rho


class EigensolverBreakdown(RuntimeError):
    """eigsh produced an unusable result (Lanczos breakdown, non-finite or impossible eigenpair)."""


def _is_numerical(exc: BaseException) -> bool:
    """True for a numerical failure (Algorithm 1 skips that alpha), not a bug or a broken environment."""
    return (isinstance(exc, (EigensolverBreakdown, ArithmeticError, np.linalg.LinAlgError))
            or type(exc).__name__ in ('LinAlgError', 'CUSOLVERError'))


# Names of the dense solver of the Lanczos tridiagonal inside CuPy's eigsh: a GPU version (from a
# patched CuPy) and stock CuPy's host version. Only the names present are wrapped.
_RITZ_SOLVERS = ('_eigsh_solve_ritz_gpu', '_eigsh_solve_ritz')


@contextlib.contextmanager
def _checked_ritz_matrix(bound):
    """
    Inside this context, eigsh raises EigensolverBreakdown on a broken Lanczos tridiagonal.

    Why: CuPy's eigsh has no breakdown check. On graphs with few distinct eigenvalues, in
    float32, the Ritz matrix can get NaN or huge entries; cuSOLVER then fails and leaves the
    CUDA context unusable, so the cold retry of Algorithm 1 cannot run. In exact arithmetic
    every entry is at most ||S|| <= bound, so we raise on a non-finite entry or one above
    10 x bound. Healthy calls are unchanged.
    """
    module = sys.modules[cpx_splinalg.eigsh.__module__]

    def checked(solve_ritz):
        def wrapper(alpha, beta, beta_k, *args, **kwargs):
            parts = [cp.ravel(cp.asarray(alpha)), cp.ravel(cp.asarray(beta))]
            if beta_k is not None:
                parts.append(cp.ravel(cp.asarray(beta_k)))
            entries = cp.concatenate(parts)
            if not bool(cp.isfinite(entries).all()):
                raise EigensolverBreakdown("Lanczos breakdown: non-finite Ritz matrix")
            largest = float(cp.max(cp.abs(entries)))
            if largest > 10.0 * bound:
                raise EigensolverBreakdown(
                    f"Lanczos breakdown: Ritz matrix entry {largest:.3g} exceeds 10 x ||S|| bound {bound:.3g}")
            return solve_ritz(alpha, beta, beta_k, *args, **kwargs)
        return wrapper

    originals = {name: getattr(module, name) for name in _RITZ_SOLVERS if hasattr(module, name)}
    for name, solve_ritz in originals.items():
        setattr(module, name, checked(solve_ritz))
    try:
        yield
    finally:
        for name, solve_ritz in originals.items():
            setattr(module, name, solve_ritz)


# ===================
# SPECTRAL ANNEALING
# ===================

PRED_MODES = ("diagonal", "secant", "prev")
LS_POLICIES = ("default", "fixed_guard", "adaptive_guard", "independent_set", "sequential_oracle")
SCHEDULES = ("sqrt", "linear")


class SpectralAnnealing:
    """
    SpecAnn solver.

    Parameters:
        instance_name: instance file name; names the results CSVs.
        dataset: dataset name; results go to results/<dataset>/<run_name>/.
        random_seed: seed of CuPy's random generator (None: not seeded).
        mode: alpha schedule mode; only 0 is supported.
        beta: not used.
        alpha: must be None (the schedule sets alpha).
        use_local_search: refine the distinct rounded candidates with the GPU local search.
        max_local_iterations: maximum number of local search passes.
        eigsh_tol: tolerance of eigsh.
        eigsh_maxiter: maximum number of eigsh iterations per alpha.
        run_name: name of the run and of its results folder (default "spectral_annealing_pred").
        disable_warm_start: start eigsh from a random vector at every alpha.
        precision: 32 or 64 (bits of the GPU floats).
        pred_mode: warm-start predictor: "diagonal" (Algorithm 1), "secant" or "prev" (the previous
            eigenvector as is, with no residual check).
        pred_tol: residual tolerance; a predicted vector with a larger residual is replaced by the
            previous eigenvector.
        shift_eps: shift margin: rho = (1 + shift_eps) * max(previous lambda1, Gershgorin bound).
        schedule: "sqrt" (alpha = sqrt(t), Algorithm 1) or "linear" (alpha = t).
        num_alphas: number K >= 2 of alpha values.
        schedule_interval: (start, end) of t; K equally spaced values of t are used. For "sqrt",
            [0, 2] gives alpha in [0, sqrt(2)].
        eigsh_ncv: Lanczos vectors per restart (None: CuPy's default, min(33, n - 1)).
        fp32_spmv: keep J in float32 inside eigsh (default: True at precision 32; needs precision
            32). False keeps float32 couplings, but the scaled J is float64, so eigsh works in FP64.
        ls_policy: "default" (Algorithm 2: parallel in-place updates and a final sweep) or a guarded
            policy of models/local_search_guards: "fixed_guard", "adaptive_guard",
            "independent_set" or "sequential_oracle".
        ls_batch: refine all candidates in one batch (True) or one per kernel launch (False;
            the SpecAnn[-FastL] ablation; only with ls_policy "default").
        **kwargs: other keys; printed and ignored.

    With fp32_spmv, the reported energy is recomputed in float64 from the input J.
    """

    def __init__(self, instance_name: str, dataset: str, random_seed: int = None,
                 mode: int = 0, beta: float = None, alpha: float = None,
                 use_local_search: bool = True, max_local_iterations: int = 1000,
                 eigsh_tol: float = 1e-6, eigsh_maxiter: int = 1000, run_name: str = None,
                 disable_warm_start: bool = False, precision: int = 32,
                 pred_mode: str = "diagonal", pred_tol: float = 1e-3, shift_eps: float = 1e-3,
                 schedule: str = "sqrt", num_alphas: int = 128, schedule_interval=(0.0, 1.0),
                 eigsh_ncv: int = None, fp32_spmv: bool = None, ls_policy: str = "default",
                 ls_batch: bool = True, **kwargs):
        self.instance_name = instance_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.mode = mode
        self.beta = beta  # Not used, kept for compatibility
        self.alpha = alpha
        self.use_local_search = use_local_search
        self.max_local_iterations = max_local_iterations
        self.eigsh_tol = float(eigsh_tol)
        self.eigsh_maxiter = int(eigsh_maxiter)
        self.eigsh_ncv = None if eigsh_ncv is None else int(eigsh_ncv)
        self.disable_warm_start = disable_warm_start

        if pred_mode not in PRED_MODES:
            raise ValueError(f"pred_mode must be one of {PRED_MODES}, got {pred_mode!r}")
        self.pred_mode = pred_mode
        self.pred_tol = float(pred_tol)  # Residual tolerance for predictor validation
        self.shift_eps = float(shift_eps)  # Epsilon for shift calculation

        # Alpha schedule (mode 0); the defaults are the paper's schedule.
        t_min, t_max = (float(v) for v in schedule_interval)
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
        if int(num_alphas) < 2 or not 0.0 <= t_min < t_max:
            raise ValueError(f"need num_alphas >= 2 and 0 <= start < end, got {num_alphas}, {schedule_interval}")
        self.schedule = schedule
        self.num_alphas = int(num_alphas)
        self.schedule_interval = (t_min, t_max)
        
        if precision == 32:
            self.dtype = cp.float32
        elif precision == 64:
            self.dtype = cp.float64
        else:
            raise ValueError(f"precision must be 32 or 64, got {precision}")
        self.precision = precision
        # fp32_spmv: FP32 end to end at precision 32. Without it, J / c is a true division,
        # which CuPy (like SciPy) returns in float64, so eigsh multiplies in FP64.
        self.fp32_spmv = (precision == 32) if fp32_spmv is None else bool(fp32_spmv)
        if ls_policy not in LS_POLICIES:
            raise ValueError(f"ls_policy must be one of {LS_POLICIES}, got {ls_policy!r}")
        self.ls_policy = ls_policy
        # ls_batch = False: same kernel, parameters, deduplication and final sweep, one
        # candidate per launch.
        self.ls_batch = bool(ls_batch)
        if not self.ls_batch and ls_policy != "default":
            raise ValueError("ls_batch = False is only defined for ls_policy 'default'")
        if self.fp32_spmv and precision != 32:
            raise ValueError("fp32_spmv needs precision 32")
        
        self.solver_name = run_name if run_name else "spectral_annealing_pred"
        
        if random_seed is not None:
            cp.random.seed(random_seed)
            self.random_seed = random_seed
            
        print(f"Initialized {self.solver_name} solver. Results will be saved under '{self.solver_name}'.")
        
        print("Parameters:")
        print(f"  instance_name: {self.instance_name}")
        print(f"  dataset: {self.dataset}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  mode: {self.mode} (schedule {self.schedule} of {self.num_alphas} values of "
              f"[{self.schedule_interval[0]}, {self.schedule_interval[1]}])")
        print(f"  beta: {self.beta} (not used)")
        print(f"  alpha: {self.alpha}")
        print(f"  use_local_search: {self.use_local_search}")
        print(f"  max_local_iterations: {self.max_local_iterations}")
        print(f"  eigsh_tol: {self.eigsh_tol}")
        print(f"  eigsh_maxiter: {self.eigsh_maxiter}")
        print(f"  eigsh_ncv: {self.eigsh_ncv if self.eigsh_ncv is not None else 'CuPy default'}")
        print(f"  disable_warm_start: {self.disable_warm_start}")
        print(f"  precision: {self.precision} bits ({self.dtype}), fp32_spmv: {self.fp32_spmv}")
        print(f"  pred_mode: {self.pred_mode}")
        print(f"  pred_tol: {self.pred_tol}")
        print(f"  shift_eps: {self.shift_eps}")
        print(f"  run_name: {run_name}")
        if kwargs:
            print(f"  additional kwargs: {kwargs}")
        print("-" * 50)

    def solve(self, J: np.ndarray):
        """
        Run SpecAnn on the couplings J and write the results CSVs.
        Returns (best energy, best spins, time in seconds).
        """
        start_time = time.time()


        J_csr = cpx_sparse.csr_matrix(J.astype(self.dtype), dtype=self.dtype)

        J_abs = J_csr.copy()
        J_abs.data = cp.abs(J_abs.data)

        deg_abs_original = J_abs @ cp.ones(J_csr.shape[1], dtype=self.dtype)
        del J_abs  # only needed for the degrees; frees a full copy of J

        max_deg_abs = cp.max(deg_abs_original)

        self._normalization_factor = float(max_deg_abs)

        if self.fp32_spmv and self.dtype == cp.float32:
            # scaled in float32, sharing indices/indptr with the unscaled matrix (no copy)
            J_csr = cpx_sparse.csr_matrix((J_csr.data / max_deg_abs, J_csr.indices, J_csr.indptr),
                                          shape=J_csr.shape)
        else:
            J_csr = J_csr / max_deg_abs
        deg_abs_normalized = deg_abs_original / max_deg_abs

        if self.alpha is None:
            if self.random_seed is not None:
                cp.random.seed(self.random_seed)
            alpha_list = self._get_alpha_values(self.mode)
            
            best_energy, best_spins, best_alpha, best_beta, best_local_iterations = float("inf"), None, None, None, 0
            alpha_results = []
            
            unique_solutions = {}
            alpha_to_solution_hash = {}
            candidate_index = {}  # spins hash -> order of first appearance (per-alpha CSV `candidate`)
            
            y_last = None
            y_prev = None
            lambda1_prev = None
            alpha_prev = None
            self._alphas_skipped = 0
            self._alphas_retried = 0
            self._alpha_errors = 0            # alphas skipped for a non-numerical exception
            self._last_alpha_error = None
            
            for alpha_idx, alpha in enumerate(alpha_list):
                try:
                    alpha_start_time = time.time()
                    self.alpha = alpha
                    
                    D_vec = cp.power(deg_abs_normalized, float(alpha))

                    evals, evecs, evecs_y, iter_count, lambda1 = self._solve_eigenvalue_problem_pred(
                        J_csr, D_vec, alpha_idx, alpha, alpha_prev,
                        y_last, y_prev, lambda1_prev, deg_abs_normalized
                    )
                    eigsh_retry = self._last_eigsh_retry
                    eigsh_converged = self._last_eigsh_converged
                    self._alphas_retried += int(eigsh_retry)
                    
                    degree_scale = float(cp.sum(D_vec))
                    normalization_factor = float(self._normalization_factor)
                    bound_rescaled = (-0.5 * lambda1 * degree_scale * normalization_factor
                                      if eigsh_converged else float('nan'))

                    if not self.disable_warm_start:
                        y_prev = y_last

                        y_new = evecs_y[:, 0]

                        y_new_norm = cp.linalg.norm(y_new)
                        if y_new_norm > 0:
                            y_new = y_new / y_new_norm

                        y_last = y_new

                        lambda1_prev = lambda1
                        alpha_prev = alpha
                    
                    spins = cp.where(evecs[:, 0] >= 0, cp.array(1.0, dtype=self.dtype), cp.array(-1.0, dtype=self.dtype))

                    # A candidate identical to an earlier alpha's has the same energy: reuse it
                    # instead of another SpMV.
                    spins_hash = self._compute_spin_signature(spins) if self.use_local_search else None
                    if spins_hash is not None and spins_hash in unique_solutions:
                        raw_energy = alpha_results[unique_solutions[spins_hash][1][0]]['raw_energy']
                    else:
                        raw_energy = self.compute_energy(spins, J_csr)
                    if np.isinf(raw_energy) or np.isnan(raw_energy):
                        print(f"Warning: Invalid energy {raw_energy} for alpha={alpha}, skipping this iteration")
                        self._alphas_skipped += 1
                        continue
                    energy = raw_energy  # Will be updated after local search if applied
                    
                    local_iterations = 0
                    
                    if self.use_local_search:
                        if spins_hash not in unique_solutions:
                            # int8 instead of a float copy: 4x less memory per stored candidate
                            unique_solutions[spins_hash] = (spins.astype(cp.int8), [len(alpha_results)])
                            candidate_index[spins_hash] = len(candidate_index)
                        else:
                            unique_solutions[spins_hash][1].append(len(alpha_results))
                        
                        alpha_to_solution_hash[len(alpha_results)] = spins_hash
                    
                    alpha_time = time.time() - alpha_start_time
                    
                    alpha_results.append({
                        'instance_name': self.instance_name,
                        'dataset': self.dataset,
                        'seed': self.random_seed,
                        'solver_name': self.solver_name,
                        'alpha': alpha,
                        'beta': None,  # Not used
                        'energy': energy,
                        'raw_energy': raw_energy,
                        'time': alpha_time,
                        'mode': self.mode,
                        'use_local_search': self.use_local_search,
                        'local_search_iterations': local_iterations,
                        'eigsh_iterations': iter_count,
                        'eigsh_retry': eigsh_retry,
                        'eigsh_converged': eigsh_converged,
                        'warm_start': not self.disable_warm_start,
                        'alpha_index': alpha_idx,
                        'precision': self.precision,
                        'fp32_spmv': self.fp32_spmv,
                        'lambda1': lambda1,
                        'degree_scale': degree_scale,
                        'energy_bound': bound_rescaled,
                        'pred_mode': self.pred_mode,
                        'pred_residual': self._last_pred_residual,
                        'pred_accepted': self._last_pred_accepted,
                        # distinct rounded candidate this alpha produced (local search on only)
                        'candidate': candidate_index.get(spins_hash),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    if energy < best_energy and not (np.isinf(energy) or np.isnan(energy)):
                        best_energy, best_spins, best_alpha, best_beta = energy, spins.copy(), alpha, None
                        best_local_iterations = local_iterations
                        
                except Exception as e:
                    # Running out of memory depends on the instance size, not on this alpha: the
                    # whole attempt fails as out_of_memory instead of skipping alphas.
                    if is_out_of_memory(e):
                        raise
                    print(f"Alpha {alpha} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self._alphas_skipped += 1
                    if not _is_numerical(e):
                        self._alpha_errors += 1
                        self._last_alpha_error = e.with_traceback(None)   # drop the traceback so its frames (and GPU arrays) are freed
                    continue

            sweep_end_time = time.time()
            self._num_candidates = len(unique_solutions) if self.use_local_search else None
            if self.use_local_search and unique_solutions:
                print(f"Applying local search to {len(unique_solutions)} unique solutions (out of {len(alpha_list)} total)")
                solution_hashes = list(unique_solutions.keys())
                spins_list = [spins for spins, _ in unique_solutions.values()]
                
                improved_solutions = {}
                
                if len(spins_list) > 0:
                    # Kernel matrix in the LS precision, sharing indices/indptr with J_csr
                    # (csr.astype would copy them too); no copy at all when dtypes match.
                    J_csr_precision = cpx_sparse.csr_matrix(
                        (J_csr.data.astype(self.dtype, copy=False), J_csr.indices, J_csr.indptr),
                        shape=J_csr.shape)
                    
                    self._ls_matrix_cpu = None
                    if self.ls_policy != "default":
                        from models.local_search_guards.policies import NEEDS_HOST_MATRIX
                        if self.ls_policy in NEEDS_HOST_MATRIX:
                            # this policy also needs the normalized couplings on the host (graph coloring)
                            self._ls_matrix_cpu = (sp.csr_matrix(J).astype(np.float32)
                                                   / self._normalization_factor)
                    refined_spins_batch, local_iterations = self._refine_spins(
                        J_csr_precision, spins_list)
                    
                    # Energies of all refined candidates in a few sparse x dense products (one
                    # pass over J per chunk instead of one SpMV each), and no float copy per
                    # candidate: only the winner is converted below.
                    refined_energies = self._batch_energies(refined_spins_batch, J_csr)
                    for i, solution_hash in enumerate(solution_hashes):
                        _, alpha_indices = unique_solutions[solution_hash]
                        # Same computation as during the sweep (same spins, same J_csr): reuse it.
                        original_energy = alpha_results[alpha_indices[0]]['raw_energy']
                        if refined_energies[i] > original_energy:
                            improved_solutions[solution_hash] = (('original', solution_hash), original_energy, local_iterations)
                        else:
                            improved_solutions[solution_hash] = (('refined', i), refined_energies[i], local_iterations)

                best_source = None
                for idx_a, result in enumerate(alpha_results):
                    if idx_a in alpha_to_solution_hash:
                        solution_hash = alpha_to_solution_hash[idx_a]
                        if solution_hash in improved_solutions:
                            source, improved_energy, local_iterations = improved_solutions[solution_hash]
                            result["energy"] = float(improved_energy)
                            result["local_search_iterations"] = local_iterations
                            
                            if result["energy"] < best_energy:
                                best_energy = result["energy"]
                                best_source = source
                                best_alpha = result["alpha"]
                                best_beta = result["beta"]
                                best_local_iterations = local_iterations

                if best_source is not None:
                    kind, key = best_source
                    best_spins = (refined_spins_batch[:, key] if kind == 'refined'
                                  else unique_solutions[key][0]).astype(self.dtype)

            total_time = time.time() - start_time
            if self.fp32_spmv and best_spins is not None:
                # The GPU energies use the float32 couplings J/c; the reported one is computed in
                # float64 from the input J. After the timer: this is reporting, not search.
                s64 = cp.asnumpy(best_spins).astype(np.float64)
                best_energy = float(-0.5 * s64 @ (J @ s64))
            # Phase split of `time`: the alpha sweep (eigensolves, rounding, deduplication) and
            # what follows it (local search, energies of the refined candidates, winner).
            self._sweep_time = sweep_end_time - start_time
            self._post_sweep_time = total_time - self._sweep_time
            # GPU memory, read after the timer: the size of CuPy's memory pool, which keeps freed
            # blocks, so it is the most this process has reserved so far. When instances run in
            # increasing size (main.py), it bounds this run's peak use (CUDA context not counted).
            self._gpu_pool_bytes = int(cp.get_default_memory_pool().total_bytes())
            
            if best_energy == float("inf") or best_spins is None:
                if alpha_results:
                    for result in alpha_results:
                        if result.get('energy') is not None and not (np.isinf(result['energy']) or np.isnan(result['energy'])):
                            best_energy = result['energy']
                            best_alpha = result['alpha']
                            best_beta = result.get('beta')
                            break
                else:
                    print("Warning: No valid results found for any alpha value!")
            
            if best_alpha is None:
                # Every alpha failed: no solution. If all failures were numerical (e.g. eigsh breaks
                # down at every alpha on 2-3 spin instances), raise NumericalFailure; otherwise (a bug,
                # a broken environment) raise a RuntimeError caused by the last non-numerical error.
                # The per-alpha rows are still written.
                self._store_timing_results(alpha_results)
                message = f"no alpha produced a solution ({self._alphas_skipped} of {len(alpha_list)} skipped)"
                if self._alpha_errors:
                    last = self._last_alpha_error
                    raise RuntimeError(f"{message}; {self._alpha_errors} of them raised a non-numerical error, "
                                       f"the last {type(last).__name__}: {last}") from last
                raise NumericalFailure(message)
            self._store_general_results(best_energy, best_alpha, best_beta, total_time, best_local_iterations)

            self._store_timing_results(alpha_results)
            # best_spins is returned for --save-spins.
            return best_energy, best_spins, total_time

        else:
            raise ValueError("Alpha is not set")

    def _refine_spins(self, J_csr, spins_list):
        """Local search of the distinct candidates. Returns (n x k int8 spins, iteration count)."""
        if self.ls_policy == "default" and not self.ls_batch:
            columns, iterations = [], 0
            for spins in spins_list:
                refined, used = flip_refinement(J_csr, [spins], max_iterations=self.max_local_iterations,
                                                dtype=self.dtype)
                columns.append(refined[:, 0])
                iterations = max(iterations, used)
            return cp.stack(columns, axis=1), iterations
        if self.ls_policy == "default":
            return flip_refinement(J_csr, spins_list, max_iterations=self.max_local_iterations,
                                   dtype=self.dtype)
        from models.local_search_guards.policies import run_policy
        refined, iterations, self._last_ls_summary = run_policy(
            self.ls_policy, J_csr, spins_list, max_iterations=self.max_local_iterations,
            dtype=self.dtype, seed=self.random_seed, matrix_cpu=self._ls_matrix_cpu)
        return refined, iterations

    def _get_alpha_values(self, mode: int) -> list:
        """The alpha values of the schedule (mode 0 only)."""
        if mode == 0:
            t = cp.linspace(*self.schedule_interval, self.num_alphas)
            return [float(x) for x in (cp.sqrt(t) if self.schedule == "sqrt" else t)]
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _compute_spin_signature(self, spins: cp.ndarray) -> bytes:
        """Packed signs of a spin vector, used as a key to find duplicate candidates."""
        bits = (spins > 0).astype(cp.uint8)
        packed = cp.packbits(bits)
        return packed.tobytes()

    def _calculate_beta(self, deg_abs: "cp.ndarray") -> float:
        """Not used."""
        beta = float((1.0 + 1e-8) * cp.max(deg_abs ** (1.0 - self.alpha)))
        return beta

    def _solve_eigenvalue_problem_pred(self, J_csr, D_vec, alpha_idx, alpha, alpha_prev,
                                      y_last, y_prev, lambda1_prev, deg_abs_normalized):
        """
        Leading eigenpair of N_alpha for one alpha: predict a start vector, then run eigsh on
        S_rho (with one cold retry). Returns ([lambda1], x = D^{-1/2} y, y, iterations, lambda1).
        """

        n = D_vec.size

        N_op = NAlphaOperator(J_csr, D_vec, self.dtype)

        gershgorin_bound = float(cp.max(cp.power(deg_abs_normalized, 1.0 - alpha)))

        rho_k = compute_shift_rho(lambda1_prev, gershgorin_bound, eps=self.shift_eps)

        S_op = SRhoOperator(N_op, rho_k)

        # Predictor diagnostics for the per-alpha CSV: residual of the predicted vector and
        # whether eigsh started from it (NaN / None when no predictor ran).
        self._last_pred_residual = float('nan')
        self._last_pred_accepted = None
        if alpha_idx == 0 or self.disable_warm_start or y_last is None:
            y_init = cp.random.randn(n).astype(self.dtype)
            y_init = y_init / cp.linalg.norm(y_init)
        elif self.pred_mode == "prev":
            # Warm start from the previous eigenvector as is: no predictor and no residual
            # check, so one SpMV less per alpha.
            y_init = y_last
        else:
            delta_alpha = alpha - alpha_prev if alpha_prev is not None else 0.0
            if self.pred_mode == "diagonal":
                y_init = diagonal_predictor(y_last, deg_abs_normalized, delta_alpha, self.dtype)
            elif self.pred_mode == "secant":
                y_init = secant_predictor(y_last, y_prev, self.dtype)
            else:
                raise ValueError(f"Unknown predictor mode: {self.pred_mode}")

            v = N_op(y_init)
            lambda_est = float(cp.dot(y_init, v))
            r = float(cp.linalg.norm(v - lambda_est * y_init))
            self._last_pred_residual = r
            self._last_pred_accepted = not r > self.pred_tol

            if r > self.pred_tol:
                y_init = y_last

        # ||rho*I - N|| <= rho + max|lambda(N)| <= rho + g bounds every Lanczos coefficient.
        spectrum_bound = rho_k + gershgorin_bound
        self._last_eigsh_retry = False
        try:
            # CuPy's eigsh uses v0 as its Lanczos work vector (u = v0) and overwrites it.
            # When the predictor is rejected, y_init *is* y_last, which is still needed
            # below for the sign alignment and, as y_prev, by the secant predictor.
            evals, evecs_y, iter_count = self._eigsh_checked(S_op, y_init.copy(), spectrum_bound)
            self._check_eigenpair(evals, evecs_y, rho_k, gershgorin_bound)
        except Exception as e:
            # Algorithm 1: a failed attempt (an exception, a Lanczos breakdown, or a non-finite
            # or impossible eigenpair) is retried cold; if the retry fails too, the exception
            # reaches the alpha loop, which skips this alpha.
            print(f"Warning: eigsh failed for alpha={alpha}, rho_k={rho_k}, error: {e}")
            self._last_eigsh_retry = True
            try:
                evals, evecs_y, iter_count = self._eigsh_checked(S_op, None, spectrum_bound)
                self._check_eigenpair(evals, evecs_y, rho_k, gershgorin_bound)
            except Exception as e2:
                print(f"Error: eigsh failed even without initial vector: {e2}")
                raise

        # eigsh stops silently at maxiter. Below that it stops only once the Lanczos residual
        # estimate is <= tol, so the pair has converged; at the limit, or when the iteration
        # count is unknown (stock CuPy), check the actual residual. A non-converged Ritz value
        # can underestimate lambda_1, so the energy bound of that alpha is not certified.
        self._last_eigsh_converged = True
        if iter_count is None or iter_count >= self.eigsh_maxiter:
            y = evecs_y[:, 0]
            residual = float(cp.linalg.norm(S_op(y) - float(evals[0]) * y) / cp.linalg.norm(y))
            self._last_eigsh_converged = residual <= self.eigsh_tol
            if not self._last_eigsh_converged:
                print(f"Warning: eigsh did not converge for alpha={alpha}: residual "
                      f"{residual:.3g} > tol {self.eigsh_tol:g}; energy bound not certified")


        mu_min = float(evals[0])
        lambda1 = rho_k - mu_min

        if y_last is not None:
            dot_before = float(cp.dot(evecs_y[:, 0], y_last))
            if dot_before < 0:
                evecs_y[:, 0] = -evecs_y[:, 0]

        norm_y = cp.linalg.norm(evecs_y[:, 0])
        if norm_y > 0:
            evecs_y[:, 0] /= norm_y

        # x = D^{-1/2} y, reusing the operator's D^{-1/2}
        X = evecs_y * N_op.D_inv_sqrt[:, cp.newaxis]
        
        if X.ndim == 1:
            X = X[:, cp.newaxis]
        
        evals_return = cp.array([lambda1], dtype=self.dtype)
        return evals_return, X, evecs_y, iter_count, lambda1

    def _eigsh_checked(self, S_op, v0, spectrum_bound):
        """Smallest eigenpair of S_op with eigsh, from start vector v0, with the Ritz-matrix check."""
        with _checked_ritz_matrix(spectrum_bound):
            return eigsh(
                S_op, k=1, which="SA",
                tol=self.eigsh_tol, maxiter=self.eigsh_maxiter, v0=v0, ncv=self.eigsh_ncv
            )

    def _check_eigenpair(self, evals, evecs_y, rho_k, gershgorin_bound):
        """Reject an eigsh result that cannot be the principal eigenpair."""
        nu = float(evals[0])
        if not np.isfinite(nu):
            raise EigensolverBreakdown(f"non-finite eigenvalue {nu}")
        if not bool(cp.isfinite(evecs_y).all()):
            raise EigensolverBreakdown("non-finite eigenvector")
        # lambda_1 of the scaled N_alpha is at most g = max d^(1-alpha) (Gershgorin), and
        # rho >= (1 + eps) g, so a value beyond g (1 + eps/2) is spurious.
        lambda1 = rho_k - nu
        if lambda1 > gershgorin_bound * (1.0 + max(0.5 * self.shift_eps, 1e-5)):
            raise EigensolverBreakdown(
                f"spurious eigenvalue {lambda1:.6g} above the Gershgorin bound {gershgorin_bound:.6g}")

    def _batch_energies(self, spins_matrix, J_csr, chunk: int = 8) -> list:
        """
        E = -0.5 s^T J s (original units) for every column of an n x k +-1 matrix.
        J is read once per chunk of columns; the chunk bounds the extra memory to 2 x n x chunk floats.
        """
        energies = []
        for start in range(0, spins_matrix.shape[1], chunk):
            if self.fp32_spmv:
                energies.extend(float(x) for x in cp.asnumpy(
                    fp32_energies(J_csr, spins_matrix[:, start:start + chunk], self._normalization_factor)))
                continue
            S = spins_matrix[:, start:start + chunk].astype(J_csr.dtype)
            e = -0.5 * cp.sum(S * (J_csr @ S), axis=0) * self._normalization_factor
            energies.extend(float(x) for x in cp.asnumpy(e))
        return energies

    def compute_energy(self, spins: "cp.ndarray", J_csr) -> float:
        """Ising energy -0.5 s^T J s of a spin vector, in the units of the input J."""
        if self.fp32_spmv:
            return float(fp32_energies(J_csr, spins, self._normalization_factor)[0])
        if hasattr(J_csr, 'dot'):
            energy = -0.5 * cp.dot(spins, J_csr.dot(spins))
        else:
            energy = -0.5 * cp.einsum("i,ij,j->", spins, J_csr, spins)

        if hasattr(self, '_normalization_factor'):
            energy *= self._normalization_factor
        return float(energy)

    def _store_general_results(self, energy: float, alpha: float, beta: float, time_taken: float, local_search_iterations: int = 0):
        """Write (or replace) this run's row in results/<dataset>/<run_name>/<instance>.csv."""
        try:
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
                'local_search_iterations': local_search_iterations,
                'alpha': alpha,
                'beta': beta,
                'mode': self.mode,
                'precision': self.precision,
                'fp32_spmv': self.fp32_spmv,
                'ls_policy': self.ls_policy,
                'alphas_skipped': getattr(self, '_alphas_skipped', 0),
                'alphas_retried': getattr(self, '_alphas_retried', 0),
                'num_candidates': getattr(self, '_num_candidates', None),
                'sweep_time': getattr(self, '_sweep_time', None),
                'post_sweep_time': getattr(self, '_post_sweep_time', None),
                'gpu_pool_bytes': getattr(self, '_gpu_pool_bytes', None),
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
            print(f"General results saved to: {csv_file}")
        except Exception as e:
            print(f"Error saving general results: {e}")

    def _store_timing_results(self, alpha_results: list):
        """Write the per-alpha rows to results/<dataset>/<run_name>/alphas/<instance>_alphas.csv."""
        if not alpha_results:
            return
            
        alphas_results_dir = Path(f"results/{self.dataset}/{self.solver_name}/alphas")
        alphas_results_dir.mkdir(parents=True, exist_ok=True)
        
        instance_base = self.instance_name.replace('.txt', '')
        csv_file = alphas_results_dir / f"{instance_base}_alphas.csv"
        
        df_new = pd.DataFrame(alpha_results)
        
        if csv_file.exists():
            df_existing = pd.read_csv(csv_file, float_precision='round_trip')

            # A rerun replaces the whole previous sweep of this (instance, dataset, seed, run).
            # Matching rows per alpha would fail: some alpha values do not round-trip through the
            # CSV, and alphas skipped in the new run would keep stale rows.
            seed = self.random_seed
            same_run = (
                (df_existing['instance_name'] == self.instance_name) &
                (df_existing['dataset'] == self.dataset) &
                (df_existing['seed'].isna() if seed is None else df_existing['seed'] == seed) &
                (df_existing['solver_name'] == self.solver_name)
            )
            df = pd.concat([df_existing[~same_run], df_new], ignore_index=True)
        else:
            df = df_new
        
        df = df.sort_values(['seed', 'alpha']).reset_index(drop=True)
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Alpha results saved to: {csv_file} ({len(alpha_results)} alpha values)")


# =================
# GPU KERNELS
# =================

_update_spins_kernel_float32 = cp.RawKernel(r'''
extern "C" __global__
void update_spins_kernel_float32(
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
''', "update_spins_kernel_float32")

_update_spins_kernel_float64 = cp.RawKernel(r'''
extern "C" __global__
void update_spins_kernel_float64(
    const int n_selected,
    const int n,
    const int k,
    const int* selected_nodes,
    const int* indptr,
    const int* indices,
    const double* data,
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

    double local_sum = 0.0;

    // Partial sum across warp
    for (int idx = row_start + lane; idx < row_end; idx += 32) {
        int j = indices[idx];
        long long idx_n = (long long)s * n + j;  // 64-bit: k*n can exceed 2^31
        local_sum += data[idx] * (double)spins[idx_n];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Lane 0 decides whether to flip
    if (lane == 0) {
        long long idx_spin = (long long)s * n + node;
        signed char si = spins[idx_spin];
        double delta_E = 2.0 * (double)si * local_sum;
        if (delta_E < 0.0) {
            spins[idx_spin] = -si;
            atomicAdd(&improvement_counts[s], 1);
        }
    }
}
''', "update_spins_kernel_float64")


# Energy of +-1 candidates (int8, candidate c at offset c * n) against float32 couplings,
# accumulated in float64: one warp per row, one partial sum per block and candidate, summed in
# a fixed order (deterministic). Used with fp32_spmv instead of a float64 copy of J.
_energy_rows_kernel_float32 = cp.RawKernel(r'''
extern "C" __global__
void energy_rows_kernel_float32(
    const int n,
    const int* indptr,
    const int* indices,
    const float* data,
    const signed char* spins,
    double* block_sums)
{
    __shared__ double warp_sums[32];
    int lane = threadIdx.x;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    long long base = (long long)blockIdx.y * n;          // candidate offset

    double acc = 0.0;
    if (row < n) {
        for (int idx = indptr[row] + lane; idx < indptr[row + 1]; idx += 32) {
            acc += (double)data[idx] * (double)spins[base + indices[idx]];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        warp_sums[threadIdx.y] = (row < n) ? acc * (double)spins[base + row] : 0.0;
    }
    __syncthreads();
    if (lane == 0 && threadIdx.y == 0) {
        double total = 0.0;
        for (int w = 0; w < blockDim.y; ++w) total += warp_sums[w];
        block_sums[(long long)blockIdx.y * gridDim.x + blockIdx.x] = total;
    }
}
''', "energy_rows_kernel_float32")


def fp32_energies(J_csr, spins, normalization=1.0):
    """
    E = -0.5 s^T J s * normalization for a +-1 vector or for every column of an n x k matrix,
    with float32 J and float64 accumulation. Returns a cupy float64 array of k energies.
    """
    S = spins.reshape(-1, 1) if spins.ndim == 1 else spins
    S = cp.asfortranarray(S.astype(cp.int8, copy=False))        # column c at offset c * n
    n, k = S.shape
    warps_per_block = 4
    grid_x = (n + warps_per_block - 1) // warps_per_block
    block_sums = cp.empty((k, grid_x), dtype=cp.float64)
    _energy_rows_kernel_float32(
        (grid_x, k), (32, warps_per_block),
        (n, J_csr.indptr, J_csr.indices, J_csr.data.astype(cp.float32, copy=False), S, block_sums))
    return (-0.5 * normalization) * block_sums.sum(axis=1)


# =================
# LOCAL SEARCH 
# =================

def gpu_final_sweep(J_csr,
                    spins_flat: cp.ndarray,
                    n: int,
                    k: int,
                    node_fraction: float = 0.5,
                    max_sub_iters: int = 5,
                    num_rounds: int = 3,
                    dtype=cp.float32):
    """
    Final sweep of the local search, on k candidates stored one after the other in spins_flat.

    Each of num_rounds rounds takes a new random subset of node_fraction * n nodes and runs up
    to max_sub_iters passes on it, stopping early when no spin flips. dtype (float32 or float64)
    is the precision of the couplings in the kernel. Returns spins_flat.
    """
    indptr  = J_csr.indptr.astype(cp.int32, copy=False)
    indices = J_csr.indices.astype(cp.int32, copy=False)
    data    = J_csr.data.astype(dtype, copy=False)

    if dtype == cp.float32:
        kernel = _update_spins_kernel_float32
    elif dtype == cp.float64:
        kernel = _update_spins_kernel_float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

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
            
            kernel(
                grid, threads_per_block,
                (num_to_select, n, k,
                 selected_nodes, indptr, indices, data,
                 spins_flat, improvement_counts)
            )
            
            if not cp.any(improvement_counts > 0):
                break

    return spins_flat


def flip_refinement(J_csr, spins_list, max_iterations=1000,
                    final_sweep_fraction=1.0, final_sweep_iters=25,
                    num_rounds=25, subset_fraction=0.9, dtype=cp.float32):
    """
    GPU local search of a batch of candidates (Algorithm 2). Returns (n x k int8 spins, number of passes).

    Args:
        J_csr: couplings (CSR matrix on the GPU)
        spins_list: the k candidates (+-1 vectors); they are not modified
        max_iterations: maximum number of passes; each pass updates a new random subset of
            nodes, and the search stops once every candidate has had a pass without a flip
        final_sweep_fraction: share of the nodes in each round of the final sweep
        final_sweep_iters: maximum passes per round of the final sweep
        num_rounds: number of rounds of the final sweep
        subset_fraction: share of the nodes updated in each pass
        dtype: precision of the couplings in the kernel (float32 or float64)
    """
    # Candidate-major flat buffer (spins_flat[s*n + j]) built with a single copy
    spins_flat = cp.stack([sp.astype(cp.int8, copy=False) for sp in spins_list], axis=0).reshape(-1)
    k, n = len(spins_list), spins_flat.size // len(spins_list)

    indptr  = J_csr.indptr.astype(cp.int32, copy=False)
    indices = J_csr.indices.astype(cp.int32, copy=False)
    data    = J_csr.data.astype(dtype, copy=False)

    if dtype == cp.float32:
        kernel = _update_spins_kernel_float32
    elif dtype == cp.float64:
        kernel = _update_spins_kernel_float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

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

        kernel(
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
        num_rounds=num_rounds,
        dtype=dtype
    )

    Spins_final = spins_flat.reshape(k, n).T.astype(cp.int8)

    return Spins_final, max_iters_used

