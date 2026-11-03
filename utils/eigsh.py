"""CuPy's eigsh, with its Lanczos iteration count when CuPy has the optional patch of docs/cupy_eigsh_patch.md.

Stock CuPy returns (eigenvalues, eigenvectors); the patched one also returns the iteration count. Both work here:
without the patch the count is None, the solvers leave their `eigsh_iterations` column empty, and they check the
convergence of every eigenpair from its residual.
"""
import cupyx.scipy.sparse.linalg as cpx_splinalg


def eigsh(A, **kwargs):
    """Return (eigenvalues, eigenvectors, iterations); iterations is None with stock CuPy."""
    out = cpx_splinalg.eigsh(A, **kwargs)
    if len(out) == 3:
        w, v, iterations = out
        return w, v, int(iterations.item() if hasattr(iterations, 'item') else iterations)
    w, v = out
    return w, v, None
