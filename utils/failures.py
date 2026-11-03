"""Exceptions raised by the solvers, and the out-of-memory test they use."""


class NumericalFailure(RuntimeError):
    """The solver ran but produced no valid solution (e.g. the eigensolver failed at every alpha)."""


# Substrings of exception messages that mean the host or the GPU ran out of memory (lower case).
_OUT_OF_MEMORY_TEXT = ('cuda out of memory', 'cudaerrormemoryallocation', 'cublas_status_alloc_failed',
                       'cusparse_status_alloc_failed', 'cusolver_status_alloc_failed', 'cufft_alloc_failed',
                       'out of memory on device', 'outofmemoryerror', 'out of memory', 'unable to allocate',
                       'cannot allocate memory', 'bad_alloc', 'out of space', 'memoryerror')


def is_out_of_memory(exc: BaseException) -> bool:
    """True if the exception, or one in its cause chain, means that the host or the GPU ran out of memory."""
    seen = set()
    while exc is not None and id(exc) not in seen and len(seen) < 10:
        seen.add(id(exc))
        if isinstance(exc, MemoryError):          # CuPy's OutOfMemoryError is a MemoryError
            return True
        if any(s in f"{type(exc).__name__}: {exc}".lower() for s in _OUT_OF_MEMORY_TEXT):
            return True
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return False
