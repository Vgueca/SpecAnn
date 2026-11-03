# Optional CuPy patch: `eigsh` returns the Lanczos iteration count

Stock CuPy's `cupyx.scipy.sparse.linalg.eigsh` returns two values. With this patch it also returns the number of
Lanczos iterations, which the solvers then record (column `eigsh_iterations`). The solvers run without it
(`utils/eigsh.py`). Patch the installed CuPy once per environment (tested with CuPy 13.6.0):

```bash
F=$(python -c "import cupyx.scipy.sparse.linalg._eigen as m; print(m.__file__)")
cp "$F" "$F.orig"
sed -i 's/^        return w\[idx\], x\[:, idx\]$/        return w[idx], x[:, idx], iter/; s/^        return cupy.sort(w)$/        return cupy.sort(w), iter/' "$F"
grep -n "return w\[idx\], x\[:, idx\], iter\|return cupy.sort(w), iter" "$F"   # both lines must print
```

- **What it returns:** `iter`, the thick-restart Lanczos counter that `eigsh` already keeps. It starts at `ncv` and
  grows by `ncv - k` at each restart.
- **Side effect:** the patch also changes the return value of CuPy's own `svds`. This repository does not use it.
- **To undo it:** copy `"$F.orig"` back over `"$F"`.
