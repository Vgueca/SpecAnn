"""Predictors of the next eigenvector, used to warm-start eigsh at each alpha."""

import cupy as cp


def diagonal_predictor(y_last, d_vec, delta_alpha, dtype):
    """
    Diagonal predictor (Algorithm 1): D^{delta_alpha/2} y_last, normalized.

    Args:
        y_last: last eigenvector (cupy array)
        d_vec: degree vector (cupy array)
        delta_alpha: change of alpha since the last eigenvector
        dtype: data type of the computation

    Returns:
        The predicted eigenvector, normalized.
    """
    w = cp.power(cp.maximum(d_vec, cp.array(1e-12, dtype=dtype)), delta_alpha / 2.0)
    y = w * y_last
    norm = cp.linalg.norm(y)
    if norm < 1e-12:
        # Zero vector: fall back to a random start
        z = cp.random.randn(*y.shape).astype(dtype)
        return z / cp.linalg.norm(z)
    return y / norm


def secant_predictor(y_last, y_prev, dtype, gamma=1.0):
    """
    Secant predictor: y_last + gamma (y_last - y_prev), normalized.

    Args:
        y_last: last eigenvector (cupy array)
        y_prev: eigenvector before it (cupy array, or None: return y_last)
        dtype: not used
        gamma: secant step, in (0, 2)

    Returns:
        The predicted eigenvector, normalized.
    """
    if y_prev is None:
        return y_last
    # Eigenvectors have an arbitrary sign: align y_prev with y_last
    dot_prod = cp.dot(y_last, y_prev)
    if dot_prod < 0:
        y_prev = -y_prev
    s = y_last - y_prev
    s_norm = cp.linalg.norm(s)
    if s_norm < 1e-12:
        return y_last
    y_pred = y_last + gamma * s
    norm = cp.linalg.norm(y_pred)
    if norm < 1e-12:
        return y_last
    return y_pred / norm
