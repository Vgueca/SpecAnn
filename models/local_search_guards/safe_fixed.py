"""Fixed Guard: fixed-size batch flips, kept only if the FP64 energy goes down."""

from __future__ import annotations

from .safe_common import refine_safe


def refine(matrix, spins, config):
    return refine_safe(matrix, spins, config, adaptive=False)
