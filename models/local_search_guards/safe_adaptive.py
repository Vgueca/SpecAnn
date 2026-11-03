"""Adaptive Guard: guarded batch flips whose batch size adapts per candidate."""

from __future__ import annotations

from .safe_common import refine_safe


def refine(matrix, spins, config):
    return refine_safe(matrix, spins, config, adaptive=True)
