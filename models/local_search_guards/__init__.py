"""Guarded local-search variants for SpecAnn.

Each variant takes a batch of candidate spin vectors and returns refined spins plus summary metrics.
"""

from .registry import REFINERS, get_refiner

__all__ = ["REFINERS", "get_refiner"]
