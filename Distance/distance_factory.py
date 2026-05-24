"""Unified factory for Distance/* calculators."""

from __future__ import annotations

try:
    from Distance.common import create_distance_calculator, available_methods
except Exception:
    from .common import create_distance_calculator, available_methods

__all__ = ["create_distance_calculator", "available_methods"]
