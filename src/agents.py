"""Legacy compatibility exports for strategy interfaces.

This file intentionally keeps top-level imports stable while the project
moves to the modular architecture under `strategies/`.
"""

from strategies.base import Strategy
from strategies.random_uniform import RandomUniformStrategy

__all__ = ["Strategy", "RandomUniformStrategy"]