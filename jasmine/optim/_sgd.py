"""Vanilla Stochastic Gradient Descent optimizer."""

from __future__ import annotations

from typing import Tuple

import jax

from jasmine._typing import OptState, Params
from jasmine.optim._base import BaseOptimizer


class SGD(BaseOptimizer):
    """Vanilla SGD: ``params -= learning_rate * grads``."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        self.learning_rate = learning_rate

    def init(self, params: Params) -> OptState:
        return {}

    def update(self, grads: Params, state: OptState) -> Tuple[Params, OptState]:
        updates = jax.tree_util.tree_map(lambda g: self.learning_rate * g, grads)
        return updates, state
