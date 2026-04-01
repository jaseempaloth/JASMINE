"""SGD with momentum optimizer."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jasmine._typing import OptState, Params
from jasmine.optim._base import BaseOptimizer


class Momentum(BaseOptimizer):
    """SGD with exponential moving average of gradients."""

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}.")
        self.learning_rate = learning_rate
        self.beta = beta

    def init(self, params: Params) -> OptState:
        return {"velocity": jax.tree_util.tree_map(jnp.zeros_like, params)}

    def update(self, grads: Params, state: OptState) -> Tuple[Params, OptState]:
        velocity = jax.tree_util.tree_map(
            lambda v, g: self.beta * v + (1.0 - self.beta) * g,
            state["velocity"],
            grads,
        )
        updates = jax.tree_util.tree_map(lambda v: self.learning_rate * v, velocity)
        return updates, {"velocity": velocity}
