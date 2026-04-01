"""Adam optimizer (Adaptive Moment Estimation)."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jasmine._typing import OptState, Params
from jasmine.optim._base import BaseOptimizer


class Adam(BaseOptimizer):
    """Adam optimizer with bias correction."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}.")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init(self, params: Params) -> OptState:
        return {
            "m": jax.tree_util.tree_map(jnp.zeros_like, params),
            "v": jax.tree_util.tree_map(jnp.zeros_like, params),
            "t": jnp.array(0, dtype=jnp.int32),
        }

    def update(self, grads: Params, state: OptState) -> Tuple[Params, OptState]:
        t = state["t"] + 1
        m = jax.tree_util.tree_map(
            lambda m_prev, g: self.beta1 * m_prev + (1.0 - self.beta1) * g,
            state["m"],
            grads,
        )
        v = jax.tree_util.tree_map(
            lambda v_prev, g: self.beta2 * v_prev + (1.0 - self.beta2) * g**2,
            state["v"],
            grads,
        )
        m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1.0 - self.beta1**t), m)
        v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1.0 - self.beta2**t), v)
        updates = jax.tree_util.tree_map(
            lambda m_h, v_h: self.learning_rate * m_h / (jnp.sqrt(v_h) + self.eps),
            m_hat,
            v_hat,
        )
        return updates, {"m": m, "v": v, "t": t}
