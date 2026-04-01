"""Regression loss functions — pure, JIT-compatible JAX functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def mse_loss(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Mean Squared Error loss."""
    return jnp.mean(jnp.square(y_true - y_pred))


@jax.jit
def mae_loss(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Mean Absolute Error loss."""
    return jnp.mean(jnp.abs(y_true - y_pred))


@jax.jit
def huber_loss(y_true: jax.Array, y_pred: jax.Array, delta: float = 1.0) -> jax.Array:
    """Huber loss — quadratic for small residuals, linear beyond ``delta``."""
    residual = jnp.abs(y_true - y_pred)
    quadratic = 0.5 * jnp.square(residual)
    linear = delta * residual - 0.5 * delta**2
    return jnp.mean(jnp.where(residual <= delta, quadratic, linear))
