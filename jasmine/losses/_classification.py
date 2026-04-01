"""Classification loss functions — pure, JIT-compatible JAX functions."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


def bce_loss(
    y_true: jax.Array,
    y_pred: jax.Array,
    from_logits: bool = False,
    sample_weight: Optional[jax.Array] = None,
) -> jax.Array:
    """Binary cross-entropy loss for labels in ``{0, 1}``."""
    if from_logits:
        log_p = jax.nn.log_sigmoid(y_pred)
        log_not_p = jax.nn.log_sigmoid(-y_pred)
        per_sample = -(y_true * log_p + (1.0 - y_true) * log_not_p)
    else:
        epsilon = 1e-15
        clipped = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
        per_sample = -(y_true * jnp.log(clipped) + (1.0 - y_true) * jnp.log(1.0 - clipped))

    if sample_weight is not None:
        return jnp.mean(per_sample * sample_weight)
    return jnp.mean(per_sample)


def categorical_ce_loss(
    y_true: jax.Array,
    y_pred: jax.Array,
    from_logits: bool = False,
) -> jax.Array:
    """Categorical cross-entropy loss for one-hot targets."""
    if from_logits:
        log_probs = jax.nn.log_softmax(y_pred, axis=-1)
    else:
        epsilon = 1e-15
        clipped = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
        log_probs = jnp.log(clipped)
    return -jnp.mean(jnp.sum(y_true * log_probs, axis=-1))


@jax.jit
def hinge_loss(
    y_true: jax.Array,
    y_pred: jax.Array,
    sample_weight: Optional[jax.Array] = None,
) -> jax.Array:
    """Hinge loss for labels in ``{-1, +1}``."""
    per_sample = jnp.maximum(0.0, 1.0 - y_true * y_pred)
    if sample_weight is not None:
        return jnp.mean(per_sample * sample_weight)
    return jnp.mean(per_sample)


@jax.jit
def squared_hinge_loss(
    y_true: jax.Array,
    y_pred: jax.Array,
    sample_weight: Optional[jax.Array] = None,
) -> jax.Array:
    """Squared hinge loss for labels in ``{-1, +1}``."""
    per_sample = jnp.maximum(0.0, 1.0 - y_true * y_pred) ** 2
    if sample_weight is not None:
        return jnp.mean(per_sample * sample_weight)
    return jnp.mean(per_sample)
