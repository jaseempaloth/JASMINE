"""Centralized input validation helpers shared across JASMINE submodules."""

from __future__ import annotations

import jax


def validate_2d_array(X: jax.Array, name: str = "X") -> None:
    """Raise ValueError if X is not a 2-D array."""
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {X.shape}.")


def validate_1d_array(y: jax.Array, name: str = "y") -> None:
    """Raise ValueError if y is not a 1-D array."""
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {y.shape}.")


def validate_matching_samples(X: jax.Array, y: jax.Array) -> None:
    """Raise ValueError if X and y have different numbers of rows."""
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Number of samples in X ({X.shape[0]}) must match number of samples"
            f" in y ({y.shape[0]})."
        )


def validate_fitted(params, method_name: str) -> None:
    """Raise ValueError if params is None (model not trained yet)."""
    if params is None:
        raise ValueError(
            f"Model has not been trained yet. Call `train` before calling `{method_name}`."
        )
