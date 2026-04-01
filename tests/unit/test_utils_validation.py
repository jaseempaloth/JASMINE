import jax.numpy as jnp
import pytest

from jasmine.utils._validation import (
    validate_1d_array,
    validate_2d_array,
    validate_fitted,
    validate_matching_samples,
)


def test_validate_2d_array_passes_for_2d():
    validate_2d_array(jnp.ones((10, 3)))  # must not raise


def test_validate_2d_array_raises_for_1d():
    with pytest.raises(ValueError, match="must be a 2D array"):
        validate_2d_array(jnp.ones((10,)))


def test_validate_1d_array_passes_for_1d():
    validate_1d_array(jnp.ones((10,)))  # must not raise


def test_validate_1d_array_raises_for_2d():
    with pytest.raises(ValueError, match="must be a 1D array"):
        validate_1d_array(jnp.ones((10, 1)))


def test_validate_matching_samples_passes_when_equal():
    validate_matching_samples(jnp.ones((10, 3)), jnp.ones((10,)))  # must not raise


def test_validate_matching_samples_raises_on_mismatch():
    with pytest.raises(ValueError, match="must match"):
        validate_matching_samples(jnp.ones((10, 3)), jnp.ones((8,)))


def test_validate_fitted_raises_when_none():
    with pytest.raises(ValueError, match="Call `train` before calling `inference`"):
        validate_fitted(None, "inference")


def test_validate_fitted_passes_when_not_none():
    validate_fitted({"w": jnp.ones(3)}, "inference")  # must not raise
