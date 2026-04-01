import jax.numpy as jnp
import pytest

from jasmine.preprocessing._scaler import StandardScaler, fit_fn


def test_scaler_fit_fn_reuses_shared_2d_array_validation_message():
    with pytest.raises(ValueError, match="X must be a 2D array"):
        fit_fn(jnp.ones((4,)))


def test_standard_scaler_requires_fit_before_transform():
    with pytest.raises(RuntimeError, match="StandardScaler is not fitted yet. Call 'fit' first."):
        StandardScaler().transform(jnp.ones((2, 2)))
