import jax
import jax.numpy as jnp
import pytest

from jasmine.losses._regression import huber_loss, mae_loss, mse_loss


def test_mse_loss_zero_for_perfect_predictions():
    y = jnp.array([1.0, 2.0, 3.0])
    assert float(mse_loss(y, y)) == pytest.approx(0.0)


def test_mse_loss_correct_value():
    y_true = jnp.array([0.0, 0.0])
    y_pred = jnp.array([1.0, 3.0])
    assert float(mse_loss(y_true, y_pred)) == pytest.approx(5.0)


def test_mae_loss_zero_for_perfect_predictions():
    y = jnp.array([1.0, 2.0, 3.0])
    assert float(mae_loss(y, y)) == pytest.approx(0.0)


def test_mae_loss_correct_value():
    y_true = jnp.array([0.0, 0.0])
    y_pred = jnp.array([1.0, 3.0])
    assert float(mae_loss(y_true, y_pred)) == pytest.approx(2.0)


def test_huber_loss_equals_mse_for_small_residuals():
    y_true = jnp.array([0.0, 0.0])
    y_pred = jnp.array([0.1, -0.1])
    expected = float(0.5 * jnp.mean(jnp.array([0.01, 0.01])))
    assert float(huber_loss(y_true, y_pred, delta=1.0)) == pytest.approx(expected)


def test_huber_loss_linear_for_large_residuals():
    y_true = jnp.array([0.0])
    y_pred = jnp.array([10.0])
    assert float(huber_loss(y_true, y_pred, delta=1.0)) == pytest.approx(9.5)


def test_mse_loss_is_differentiable():
    y_true = jnp.array([1.0, 2.0])

    def loss_fn(pred):
        return mse_loss(y_true, pred)

    grad = jax.grad(loss_fn)(jnp.array([0.0, 0.0]))
    assert grad.shape == (2,)
