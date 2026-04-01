import jax
import jax.numpy as jnp
import pytest

from jasmine.losses._classification import (
    bce_loss,
    categorical_ce_loss,
    hinge_loss,
    squared_hinge_loss,
)


def test_bce_loss_zero_for_perfect_probability_predictions():
    y_true = jnp.array([0.0, 1.0])
    y_pred = jnp.array([1e-7, 1.0 - 1e-7])
    assert float(bce_loss(y_true, y_pred)) == pytest.approx(0.0, abs=1e-4)


def test_bce_loss_from_logits_matches_probability_path():
    y_true = jnp.array([1.0, 0.0])
    logits = jnp.array([2.0, -2.0])
    probs = jax.nn.sigmoid(logits)
    assert float(bce_loss(y_true, logits, from_logits=True)) == pytest.approx(
        float(bce_loss(y_true, probs, from_logits=False)),
        abs=1e-5,
    )


def test_hinge_loss_zero_for_correct_margin():
    y_true = jnp.array([1.0, -1.0])
    y_pred = jnp.array([2.0, -2.0])
    assert float(hinge_loss(y_true, y_pred)) == pytest.approx(0.0)


def test_hinge_loss_positive_for_wrong_predictions():
    y_true = jnp.array([1.0])
    y_pred = jnp.array([-1.0])
    assert float(hinge_loss(y_true, y_pred)) == pytest.approx(2.0)


def test_squared_hinge_loss_positive_for_wrong_predictions():
    y_true = jnp.array([1.0])
    y_pred = jnp.array([-1.0])
    assert float(squared_hinge_loss(y_true, y_pred)) == pytest.approx(4.0)


def test_bce_loss_is_differentiable():
    y_true = jnp.array([1.0, 0.0])

    def loss_fn(logits):
        return bce_loss(y_true, logits, from_logits=True)

    grad = jax.grad(loss_fn)(jnp.array([0.0, 0.0]))
    assert grad.shape == (2,)


def test_categorical_ce_loss_zero_for_perfect_predictions():
    y_true = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = jnp.array([[1.0 - 1e-7, 1e-7], [1e-7, 1.0 - 1e-7]])
    assert float(categorical_ce_loss(y_true, y_pred)) == pytest.approx(0.0, abs=1e-4)
