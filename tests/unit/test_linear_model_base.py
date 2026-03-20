import jax.numpy as jnp
import pytest

from jasmine.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from jasmine.linear_model.base import BaseLinearModel, BinaryClassifierMixin, RegressorMixin


def test_linear_models_inherit_shared_linear_model_base():
    linear = LinearRegression()
    logistic = LogisticRegression()

    assert isinstance(linear, BaseLinearModel)
    assert isinstance(logistic, BaseLinearModel)
    assert isinstance(linear, RegressorMixin)
    assert isinstance(logistic, BinaryClassifierMixin)


def test_shared_parameter_initialization_handles_optional_bias():
    linear_params = LinearRegression(use_bias=False).init_params(3)
    logistic_params = LogisticRegression(use_bias=False).init_params(3)

    assert linear_params["w"].shape == (3,)
    assert logistic_params["w"].shape == (3,)
    assert linear_params["b"] is None
    assert logistic_params["b"] is None


def test_shared_fitted_state_validation_is_used_by_mixins():
    X = jnp.ones((2, 2))

    with pytest.raises(ValueError, match="Call `train` before calling `inference`"):
        LinearRegression().inference(X)

    with pytest.raises(ValueError, match="Call `train` before calling `predict_probabilities`"):
        LogisticRegression().predict_probabilities(X)


def test_ridge_and_lasso_configure_expected_penalties():
    ridge = Ridge(alpha=0.25)
    lasso = Lasso(alpha=0.75)

    assert ridge.alpha == 0.25
    assert ridge.l1_penalty == 0.0
    assert ridge.l2_penalty == 0.25
    assert lasso.alpha == 0.75
    assert lasso.l1_penalty == 0.75
    assert lasso.l2_penalty == 0.0


def test_preprocess_training_data_centers_regression_features_and_targets():
    model = LinearRegression()
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([5.0, 9.0])

    X_processed, y_processed, X_offset, y_offset = model._preprocess_training_data(X, y)

    assert jnp.allclose(X_processed.mean(axis=0), jnp.zeros(2))
    assert jnp.allclose(y_processed.mean(), 0.0)
    assert jnp.allclose(X_offset, jnp.array([2.0, 3.0]))
    assert jnp.allclose(y_offset, 7.0)


def test_preprocess_training_data_keeps_classifier_targets_unchanged():
    model = LogisticRegression()
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([0, 1])

    X_processed, y_processed, X_offset, y_offset = model._preprocess_training_data(X, y)

    assert jnp.allclose(X_processed.mean(axis=0), jnp.zeros(2))
    assert jnp.array_equal(y_processed, y)
    assert jnp.allclose(X_offset, jnp.array([2.0, 3.0]))
    assert jnp.allclose(y_offset, 0.0)


def test_set_intercept_maps_centered_bias_back_to_raw_space():
    model = LinearRegression()
    centered_params = {"w": jnp.array([2.0, -1.0]), "b": jnp.array(0.5)}
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    X_offset = jnp.array([2.0, 3.0])
    y_offset = jnp.array(10.0)

    params = model._set_intercept(centered_params, X_offset, y_offset)

    raw_predictions = model.forward(params, X)
    centered_predictions = model.forward(centered_params, X - X_offset) + y_offset

    assert jnp.allclose(raw_predictions, centered_predictions)
    assert jnp.allclose(model.intercept_, params["b"])
