import jax.numpy as jnp

from jasmine.linear_model import LinearRegression
from jasmine.optim import Adam, Momentum, SGD


def _make_data():
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = jnp.array([3.0, 7.0, 11.0, 15.0])
    return X, y


def test_linear_regression_trains_with_sgd_optimizer():
    X, y = _make_data()
    model = LinearRegression(n_epochs=5, optimizer=SGD(learning_rate=0.01))
    history = model.train(X, y, verbose=0)
    assert len(history["loss"]) == 5
    assert model.inference(X).shape == (4,)


def test_linear_regression_trains_with_momentum_optimizer():
    X, y = _make_data()
    model = LinearRegression(n_epochs=5, optimizer=Momentum(learning_rate=0.01))
    history = model.train(X, y, verbose=0)
    assert len(history["loss"]) == 5


def test_linear_regression_trains_with_adam_optimizer():
    X, y = _make_data()
    model = LinearRegression(n_epochs=5, optimizer=Adam(learning_rate=0.001))
    history = model.train(X, y, verbose=0)
    assert len(history["loss"]) == 5


def test_no_optimizer_still_works():
    X, y = _make_data()
    model = LinearRegression(n_epochs=5, optimizer=None)
    history = model.train(X, y, verbose=0)
    assert len(history["loss"]) == 5
