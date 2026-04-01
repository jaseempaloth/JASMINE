import jax
import jax.numpy as jnp
import pytest

from jasmine.optim._sgd import SGD


@pytest.fixture
def params():
    return {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array(0.5)}


@pytest.fixture
def grads():
    return {"w": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array(0.05)}


def test_sgd_init_returns_empty_state(params):
    opt = SGD(learning_rate=0.01)
    state = opt.init(params)
    assert state == {}


def test_sgd_update_returns_correct_updates(params, grads):
    lr = 0.1
    opt = SGD(learning_rate=lr)
    updates, new_state = opt.update(grads, opt.init(params))
    assert jnp.allclose(updates["w"], lr * grads["w"])
    assert jnp.allclose(updates["b"], lr * grads["b"])
    assert new_state == {}


def test_sgd_state_is_unchanged_after_update(params, grads):
    opt = SGD(learning_rate=0.01)
    state = opt.init(params)
    _, new_state = opt.update(grads, state)
    assert new_state == {}


def test_sgd_applied_update_reduces_loss():
    opt = SGD(learning_rate=0.1)
    params = {"w": jnp.array([2.0])}

    def loss_fn(p):
        return jnp.sum(p["w"] ** 2)

    grads = jax.grad(loss_fn)(params)
    updates, _ = opt.update(grads, opt.init(params))
    new_params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
    assert float(loss_fn(new_params)) < float(loss_fn(params))
