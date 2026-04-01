import jax
import jax.numpy as jnp
import pytest

from jasmine.optim._adam import Adam


@pytest.fixture
def params():
    return {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}


@pytest.fixture
def grads():
    return {"w": jnp.array([1.0, 1.0]), "b": jnp.array(1.0)}


def test_adam_init_creates_zero_moments_and_t0(params):
    opt = Adam()
    state = opt.init(params)
    assert "m" in state and "v" in state and "t" in state
    assert jnp.allclose(state["m"]["w"], jnp.zeros_like(params["w"]))
    assert jnp.allclose(state["v"]["w"], jnp.zeros_like(params["w"]))
    assert int(state["t"]) == 0


def test_adam_step_counter_increments(params, grads):
    opt = Adam()
    state = opt.init(params)
    _, s1 = opt.update(grads, state)
    _, s2 = opt.update(grads, s1)
    assert int(s1["t"]) == 1
    assert int(s2["t"]) == 2


def test_adam_update_shape_matches_grads(params, grads):
    opt = Adam()
    updates, _ = opt.update(grads, opt.init(params))
    assert updates["w"].shape == grads["w"].shape
    assert updates["b"].shape == grads["b"].shape


def test_adam_update_bounded_by_learning_rate(params, grads):
    lr = 0.001
    opt = Adam(learning_rate=lr)
    updates, _ = opt.update(grads, opt.init(params))
    assert float(jnp.max(jnp.abs(updates["w"]))) <= lr * 10


def test_adam_converges_on_simple_quadratic():
    def loss_fn(p):
        return jnp.sum(p["w"] ** 2)

    p = {"w": jnp.array([5.0, 5.0])}
    opt = Adam(learning_rate=0.1)
    state = opt.init(p)
    for _ in range(100):
        g = jax.grad(loss_fn)(p)
        u, state = opt.update(g, state)
        p = jax.tree_util.tree_map(lambda x, upd: x - upd, p, u)
    assert float(loss_fn(p)) < 0.01
