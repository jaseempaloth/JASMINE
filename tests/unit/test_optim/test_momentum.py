import jax
import jax.numpy as jnp
import pytest

from jasmine.optim._momentum import Momentum


@pytest.fixture
def params():
    return {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}


@pytest.fixture
def grads():
    return {"w": jnp.array([1.0, 1.0]), "b": jnp.array(1.0)}


def test_momentum_init_creates_zero_velocity(params):
    opt = Momentum(learning_rate=0.01, beta=0.9)
    state = opt.init(params)
    assert "velocity" in state
    assert jnp.allclose(state["velocity"]["w"], jnp.zeros_like(params["w"]))
    assert jnp.allclose(state["velocity"]["b"], jnp.zeros_like(params["b"]))


def test_momentum_first_update_with_zero_velocity(params, grads):
    lr, beta = 0.1, 0.9
    opt = Momentum(learning_rate=lr, beta=beta)
    updates, new_state = opt.update(grads, opt.init(params))
    expected_v = (1.0 - beta) * grads["w"]
    assert jnp.allclose(new_state["velocity"]["w"], expected_v)
    assert jnp.allclose(updates["w"], lr * expected_v)


def test_momentum_velocity_accumulates_across_steps(params, grads):
    opt = Momentum(learning_rate=0.1, beta=0.9)
    state = opt.init(params)
    _, state1 = opt.update(grads, state)
    _, state2 = opt.update(grads, state1)
    assert float(jnp.sum(jnp.abs(state2["velocity"]["w"]))) > float(
        jnp.sum(jnp.abs(state1["velocity"]["w"]))
    )


def test_momentum_converges_faster_than_sgd_on_quadratic():
    from jasmine.optim._sgd import SGD

    def loss_fn(p):
        return jnp.sum(p["w"] ** 2)

    init_params = {"w": jnp.array([5.0, 5.0])}
    n_steps = 20

    opt_sgd = SGD(learning_rate=0.1)
    p_sgd = init_params
    s_sgd = opt_sgd.init(p_sgd)
    for _ in range(n_steps):
        g = jax.grad(loss_fn)(p_sgd)
        u, s_sgd = opt_sgd.update(g, s_sgd)
        p_sgd = jax.tree_util.tree_map(lambda p, u: p - u, p_sgd, u)

    opt_mom = Momentum(learning_rate=0.1, beta=0.5)
    p_mom = init_params
    s_mom = opt_mom.init(p_mom)
    for _ in range(n_steps):
        g = jax.grad(loss_fn)(p_mom)
        u, s_mom = opt_mom.update(g, s_mom)
        p_mom = jax.tree_util.tree_map(lambda p, u: p - u, p_mom, u)

    assert float(loss_fn(p_mom)) < float(loss_fn(p_sgd))
