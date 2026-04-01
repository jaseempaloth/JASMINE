import jax.numpy as jnp
import pytest

from jasmine.linear_model import ElasticNet
from jasmine.linear_model._base import BaseLinearModel, RegressorMixin


def test_elastic_net_is_a_linear_model():
    assert issubclass(ElasticNet, BaseLinearModel)
    assert issubclass(ElasticNet, RegressorMixin)


def test_elastic_net_configures_both_penalties():
    model = ElasticNet(alpha=1.0, l1_ratio=0.3)
    assert model.alpha == 1.0
    assert model.l1_ratio == 0.3
    assert model.l1_penalty == pytest.approx(0.3)
    assert model.l2_penalty == pytest.approx(0.7)


def test_elastic_net_l1_ratio_zero_is_ridge():
    model = ElasticNet(alpha=0.5, l1_ratio=0.0)
    assert model.l1_penalty == pytest.approx(0.0)
    assert model.l2_penalty == pytest.approx(0.5)


def test_elastic_net_l1_ratio_one_is_lasso():
    model = ElasticNet(alpha=0.5, l1_ratio=1.0)
    assert model.l1_penalty == pytest.approx(0.5)
    assert model.l2_penalty == pytest.approx(0.0)


def test_elastic_net_trains_and_predicts():
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = jnp.array([3.0, 7.0, 11.0, 15.0])
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, n_epochs=10, learning_rate=0.01)
    history = model.train(X, y, verbose=0)
    assert len(history["loss"]) == 10
    assert model.inference(X).shape == (4,)


def test_elastic_net_raises_for_invalid_l1_ratio():
    with pytest.raises(ValueError, match="l1_ratio"):
        ElasticNet(l1_ratio=1.5)
