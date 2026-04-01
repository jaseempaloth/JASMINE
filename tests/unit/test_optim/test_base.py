import pytest

from jasmine.optim._base import BaseOptimizer


def test_base_optimizer_is_abstract():
    with pytest.raises(TypeError):
        BaseOptimizer()
