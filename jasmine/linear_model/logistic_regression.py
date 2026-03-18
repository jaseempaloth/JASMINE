import os

import jax

from jasmine.linear_model.base import BaseLinearModel, BinaryClassifierMixin
from jasmine.metrics import binary_cross_entropy


class LogisticRegression(BinaryClassifierMixin, BaseLinearModel):
    """Binary logistic regression trained with gradient descent."""

    stop_on_invalid_loss = True

    def __init__(
        self,
        use_bias=True,
        learning_rate=0.01,
        n_epochs=1000,
        loss_function=binary_cross_entropy,
        l1_penalty=0.0,
        l2_penalty=0.0,
    ):
        super().__init__(
            use_bias=use_bias,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            loss_function=loss_function,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
        )

    def _initialization_key(self):
        random_state = int.from_bytes(os.urandom(4), "big")
        return jax.random.PRNGKey(random_state)

    def loss_fn(self, params, X, y):
        logits = self.forward(params, X)
        loss = self.loss_function(y, logits, from_logits=True)
        return loss + self._regularization_penalty(params)
