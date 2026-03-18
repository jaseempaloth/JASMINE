from jasmine.linear_model.base import BaseLinearModel, RegressorMixin
from jasmine.metrics import mean_squared_error


class LinearRegression(RegressorMixin, BaseLinearModel):
    """Linear regression trained with gradient descent."""

    center_targets = True
    log_every_epoch = True

    def __init__(
        self,
        use_bias=True,
        learning_rate=0.01,
        n_epochs=1000,
        loss_function=mean_squared_error,
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

    def loss_fn(self, params, X, y):
        predictions = self.forward(params, X)
        loss = self.loss_function(y, predictions)
        return loss + self._regularization_penalty(params)
