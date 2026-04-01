from ._linear import LinearRegression
from jasmine.losses import mse_loss


class Lasso(LinearRegression):
    """Linear regression with L1 regularization."""

    def __init__(
        self,
        alpha=1.0,
        use_bias=True,
        learning_rate=0.01,
        n_epochs=1000,
        loss_function=mse_loss,
        optimizer=None,
    ):
        self.alpha = alpha
        super().__init__(
            use_bias=use_bias,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            loss_function=loss_function,
            l1_penalty=alpha,
            l2_penalty=0.0,
            optimizer=optimizer,
        )
