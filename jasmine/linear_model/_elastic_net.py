"""ElasticNet: linear regression with combined L1 and L2 regularization."""

from __future__ import annotations

from jasmine.linear_model._linear import LinearRegression
from jasmine.losses import mse_loss


class ElasticNet(LinearRegression):
    """Linear regression with Elastic Net regularization (L1 + L2)."""

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        use_bias: bool = True,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        loss_function=mse_loss,
        optimizer=None,
    ) -> None:
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        super().__init__(
            use_bias=use_bias,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            loss_function=loss_function,
            l1_penalty=alpha * l1_ratio,
            l2_penalty=alpha * (1.0 - l1_ratio),
            optimizer=optimizer,
        )
