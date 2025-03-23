import jax
import jax.numpy as jnp
from ..metrics.loss import MSELoss, MAELoss, RMSELoss

class LinearRegression:
    def __init__(self, fit_intercept=True, loss='mse'):
        """
        Linear Regression model using JAX.
        
        Args:
            fit_intercept (bool): Whether to fit an intercept term
            loss (str or Loss): Loss function to use. Can be 'mse', 'mae', 'rmse' or a Loss instance
        """
        