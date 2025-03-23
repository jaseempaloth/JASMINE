import jax
import jax.numpy as jnp
from jax import grad
from ..metrics.loss import MSELoss, MAELoss, RMSELoss

class LinearRegression:
    def __init__(self, fit_intercept=True, loss=MSELoss):
        """
        Linear Regression model using JAX.
        
        Args:
            fit_intercept (bool): Whether to fit an intercept term
            loss (str or Loss): Loss function to use. Can be 'mse', 'mae', 'rmse' or a Loss instance
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.loss = loss()
    
    def forward(self, params, X):
        """
        Forward pass for the linear model.
        
        Args:
            params (jnp.ndarray): Model parameters
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted values
        """
        if self.fit_intercept:
            return jnp.dot(X, params[1:]) + params[0]
        else:
            return jnp.dot(X, params)

    def fit(self, X, y, learning_rate=0.01, max_iter=1000, tol=1e-4):
        """
        Fit the linear regression model using gradient descent.
        
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            learning_rate (float): Learning rate for gradient descent
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
            
        Returns:
            self: The fitted model
        """
        # Initialize parameters
        if self.fit_intercept:
            params = jnp.zeros(X.shape[1] + 1)
        else:
            params = jnp.zeros(X.shape[1])
        
        # Gradient descent loop
        prev_loss = float('inf')
        for i in range(max_iter):
            # Compute gradients
            grads = self.loss.grad(params, X, y, self)
            
            # Update parameters
            params = params - learning_rate * grads

            # Check for convergence
            current_loss = self.loss(params, X, y, self)
            if jnp.abs(current_loss - prev_loss) < tol:
                break
            prev_loss = current_loss
        
        self.params_ = params

        if self.fit_intercept:
            self.intercept_ = self.params_[0]
            self.coef_ = self.params_[1:]
        else:
            self.coef_ = self.params_
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Args:
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted values
        """
        if not hasattr(self, 'params_'):
            raise ValueError("Model has not been fitted yet.")
        return self.forward(self.params_, X)
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            
        Returns:
            float: R^2 score
        """
        y_pred = self.predict(X)
        u = jnp.sum(jnp.square(y - y_pred))
        v = jnp.sum(jnp.square(y - jnp.mean(y)))
        return 1 - u / v if v != 0 else 0


        
        



