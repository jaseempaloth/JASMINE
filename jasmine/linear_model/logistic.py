import jax
import jax.numpy as jnp
from jax import grad, jit
from ..metrics import CrossEntropyLoss

class LogisticRegression:
    def __init__(self, fit_intercept=True, loss=CrossEntropyLoss):
        """
        Logistic Regression model using JAX.
        
        Args:
            fit_intercept (bool): Whether to fit an intercept term
            loss (str or Loss): Loss function to use. Can be 'cross_entropy' or a Loss instance
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.loss = loss()
    
    def forward(self, params, X):
        """
        Forward pass for the logistic model.
        
        Args:
            params (jnp.ndarray): Model parameters
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted probabilities
        """
        logits = jnp.dot(X, params[1:]) + params[0] if self.fit_intercept else jnp.dot(X, params)
        # Apply sigmoid for binary classification, softmax for multi-class
        if logits.ndim == 1 or logits.shape[1] == 1:
            return jax.nn.sigmoid(logits)
        else:
            return jax.nn.softmax(logits)

    
    def _update_params(self, params, X, y, learning_rate):
        """
        Update model parameters using gradient descent. This helper function is JIT-compiled.

        Args:
            params (jnp.ndarray): Current model parameters.
            X (jnp.ndarray): Input features.
            y (jnp.ndarray): Target values.
            learning_rate (float): Learning rate for gradient descent.
        
        Returns:
            jnp.ndarray: Updated parameters.
        """
        grads = self.loss.grad(params, X, y, self)
        return params - learning_rate * grads 
    
    def fit(self, X, y, learning_rate=0.01, max_iter=1000, tol=1e-4):
        """
        Fit the logistic regression model using gradient descent.
        
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
            # Use the _update_params helper function to update parameters
            params = self._update_params(params, X, y, learning_rate)
            # Compute loss
            current_loss = self.loss(params, X, y, self)
            # Check for convergence
            if abs(current_loss - prev_loss) < tol:
                break
            prev_loss = current_loss
        
        # Store the fitted parameters
        self.params_ = params
        if self.fit_intercept:
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            self.coef_ = params
        return self
    
    @jit
    def predict_proba(self, X):
        """
        Predict class probabilities using the logistic regression model.
        
        Args:
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted class probabilities
        """
        if self.params_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.forward(self.params_, X)

    @jit
    def predict(self, X):
        """
        Predict using the logistic regression model.
        
        Args:
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted class labels
        """
        proba = self.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] == 1:
            # For binary classification, return 0 or 1 based on threshold 0.5
            return (proba >= 0.5).astype(jnp.int32)
        else:
            # For multi-class classification, return the index of the max probability
            return jnp.argmax(proba, axis=1).astype(jnp.int32)
    
    def score(self, X, y):
        """
        Return the accuracy of the model on the given data.
        
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            
        Returns:
            float: Accuracy score
        """
        if self.params_ is None:
            raise ValueError("Model has not been fitted yet.")
        y_pred = self.predict(X)
        return jnp.mean(y_pred == y)

        

        

