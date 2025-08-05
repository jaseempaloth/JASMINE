import jax.numpy as jnp
from typing import Optional, Dict, Union

class StandardScaler:
    """
    StandardScaler standardizes features by removing the mean and scaling to unit variance.
    
    Attributes:
        copy (bool): If True, a copy of X will be created; otherwise, it will be modified in place.
        with_mean (bool): If True, center the data before scaling.
        with_std (bool): If True, scale the data to unit variance.
        epsilon (float): Small value to avoid division by zero.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.is_fitted = False
        self.mean = None
        self.scale = None
        self.n_features = None


    def validate_input(self, X: jnp.ndarray, check_features: bool = True) -> jnp.ndarray:
        X = jnp.asarray(X)

        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array") 
        
        # Cehck for empty array
        if X.shape[0] == 0:
            raise ValueError("Input array cannot be empty")
        
        # Check feature consistency (except during initial fit)
        if not check_features and self.is_fitted:
            if X.shape[1] != self.n_features_in:
                raise ValueError(
                    f"X has {X.shape[1]} features, but StandardScaler is expecting "
                    f"{self.n_features_in} features as seen in fit."
                )
            
        return X
    
    def fit(self, X: jnp.ndarray):
        """
        Compute the mean and standard deviation to be used for later scaling.
        
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            self: Fitted scaler instance.
        """
        X = self.validate_input(X, check_features=False)

        self.n_features = X.shape[1]
     
        self.mean = jnp.mean(X, axis=0)
        std = jnp.std(X, axis=0)

        # Handle zero variance features with epsilon for numerical stability
        self.scale = jnp.maximum(std, self.epsilon)

        self.is_fitted = True
        return self
    
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform standardization by centering and scaling.
        
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: Transformed features.
        """
        if not self.is_fitted:
            raise RuntimeError("StandardScaler is not fitted yet. Call 'fit' first.")

        X = self.validate_input(X, check_features=True)
        return (X - self.mean) / self.scale
     
    def fit_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the scaler and transform the data in one step.
        
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: Transformed features.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse transform the standardized data back to original scale.
        
        Args:
            X (jnp.ndarray): Standardized features of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: Original scale features.
        """
        if not self.is_fitted:
            raise RuntimeError("StandardScaler is not fitted yet. Call 'fit' first.")
        
        X = self.validate_input(X, check_features=True)
        return X * self.scale + self.mean
    
    
        

