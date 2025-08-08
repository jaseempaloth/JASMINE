import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional

ScalarParams = Dict[str, jnp.ndarray]

def fit_fn(X: jnp.ndarray, epsilon: float = 1e-8) -> ScalarParams:
    """
    Pure function to compute scaling parameters.
    
    Args:
        X (jnp.ndarray): The data to fit.
        epsilon (float): A small value to prevent division by zero.
        
    Returns:
        A dictionary containing the 'mean' and 'scale' parameters.
    """
    # Check dimensions
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array") 
    
    # Check for empty array
    if X.shape[0] == 0:
        raise ValueError("Input array cannot be empty")
    
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)

    # Handle zero variance features with epsilon for numerical stability
    scale = jnp.maximum(std, epsilon)
     
    return {'mean': mean, 'scale': scale}

@partial(jax.jit)
def transform_fn(X: jnp.ndarray, params: ScalarParams) -> jnp.ndarray:
    """
    Pure function to transform data using the fitted parameters.
    
    Args:
        X (jnp.ndarray): The data to transform.
        params (ScalarParams): The scaling parameters containing 'mean' and 'scale'.
        
    Returns:
        jnp.ndarray: The transformed data.
    """
    return (X - params['mean']) / params['scale']

@partial(jax.jit)
def inverse_transform_fn(X: jnp.ndarray, params: ScalarParams) -> jnp.ndarray:
    """
    Pure function to inverse transform data using the fitted parameters.

    Args:
        X (jnp.ndarray): The data to inverse transform.
        params (ScalarParams): The scaling parameters containing 'mean' and 'scale'.

    Returns:
        jnp.ndarray: The inverse transformed data.
    """
    return (X * params['scale']) + params['mean']

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
        self.params: Optional[ScalarParams] = None

    @property
    def is_fitted(self) -> bool:
        """
        Check if the scaler has been fitted.
        
        Returns:
            bool: True if fitted, False otherwise.
        """
        return self.params is not None
    
    def fit(self, X: jnp.ndarray):
        """
        Fit the scaler to the data.
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            self: Fitted scaler instance.
        """
        self.params = fit_fn(X, self.epsilon)
        return self
    
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Transform the data using the fitted parameters.
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
        Returns:
            jnp.ndarray: Transformed features.
        """
        if not self.is_fitted:
            raise RuntimeError("StandardScaler is not fitted yet. Call 'fit' first.")

        return transform_fn(X, self.params)
     
    def fit_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the scaler and transform the data in one step.
        
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: Transformed features.
        """
        self.fit(X)
        return self.transform(X)
    
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
        
        return inverse_transform_fn(X, self.params)
    
    
    
        

