import jax
import jax.numpy as jnp

@jax.jit
def binary_cross_entropy(y_true, y_pred):
    """
    Compute the Binary Cross-Entropy loss between true and predicted values.
    
    Args:
        y_true (jnp.ndarray): True binary target values (0 or 1).
        y_pred (jnp.ndarray): Predicted probabilities (between 0 and 1).
        
    Returns:
        float: Computed Binary Cross-Entropy loss.
    """
    epsilon = 1e-15  # To avoid log(0)
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    """
    Compute the Categorical Cross-Entropy loss between true and predicted values.
    
    Args:
        y_true (jnp.ndarray): True categorical target values (one-hot encoded).
        y_pred (jnp.ndarray): Predicted probabilities for each class.
        
    Returns:
        float: Computed Categorical Cross-Entropy loss.
    """
    epsilon = 1e-15  # To avoid log(0)
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))

@jax.jit
def accuracy_score(y_true, y_pred):
    """
    Compute the accuracy score between true and predicted values.
    
    Args:
        y_true (jnp.ndarray): True target values.
        y_pred (jnp.ndarray): Predicted target values.
        
    Returns:
        float: Computed accuracy score.
    """
    return jnp.mean(y_true == y_pred)