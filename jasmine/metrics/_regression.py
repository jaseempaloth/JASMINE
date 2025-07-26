import jax.numpy as jnp

def mean_squared_error(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.
    
    Args:
        y_true (jnp.ndarray): True target values.
        y_pred (jnp.ndarray): Predicted target values.
        
    Returns:
        float: Computed MSE value.
    """
    return jnp.mean(jnp.square(y_true - y_pred))

def mean_absolute_error(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE) between true and predicted values.
    
    Args:
        y_true (jnp.ndarray): True target values.
        y_pred (jnp.ndarray): Predicted target values.
        
    Returns:
        float: Computed MAE value.
    """
    return jnp.mean(jnp.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    
    Args:
        y_true (jnp.ndarray): True target values.
        y_pred (jnp.ndarray): Predicted target values.
        
    Returns:
        float: Computed RMSE value.
    """
    return jnp.sqrt(mean_squared_error(y_true, y_pred))