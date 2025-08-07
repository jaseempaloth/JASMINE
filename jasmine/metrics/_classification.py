import jax
import jax.numpy as jnp


def binary_cross_entropy(y_true, y_pred, from_logits: bool=False):
    """
    Compute the Binary Cross-Entropy loss between true and predicted values.
    
    Args:
        y_true (jnp.ndarray): Ground truth binary labels (0 or 1).
        y_pred (jnp.ndarray): Predicted probabilities (between 0 and 1).
        from_logits (bool): If True, y_pred is expected to be logits. If False, y_pred should be probabilities.
        
    Returns:
        float: Computed Binary Cross-Entropy loss.
    """
    if from_logits:
        # Numerically stable calculation using jax.nn.log_sigmoid.
        # This is equivalent to applying sigmoid and then binary cross-entropy,
        # but prevents numerical issues with very small or large logits.
        log_p = jax.nn.log_sigmoid(y_pred)
        log_not_p = jax.nn.log_sigmoid(-y_pred) # log(1 - sigmoid(x)) = log(sigmoid(-x))
        return -jnp.mean(y_true * log_p + (1. - y_true) * log_not_p)

    # Standard binary cross-entropy with clipping for stability 
    epsilon = 1e-15  # To avoid log(0)
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred, from_logits: bool=False):
    """
    Compute the Categorical Cross-Entropy loss between true and predicted values.
    
    Args:
        y_true (jnp.ndarray): True categorical target values (one-hot encoded).
        y_pred (jnp.ndarray): Predicted probabilities or logits for each class.
        from_logits (bool): If True, y_pred is expected to be a raw logit output.

        
    Returns:
        float: Computed Categorical Cross-Entropy loss.
    """
    if from_logits:
        # Use numerically stable log_softmaxs
        log_probs = jax.nn.log_softmax(y_pred, axis=-1)
    else:
        epsilon = 1e-15  # To avoid log(0)
        y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
        log_probs = jnp.log(y_pred)

    return -jnp.mean(jnp.sum(y_true * log_probs, axis=-1))

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

@jax.jit
def euclidean_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the squared Euclidean distance between a single point and a batch of points.

    Note: We use squared Euclidean distance because it's computationally cheaper
    (avoids the square root) and preserves the ranking of distances, which is
    all that KNN needs to find the nearest neighbors.
    
    Args:
        x1 (jnp.ndarray): A single data point of shape (n_features,).
        x2 (jnp.ndarray): A batch of data points of shape (n_samples, n_features).
        
    Returns:
        jnp.ndarray: Computed Euclidean distance(n_samples,).
    """
    return jnp.sum((x1 - x2) ** 2, axis=1)

@jax.jit
def manhattan_distance(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Manhattan distance between a single point and a batch of points.
    
    Args:
        x1 (jnp.ndarray): A single data point of shape (n_features,).
        x2 (jnp.ndarray): A batch of data points of shape (n_samples, n_features).
        
    Returns:
        jnp.ndarray: Computed Manhattan distance(n_samples,).
    """
    return jnp.sum(jnp.abs(x1 - x2), axis=1)