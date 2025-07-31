import jax
import jax.numpy as jnp

@jax.jit
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
        # This formula is equivalent to sigmoid followed by binary cross-entropy,
        # It is more numerically stable.
        return jnp.mean(jnp.maximum(y_pred, 0) - y_pred * y_true + jnp.log(1 + jnp.exp(-jnp.abs(y_pred))))

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