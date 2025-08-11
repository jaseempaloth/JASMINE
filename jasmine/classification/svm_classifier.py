import jax
import jax.numpy as jnp
import os
import time
from typing import Callable, Optional, Dict, Tuple

@jax.jit
def hinge_loss(y_true: jnp.ndarray,
            y_pred_scores: jnp.ndarray,
            sample_weight: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Computes the standard Hinge Loss for SVM.

    Args:
        y_true: Ground truth labels, must be in the set {-1, 1}.
        y_pred_scores: The raw output scores (logits) from the linear model.
        sample_weight: Weights for each sample. If None, all samples are weighted equally.
    """
    loss = jnp.maximum(0, 1 - y_true * y_pred_scores)
    if sample_weight is not None:
        return jnp.mean(loss * sample_weight)
    return jnp.mean(loss)

@jax.jit
def squared_hinge_loss(y_true: jnp.ndarray,
                       y_pred_scores: jnp.ndarray,
                       sample_weight: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Computes the Squared Hinge Loss for SVM.

    This loss function is differentiable and penalizes outliers more strongly
    than the standard Hinge Loss.

    Args:
        y_true: Ground truth labels, must be in the set {-1, 1}.
        y_pred_scores: The raw output scores (logits) from the linear model.
        sample_weight: Weights for each sample. If None, all samples are weighted equally.
    """
    loss = jnp.maximum(0, 1 - y_true * y_pred_scores) ** 2
    if sample_weight is not None:
        return jnp.mean(loss * sample_weight)
    return jnp.mean(loss)

class SVMClassifier:
    """
    A linear Support Vector Machine (SVM) classifier.
    
    This model uses the Hinge Loss and gradient descent for optimization and
    includes support for class weighting and early stopping.
    """
    def __init__(
            self,
            C: float = 1.0,
            learning_rate: float = 0.01,
            n_epochs: int = 1000,
            class_weight: Optional[Dict] = None,
            loss_function: Callable = hinge_loss
    ):
        """
        Args:
            C: Regularization parameter. The strength of the regularization is
               inversely proportional to C. Must be strictly positive.
            learning_rate: The step size for gradient descent.
            max_iter: The maximum number of passes over the training data.
            class_weight: Weights associated with classes in the form {class_label: weight}.
            loss_function: The loss function to use. Defaults to hinge_loss.
        """
        if C <= 0:
            raise ValueError("Regularization parameter C must be positive.")
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.class_weight = class_weight
        self.loss_function = loss_function
        self.params = None
    
    def __init__(self, n_features: int, key: Optional[jax.random.PRNGKey] = None):
        """
        Initialize model parameters.
        
        Args:
            n_features (int): Number of features in the input data.
            key (jax.random.PRNGKey, optional): Random key for parameter initialization.
        """
        if key is None:
            random_state = int.from_bytes(os.urandom(4), 'big')
            key = jax.random.PRNGKey(random_state)
        w_key, _ = jax.random.split(key)
        params = {
            "w": jax.random.normal(w_key, (n_features,))
        }
        if self.use_bias:
            params["b"] = jnp.array(0.0)        
        return params
    
    @staticmethod
    def forward(params: dict, X: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the SVM classifier.

        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted values
        """
        return X @ params["w"] + params["b"]
    
    def loss_fn(self, params: dict, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the loss for the SVM classifier.
        """
        scores = self.forward(params, X)

        sample_weights = None
        if self.class_weight:
            # Create sample weights based on the class_weight dictionary
            weight_neg = self.class_weight.get(-1, 1.0)
            weight_pos = self.class_weight.get(1, 1.0)
            sample_weights = jnp.where(y == 1, weight_pos, weight_neg)            
            
            # Calculate the data loss using the specified loss function, scaled by C
            data_loss = self.C * self.loss_function(y, scores, sample_weights)

            # Add L2 penalty (weight decay)
            reg_loss = 0.5 * jnp.sum(params["w"] ** 2)

            return data_loss + reg_loss
    
    


        


        
