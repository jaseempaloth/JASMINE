import jax
from jax import jit, grad, vmap
from jax import numpy as jnp
from abc import ABC, abstractmethod
from typing import Dict, Callable, Any

class Loss(ABC):
    """
    Abstract base class for all loss functions.
    """
    @abstractmethod
    def __call__(self, params, X, y, model):
        """
        Compute the loss.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            model (callable): The model function to compute predictions
            
        Returns:
            jnp.ndarray: Computed loss
        """
        pass

    def grad(self, params, X, y, model):
        """
        Compute the gradient of the loss function with respect to the parameters.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            model (callable): The model function to compute predictions
            
        Returns:
            dict: Gradients of the loss function with respect to params
        """
        return self._grad(params, X, y, model)
    
    def __init__(self):
        """
        Initialize the loss function and its gradient.
        """
        self._grad = grad(self.__call__, argnums=0)
    

class MSELoss(Loss):
    """
    Mean Squared Error (MSE) loss.
    MSE = (1/n) * sum((preds - y)^2)
    """
    def __call__(self, params, X, y, model):
        """
        Compute the Mean Squared Error (MSE) loss.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            model (callable): The model function to compute predictions
            
        Returns:
            jnp.ndarray: Computed MSE loss
        """
        preds = model.forward(params, X)
        return jnp.mean(jnp.square(preds - y))


class MAELoss(Loss):
    """
    Mean Absolute Error (MAE) loss.
    MAE = (1/n) * sum(|preds - y|)
    """
    def __call__(self, params, X, y, model):
        """
        Compute the Mean Absolute Error (MAE) loss.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            model (callable): The model function to compute predictions
            
        Returns:
            jnp.ndarray: Computed MAE loss
        """
        preds = model.forward(params, X)
        return jnp.mean(jnp.abs(preds - y))
    

class RMSELoss(Loss):
    """
    Root Mean Squared Error (RMSE) loss.
    RMSE = sqrt((1/n) * sum((preds - y)^2))
    """
    def __call__(self, params, X, y, model):
        """
        Compute the Root Mean Squared Error (RMSE) loss.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            model (callable): The model function to compute predictions
            
        Returns:
            jnp.ndarray: Computed RMSE loss
        """
        preds = model.forward(params, X)
        return jnp.sqrt(jnp.mean(jnp.square(preds - y)))
    
class CrossEntropyLoss(Loss):
    """Cross-Entropy Loss for binary and multi-class classification."""
    @jit
    def __call__(self, params, X, y, model):
        """
        Compute the Cross-Entropy loss.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            model (callable): The model function to compute predictions
            
        Returns:
            jnp.ndarray: Computed Cross-Entropy loss
        """
        preds = model.forward(params, X)
        # Binary classification (1D preds or single output)
        if preds.ndim == 1 or preds.shape[1] == 1:
            logits = jax.nn.sigmoid(preds)
            logits = jnp.clip(logits, 1e-9, 1.0)
            return -jnp.mean(y * jnp.log(logits) + (1 - y) * jnp.log(1 - logits))
        else:
            # Multi-class classification (2D preds)
            logits = jax.nn.softmax(preds)
            logits = jnp.clip(logits, 1e-9, 1.0)
            return -jnp.mean(jnp.sum(y * jnp.log(logits), axis=1))
    
