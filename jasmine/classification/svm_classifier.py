import jax
import jax.numpy as jnp
import os
from typing import Callable, Optional, Dict, Tuple
from jasmine.metrics import accuracy_score

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
    
    def init_params(self, n_features: int, key: Optional[jax.random.PRNGKey] = None):
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
            "w": jax.random.normal(w_key, (n_features,)),
            "b": jnp.array(0.0)
        }      
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
    
    def train(self, X: jnp.ndarray, y: jnp.ndarray,
              validation_data: Optional[Tuple] = None,
              early_stopping_patience: Optional[int] = None,
              verbose: int = 1):
        """
        Train the SVM classifier.

        Important: The labels `y` must be transformed to {-1, 1}.

        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target labels
            validation_data (tuple, optional): Tuple of (X_val, y_val) for validation
            early_stopping_patience (int, optional): Number of epochs with no improvement to wait before stopping
            verbose (int): Verbosity level

        Returns:
            dict: Fitted model parameters
        """
        if not jnp.all((y == 1) |(y == - 1)):
            raise ValueError("Labels must be in the set {-1, 1}.")
        
        current_params = self.init_params(X.shape[1])
        history = {"loss": [], "val_loss": []}

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_params = None

        @jax.jit
        def update_step(params, X, y):
            grads = jax.grad(self.loss_fn)(params, X, y)
            return jax.tree_util.tree_map(
                lambda p, g: p - self.learning_rate * g, params, grads
            )
        
        print_interval = max(1, self.n_epochs // 10) if verbose > 0 else self.n_epochs
        
        for epoch in range(self.n_epochs):
            current_params = update_step(current_params, X, y)
            
            # Only compute loss when needed for logging or validation
            should_log = verbose > 0 and (epoch + 1) % print_interval == 0
            
            if validation_data is not None or should_log:
                train_loss = self.loss_fn(current_params, X, y)
                history['loss'].append(train_loss)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.loss_fn(current_params, X_val, y_val)
                history['val_loss'].append(val_loss)
            
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_params = current_params
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= early_stopping_patience:
                        if verbose > 0:
                            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                        self.params = best_params
                        return history
            
            if should_log:
                log_msg = f"Epoch {epoch + 1:4d}/{self.n_epochs} - Loss: {train_loss:.6f}"
                if validation_data is not None:
                    log_msg += f" - Val Loss: {val_loss:.6f}"
                print(log_msg, end='\r')

        if verbose > 0:
            print()
        
        self.params = best_params if best_params is not None else current_params
        return history
    
    def inference(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted values
        """
        if self.params is None:
            raise ValueError("Model has not been trained yet. Call `train` before calling `inference`.")
        
        scores = self.forward(self.params, X)
        return jnp.sign(scores).astype(int)
    
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray, metrics_fn=accuracy_score) -> float:
        """
        Evaluate the model using the specified metrics function.
        
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): True labels
            metrics_fn (callable): Metrics function to compute the score
            
        Returns:
            float: Computed metrics score
        """
        if self.params is None:
            raise ValueError("Model has not been trained yet. Call `train` before calling `evaluate`.")
        
        class_predictions = self.inference(X)
        return metrics_fn(y, class_predictions)
    

            




        


        
