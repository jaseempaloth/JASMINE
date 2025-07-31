import jax
import jax.numpy as jnp
import jax.nn
from jasmine.metrics import binary_cross_entropy, accuracy_score
import time
import os

class LogisticRegression:
    def __init__(self, use_bias=True, learning_rate=0.01, n_epochs=1000, 
                loss_function=binary_cross_entropy, l1_penalty=0.0, l2_penalty=0.0):
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.loss_function = loss_function
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.params = None
    
    def init_params(self, n_features, key=None):
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
    def forward(params, X):
        """
        Forward pass for the logistic regression model.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
        """
        use_bias = "b" in params and params["b"] is not None
        if params["w"] is None:
            raise ValueError("Model weights must be initialized before calling 'forward'.")
        
        # Calculate the linear combination (logits)
        logits = X @ params["w"]
        if use_bias:
            logits += params["b"]      
        return logits

    def loss_fn(self, params, X, y):
        """
        Compute the loss function.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target labels
            
        Returns:
            jnp.ndarray: Computed loss value
        """
        logits = self.forward(params, X)
        loss = self.loss_function(y, logits, from_logits=True)

        # Add L2 regularization (Ridge) 
        if self.l2_penalty > 0.0:
            l2_loss = self.l2_penalty * jnp.sum(params["w"] ** 2)
            loss += l2_loss

        # Add L1 regularization (Lasso)
        if self.l1_penalty > 0.0:
            l1_loss = self.l1_penalty * jnp.sum(jnp.abs(params["w"]))
            loss += l1_loss

        return loss

    def train(self, X, y, validation_data=None, early_stopping_patience=None, verbose=1):
        """
        Train the logistic regression model.
        
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target labels
            validation_data (tuple, optional): Tuple of (X_val, y_val) for validation
            early_stopping_patience (int, optional): Number of epochs with no improvement to wait before stopping
            verbose (int): Verbosity level
            
        Returns:
            dict: Final model parameters after training
        """
        current_params = self.init_params(X.shape[1])
        history = {"loss": [], "val_loss": []}

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_params = None

        @jax.jit
        def update_step(params, X, y):
            grads = jax.grad(self.loss_fn)(params, X, y)
            return jax.tree_util.tree_map(
                lambda p, g: p - self.learning_rate * g, params, grads
            )

        start_time = time.time()
        for epoch in range(self.n_epochs):
            current_params = update_step(current_params, X, y)
            train_loss = self.loss_fn(current_params, X, y)
            history["loss"].append(train_loss)
            log_msg = f"Epoch {epoch + 1}/{self.n_epochs} - Loss: {train_loss:.4f}"

            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.loss_fn(current_params, X_val, y_val)
                history["val_loss"].append(val_loss)
                log_msg += f" - Val Loss: {val_loss:.4f}"

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
            
            if verbose > 0:
                print(log_msg, end='\r')
        
        if verbose > 0:
            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time:.2f} seconds.")
            
        self.params = best_params if best_params is not None else current_params
        return history
    
    def predict_probabilities(self, X):
        """
        Predict probabilities for the input features.
        
        Args:
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted probabilities
        """
        if self.params is None:
            raise ValueError("Model has not been trained yet. Call `train` before calling `predict_probabilities`.")
        
        logits = self.forward(self.params, X)
        return jax.nn.sigmoid(logits)
    
    def inference(self, X, threshold=0.5):
        """
        Perform inference on the input features.
        
        Args:
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted class labels
        """
        probabilities = self.predict_probabilities(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y, metrics_fn=accuracy_score):
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

        
