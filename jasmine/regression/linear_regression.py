import jax 
import jax.numpy as jnp
from ..metrics._regression import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

class LinearRegression:
    def __init__(self, use_bias=True, learning_rate=0.01, n_epochs=1000, loss_function=mean_squared_error, l1_penalty=0.0, l2_penalty=0.0):
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
            key = jax.random.PRNGKey(42)

        w_key, _ = jax.random.split(key)

        params = {
            "w" : jax.random.normal(w_key, (n_features,))
        }

        if self.use_bias:
            params["b"] = jnp.array(0.0)
        else:
            params["b"] = None

        self.params = params
        return params

    @staticmethod
    def forward(params, X):
        """
        Forward pass for the linear model.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted values
        """
        use_bias = "b" in params and params["b"] is not None

        if params["w"] is None:
            raise ValueError("Model weights must be initialized before calling forward.")
        
        if use_bias:
            return X @ params["w"] + params["b"]
        else:
            return X @ params["w"]
        
    def loss_fn(self, params, X, y):
        """
        Compute the loss for the linear model.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            
        Returns:
            float: Computed loss value
        """
        # Base loss 
        predictions = self.forward(params, X)
        loss = self.loss_function(y, predictions)

        # Add L2 regularization (Ridge) 
        if self.l2_penalty > 0.0:
            l2_loss = self.l2_penalty * jnp.sum(params["w"] ** 2)
            loss += l2_loss

        # Add L1 regularization (Lasso)
        if self.l1_penalty > 0.0:
            l1_loss = self.l1_penalty * jnp.sum(jnp.abs(params["w"]))
            loss += l1_loss

        return loss

    def train(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            
        Returns:
            dict: Fitted model parameters
        """
        # Initialize parameters
        current_params = self.init_params(X.shape[1])

        # Define the update step as a pure, JIT-compiled function
        @jax.jit
        def update_step(params, X, y):
            # Compute gradients
            grads = jax.grad(self.loss_fn)(params, X, y)

            return jax.tree_util.tree_map(
                lambda p, g: p - self.learning_rate * g, params, grads
            )

        # Training loop
        for epoch in range(self.n_epochs):
            # Get the new, updated parameters from the pure function
            current_params = update_step(current_params, X, y)

            if epoch % 100 == 0:
                loss_value = self.loss_fn(current_params, X, y)
                print(f"Epoch {epoch}, Loss: {loss_value}")
        # Store the final parameters and return the instance
        self.params = current_params
        return self
        
    def inference(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted values
        """
        if self.params is None:
            raise ValueError("Model has not been trained yet. Call `train` before calling `inference`.")
        
        return self.forward(self.params, X)

    def evaluate(self, X, y, metrics_fn=r2_score):
        """
        Evaluate the model using specified metrics.
        
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            metrics (callable): Metric function to evaluate the model
            
        Returns:
            float: Computed metric value
        """
        if self.params is None:
            raise ValueError("Model has not been trained yet. Call `train` before calling `evaluate`.")
        
        predictions = self.inference(X)
        return metrics_fn(y, predictions)
        

        
    

